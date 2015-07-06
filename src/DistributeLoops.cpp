#include <algorithm>
#include <set>
#include <sstream>

#include "Bounds.h"
#include "DistributeLoops.h"
#include "IRMutator.h"
#include "Scope.h"
#include "IRPrinter.h"
#include "Deinterleave.h"
#include "Substitute.h"
#include "IROperator.h"
#include "IREquality.h"
#include "ExprUsesVar.h"
#include "Simplify.h"
#include "Var.h"

namespace Halide {
namespace Internal {

using std::string;
using std::vector;
using std::set;
using std::map;

namespace {
// Return a string representation of the given box.
string box2str(const Box &b) {
    std::stringstream mins, maxs;
    mins << "(";
    maxs << "(";
    for (unsigned i = 0; i < b.size(); i++) {
        mins << simplify(b[i].min);
        maxs << simplify(b[i].max);
        if (i < b.size() - 1) {
            mins << ", ";
            maxs << ", ";
        }
    }
    mins << ")";
    maxs << ")";
    return mins.str() + " to " + maxs.str();
}
}

class DistributeLoops : public IRMutator {
    // Build a set of all the variables referenced.
    class GetVariablesInExpr : public IRVisitor {
        using IRVisitor::visit;
        void visit(const Variable *var) {
            names.insert(var->name);
        }
    public:
        set<string> names;
    };

    // Build a list of all input and output buffers using a particular
    // variable as an index.
    class FindBuffersUsingVariable : public IRVisitor {
        using IRVisitor::visit;
        void visit(const Call *call) {
            GetVariablesInExpr vars;
            for (Expr arg : call->args) {
                arg.accept(&vars);
            }
            if (vars.names.count(name)) {
                inputs.push_back(call->name);
            }
        }

        void visit(const Provide *provide) {
            GetVariablesInExpr vars;
            for (Expr arg : provide->args) {
                arg.accept(&vars);
            }
            if (vars.names.count(name)) {
                outputs.push_back(provide->name);
            }
        }
    public:
        string name;
        vector<string> inputs, outputs;
        FindBuffersUsingVariable(string n) : name(n) {}
    };

    inline Expr num_processors() const {
        return Call::make(Int(32), "halide_do_distr_size", {}, Call::Extern);
    }

    inline Expr rank() const {
        return Call::make(Int(32), "halide_do_distr_rank", {}, Call::Extern);
    }

    // Return a new loop that has iterations determined by processor
    // rank.
    Stmt distribute_loop_iterations(const For *for_loop) const {
        Var P("P"), p("p");
        Expr slice_size = (for_loop->extent - for_loop->min) / P;
        Expr newmin = slice_size*p, newmax = slice_size*(p+1);
        Expr newextent = simplify(newmax - newmin);
        // TODO: choose correct loop type here.
        Stmt newloop = For::make(for_loop->name, newmin, newextent,
                                 ForType::Serial, for_loop->device_api,
                                 for_loop->body);
        newloop = LetStmt::make(P.name(), num_processors(),
                                LetStmt::make(p.name(), rank(), newloop));
        return newloop;
    }

public:
    using IRMutator::visit;

    void visit(const For *for_loop) {
        if (for_loop->for_type != ForType::Distributed) {
            IRMutator::visit(for_loop);
            return;
        }
        // Find all input and output buffers in the loop body
        // using the distributed loop variable.
        FindBuffersUsingVariable find(for_loop->name);
        for_loop->body.accept(&find);

        map<string, Box> required, provided;
        required = boxes_required(for_loop->body);
        provided = boxes_provided(for_loop->body);

        // Split original loop into chunks of iterations for each rank.
        Stmt newloop = distribute_loop_iterations(for_loop);

        // // Send result back to root processor
        // Expr condition = NE::make(p, 0);

        // // Get address of buffer
        // Expr first_elem = Load::make(UInt(8), find.outputs[0], 0, Buffer(), Parameter());
        // Expr buf = Call::make(Handle(), Call::address_of, {first_elem}, Call::Intrinsic);
        // // Get number of bytes in buffer
        // Expr buffer_min = Call::make(UInt(8), Call::extract_buffer_min, {0}
        // Expr send = Call::make(Int(32), "halide_do_distr_send", {buf, count, 0}, Call::Extern);
        // Expr maybesend = IfThenElse::make(condition, send);

        stmt = newloop;
    }

};

Stmt distribute_loops(Stmt s) {
    return DistributeLoops().mutate(s);
}

}
}
