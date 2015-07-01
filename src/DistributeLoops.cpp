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
    class GetVariablesInExpr : public IRVisitor {
        using IRVisitor::visit;
        void visit(const Variable *var) {
            names.insert(var->name);
        }
    public:
        set<string> names;
    };

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
public:
    using IRMutator::visit;

    void visit(const For *for_loop) {
        if (for_loop->for_type == ForType::Distributed) {
            // Find all input and output buffers in the loop body
            // using the distributed loop variable.
            FindBuffersUsingVariable find(for_loop->name);
            for_loop->body.accept(&find);

            map<string, Box> required, provided;
            required = boxes_required(for_loop->body);
            provided = boxes_provided(for_loop->body);

            // debug(0) << "Input buffers needed for distributed loop " << for_loop->name << ":\n";
            // for (string b : find.inputs) {
            //     debug(0) << b << ": " << box2str(required[b]) << "\n";
            // }
            // debug(0) << "Output buffers needed for distributed loop " << for_loop->name << ":\n";
            // for (string b : find.outputs) {
            //     debug(0) << b << ": " << box2str(provided[b]) << "\n";
            // }

            Var P("P"), p("p");
            Expr slice_size = (for_loop->extent - for_loop->min) / P;
            Expr newmin = slice_size*p, newmax = slice_size*(p+1);
            Expr newextent = simplify(newmax - newmin);
            // debug(0) << "  new bounds for " << for_loop->name << ": "
            //          << newmin << " to " << newmax << "\n";

            Stmt chunkedloop = For::make(for_loop->name, newmin, newextent,
                                         ForType::Serial, for_loop->device_api,
                                         for_loop->body);
            Expr mpi_numprocs = Call::make(Int(32), "halide_do_distr_size", {}, Call::Extern);
            Expr mpi_rank = Call::make(Int(32), "halide_do_distr_rank", {}, Call::Extern);
            stmt = LetStmt::make(P.name(), mpi_numprocs,
                                 LetStmt::make(p.name(), mpi_rank, chunkedloop));
        } else {
            IRMutator::visit(for_loop);
        }
    }

};

Stmt distribute_loops(Stmt s) {
    return DistributeLoops().mutate(s);
}

}
}
