#include <algorithm>
#include <set>
#include <sstream>

#include "Bounds.h"
#include "Parameter.h"
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
#include "Image.h"

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

// Helper class that wraps information common to Buffer and Parameter classes.
// Also provides a wrapper for Provide nodes which do not have buffer references.
class AbstractBuffer {
public:
    AbstractBuffer(const Buffer &buffer) {
        _type = buffer.type();
        _name = buffer.name();
        _dimensions = buffer.dimensions();
        for (int d = 0; d < buffer.dimensions(); d++) {
            mins.push_back(buffer.min(d));
            extents.push_back(buffer.extent(d));
            strides.push_back(buffer.stride(d));
        }
    }

    AbstractBuffer(const Parameter &param) {
        _type = param.type();
        _name = param.name();
        _dimensions = param.dimensions();
        for (int d = 0; d < param.dimensions(); d++) {
            mins.push_back(param.min_constraint(d));
            extents.push_back(param.extent_constraint(d));
            strides.push_back(param.stride_constraint(d));
        }
    }

    AbstractBuffer(const Provide *provide) {
        _type = provide->values[0].type();
        _name = provide->name;
        _dimensions = -1;
    }

    int dimensions() const {
        internal_assert(_dimensions >= 0) << "Called dimensions on AbstractBuffer of Provide type.\n";
        return _dimensions;
    }

    Expr extent(int dim) const {
        return extents[dim];
    }

    Expr min(int dim) const {
        return mins[dim];
    }

    Expr stride(int dim) const {
        return strides[dim];
    }

    Type type() const {
        return _type;
    }

    const string &name() const {
        return _name;
    }

    Expr elem_size() const {
        return _type.bytes();
    }
private:
    string _name;
    Type _type;
    int _dimensions;
    vector<Expr> mins;
    vector<Expr> extents;
    vector<Expr> strides;
};
}

class DistributeLoops : public IRMutator {
    // Build a set of all the variables referenced.
    class GetVariablesInExpr : public IRVisitor {
        using IRVisitor::visit;
        void visit(const Variable *var) {
            names.insert(var->name);
            IRVisitor::visit(var);
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
                internal_assert(call->call_type == Call::Image);
                if (call->image.defined()) {
                    inputs.push_back(AbstractBuffer(call->image));
                } else {
                    inputs.push_back(AbstractBuffer(call->param));
                }

            }
            IRVisitor::visit(call);
        }

        void visit(const Provide *provide) {
            GetVariablesInExpr vars;
            for (Expr arg : provide->args) {
                arg.accept(&vars);
            }
            if (vars.names.count(name)) {
                internal_assert(provide->values.size() == 1);
                outputs.push_back(AbstractBuffer(provide));
            }
            IRVisitor::visit(provide);
        }

        map<string, Expr> elem_sizes;
    public:
        string name;
        vector<AbstractBuffer> inputs, outputs;
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
        Expr slice_size = for_loop->extent / num_processors();
        Expr newmin = slice_size*rank(), newmax = slice_size*(rank()+1);
        Expr newextent = simplify(newmax - newmin);
        // TODO: choose correct loop type here.
        Stmt newloop = For::make(for_loop->name, newmin, newextent,
                                 ForType::Serial, for_loop->device_api,
                                 for_loop->body);
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

        // Split original loop into chunks of iterations for each rank.
        Stmt newloop = distribute_loop_iterations(for_loop);

        // Get required regions of input buffers for new loop.
        map<string, Box> required, provided;
        required = boxes_required(newloop);
        provided = boxes_provided(newloop);
        for (const AbstractBuffer &in : find.inputs) {
            Box b = required[in.name()];
        }

        // Send required regions of input buffers.
        Box b = required[find.inputs[0].name()];
        Expr rowsize = b[0].max - b[0].min + 1;
        Expr elemsize = find.inputs[0].elem_size();
        Expr bytes = rowsize * elemsize;

        Stmt p = Evaluate::make(print({string("Rank"), rank(), string(", rowsize:"), rowsize,
                        string(", #bytes="), bytes}));
        stmt = Block::make(p, newloop);

        // // Get address of buffer
        // Expr first_elem = Load::make(UInt(8), find.outputs[0], 0, Buffer(), Parameter());
        // Expr buf = Call::make(Handle(), Call::address_of, {first_elem}, Call::Intrinsic);
        // // Get number of bytes in buffer
        // Expr buffer_min = Call::make(UInt(8), Call::extract_buffer_min, {0}
        // Expr send = Call::make(Int(32), "halide_do_distr_send", {buf, count, 0}, Call::Extern);
        // Expr maybesend = IfThenElse::make(condition, send);

        //stmt = newloop;
    }

};

Stmt distribute_loops(Stmt s) {
    return DistributeLoops().mutate(s);
}

}
}
