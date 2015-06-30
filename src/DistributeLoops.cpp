#include <algorithm>
#include <set>

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

namespace Halide {
namespace Internal {

using std::string;
using std::vector;
using std::set;

class DistributeLoops : public IRMutator {
    class GetVariablesInExpr : public IRVisitor {
        using IRVisitor::visit;
        void visit(const Variable *var) {
            names.insert(var->name);
        }
    public:
        set<string> names;
    };

    class FindInputBuffers : public IRVisitor {
        using IRVisitor::visit;
        void visit(const Call *call) {
            GetVariablesInExpr vars;
            for (Expr arg : call->args) {
                arg.accept(&vars);
            }
            if (vars.names.count(name)) {
                buffers.push_back(call->name);
            }
        }
    public:
        string name;
        vector<string> buffers;
        FindInputBuffers(string n) : name(n) {}
    };
public:
    using IRMutator::visit;

    void visit(const For *for_loop) {
        if (for_loop->for_type == ForType::Distributed) {
            // Find all input buffers in the loop body using the
            // distributed loop variable.
            FindInputBuffers inputs(for_loop->name);
            for_loop->body.accept(&inputs);

            stmt = For::make(for_loop->name, for_loop->min, for_loop->extent,
                             ForType::Serial, for_loop->device_api, for_loop->body);
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
