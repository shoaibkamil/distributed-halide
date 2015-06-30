#include <algorithm>

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

class DistributeLoops : public IRMutator {

    using IRMutator::visit;

    void visit(const For *for_loop) {
        if (for_loop->for_type == ForType::Distributed) {
            stmt = For::make(for_loop->name, for_loop->min, for_loop->extent,
                             ForType::Serial, for_loop->device_api, for_loop->body);
            // The for loop becomes a simple let statement
            //stmt = LetStmt::make(for_loop->name, for_loop->min, body);
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
