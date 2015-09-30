#ifndef HALIDE_DISTRIBUTE_LOOPS_H
#define HALIDE_DISTRIBUTE_LOOPS_H

/** \file
 * Defines the lowering pass that distributes loops marked as such
 */

#include "IR.h"

namespace Halide {
namespace Internal {

/** Take a statement with for loops marked for distribution, and turn
 * them into loops that operate on a subset of their input data
 * according to their MPI rank.
 */
Stmt distribute_loops(Stmt s, const std::map<std::string, Function> &env, const FuncValueBounds &func_bounds, bool cap_extents=false);

/** Inject communication calls to gather required distributed data
 * onto the current rank.
 */
Stmt inject_communication(Stmt s, const std::map<std::string, Function> &env);

/** Change "distribute" loop types into their corresponding
 * non-distributed types. */
Stmt change_distributed_annotation(Stmt s);

void distribute_loops_test();

}
}

#endif
