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
Stmt distribute_loops(Stmt s, const std::map<std::string, Function> &env);
Stmt inject_communication(Stmt s, const std::map<std::string, Function> &env);

Stmt distribute_loops_only(Stmt s, const std::map<std::string, Function> &env, bool cap_extents);

void distribute_loops_test();

}
}

#endif
