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
Stmt distribute_loops(Stmt);
Stmt distribute_loops_only(Stmt s);

void distribute_loops_test();

}
}

#endif
