#include "runtime_internal.h"

#include "HalideRuntime.h"

typedef int (*halide_task)(void *user_context, int, uint8_t *);

extern "C" {
extern int printf(const char *format, ...);
extern double ceil(double d);

typedef int MPI_Comm;
extern int MPI_Comm_size(MPI_Comm, int *);
extern int MPI_Comm_rank(MPI_Comm, int *);
extern int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm);
#define MPI_COMM_WORLD ((MPI_Comm)0x44000000)
MPI_Comm HALIDE_MPI_COMM;

WEAK int halide_do_task(void *user_context, halide_task f, int idx,
                        uint8_t *closure);

} // extern "C"

namespace Halide { namespace Runtime { namespace Internal {

WEAK int halide_num_processes;
WEAK bool halide_mpi_initialized = false;

WEAK void halide_initialize_mpi() {
    MPI_Comm_dup(MPI_COMM_WORLD, &HALIDE_MPI_COMM);
    MPI_Comm_size(HALIDE_MPI_COMM, &halide_num_processes);
    halide_mpi_initialized = true;
}

WEAK int default_do_task(void *user_context, halide_task f, int idx,
                        uint8_t *closure) {
    return f(user_context, idx, closure);
}

WEAK int default_do_distr_for(void *user_context, halide_task f,
                            int min, int size, uint8_t *closure) {
    if (!halide_mpi_initialized) {
        halide_initialize_mpi();
    }
    int rank = 0;
    MPI_Comm_rank(HALIDE_MPI_COMM, &rank);
    int b = (int)ceil((double)size / halide_num_processes);
    int start = min + b*rank,
        finish = min + b*(rank+1);
    finish = finish <= size ? finish : size;
    for (int x = start; x < finish; x++) {
        int result = halide_do_task(user_context, f, x, closure);
        if (result) {
            return result;
        }
    }
    printf("After distr_for on rank %d over (%d,%d)\n", rank, start, finish);
    // Return zero if the job succeeded, otherwise return the exit
    // status.
    return 0;
}

WEAK int (*halide_custom_do_task)(void *user_context, halide_task, int, uint8_t *) = default_do_task;
WEAK int (*halide_custom_do_distr_for)(void *, halide_task, int, int, uint8_t *) = default_do_distr_for;

}}} // namespace Halide::Runtime::Internal

extern "C" {

WEAK int (*halide_set_custom_do_task(int (*f)(void *, halide_task, int, uint8_t *)))
          (void *, halide_task, int, uint8_t *) {
    int (*result)(void *, halide_task, int, uint8_t *) = halide_custom_do_task;
    halide_custom_do_task = f;
    return result;
}


WEAK int (*halide_set_custom_do_distr_for(int (*f)(void *, halide_task, int, int, uint8_t *)))
          (void *, halide_task, int, int, uint8_t *) {
    int (*result)(void *, halide_task, int, int, uint8_t *) = halide_custom_do_distr_for;
    halide_custom_do_distr_for = f;
    return result;
}

WEAK int halide_do_task(void *user_context, halide_task f, int idx,
                        uint8_t *closure) {
    return (*halide_custom_do_task)(user_context, f, idx, closure);
}

WEAK int halide_do_distr_for(void *user_context, int (*f)(void *, int, uint8_t *),
                           int min, int size, uint8_t *closure) {
  return (*halide_custom_do_distr_for)(user_context, f, min, size, closure);
}

WEAK int halide_do_distr_size() {
    if (!halide_mpi_initialized) {
        halide_initialize_mpi();
    }
    return halide_num_processes;
}

WEAK int halide_do_distr_rank() {
    if (!halide_mpi_initialized) {
        halide_initialize_mpi();
    }
    int rank = 0;
    MPI_Comm_rank(HALIDE_MPI_COMM, &rank);
    return rank;
}


} // extern "C"
