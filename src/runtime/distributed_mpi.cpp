#include "runtime_internal.h"

#include "HalideRuntime.h"

typedef int (*halide_task)(void *user_context, int, uint8_t *);

extern "C" {
extern int printf(const char *format, ...);
extern double ceil(double d);

typedef int MPI_Comm;
#define MPI_COMM_WORLD ((MPI_Comm)0x44000000)

typedef int MPI_Request;
#define MPI_REQUEST_NULL   ((MPI_Request)0x2c000000)

typedef int MPI_Datatype;
#define MPI_CHAR           ((MPI_Datatype)0x4c000101)
#define MPI_SIGNED_CHAR    ((MPI_Datatype)0x4c000118)
#define MPI_UNSIGNED_CHAR  ((MPI_Datatype)0x4c000102)
#define MPI_BYTE           ((MPI_Datatype)0x4c00010d)
#define MPI_WCHAR          ((MPI_Datatype)0x4c00040e)
#define MPI_SHORT          ((MPI_Datatype)0x4c000203)
#define MPI_UNSIGNED_SHORT ((MPI_Datatype)0x4c000204)
#define MPI_INT            ((MPI_Datatype)0x4c000405)
#define MPI_UNSIGNED       ((MPI_Datatype)0x4c000406)
#define MPI_LONG           ((MPI_Datatype)0x4c000807)
#define MPI_UNSIGNED_LONG  ((MPI_Datatype)0x4c000808)
#define MPI_FLOAT          ((MPI_Datatype)0x4c00040a)
#define MPI_DOUBLE         ((MPI_Datatype)0x4c00080b)
#define MPI_LONG_DOUBLE    ((MPI_Datatype)0x4c00100c)
#define MPI_LONG_LONG_INT  ((MPI_Datatype)0x4c000809)
#define MPI_UNSIGNED_LONG_LONG ((MPI_Datatype)0x4c000819)
#define MPI_LONG_LONG      MPI_LONG_LONG_INT

typedef struct MPI_Status {
    int count_lo;
    int count_hi_and_cancelled;
    int MPI_SOURCE;
    int MPI_TAG;
    int MPI_ERROR;
} MPI_Status;

extern int MPI_Comm_size(MPI_Comm, int *);
extern int MPI_Comm_rank(MPI_Comm, int *);
extern int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm);
extern int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                    MPI_Comm comm);
extern int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                     MPI_Comm comm, MPI_Request *request);
extern int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
                    MPI_Comm comm, MPI_Status *status);
MPI_Comm HALIDE_MPI_COMM;
MPI_Request HALIDE_MPI_REQ;

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

WEAK int halide_do_distr_send(const void *buf, int count, int dest) {
    if (!halide_mpi_initialized) {
        halide_initialize_mpi();
    }
    int rank = 0;
    MPI_Comm_rank(HALIDE_MPI_COMM, &rank);

    int tag = 0;
    printf("[rank %d] Issuing send buf %p, count %d, dest %d\n",
           rank, buf, count, dest);
    printf("[rank %d] Sending %u\n", rank, *(unsigned *)buf);
    return MPI_Isend(buf, count, MPI_UNSIGNED_CHAR, dest, tag, HALIDE_MPI_COMM, &HALIDE_MPI_REQ);
    //return MPI_Send(buf, count, MPI_UNSIGNED_CHAR, dest, tag, HALIDE_MPI_COMM);
}

WEAK int halide_do_distr_recv(void *buf, int count, int source) {
    if (!halide_mpi_initialized) {
        halide_initialize_mpi();
    }
    int rank = 0;
    MPI_Comm_rank(HALIDE_MPI_COMM, &rank);

    int tag = 0;
    MPI_Status status;
    printf("[rank %d] Issuing recv buf %p, count %d, source %d\n",
           rank, buf, count, source);
    int retval = MPI_Recv(buf, count, MPI_UNSIGNED_CHAR, source, tag, HALIDE_MPI_COMM, &status);
    printf("[rank %d] Received %u\n", rank, *(unsigned *)buf);
    return retval;
}

} // extern "C"
