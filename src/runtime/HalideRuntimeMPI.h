#ifndef HALIDE_HALIDERUNTIMEMPI_H
#define HALIDE_HALIDERUNTIMEMPI_H

#include "HalideRuntime.h"

#ifdef __cplusplus
extern "C" {
#endif

/** \file
 *  Routines specific to the Halide MPI runtime.
 */

extern int halide_do_distr_size();
extern int halide_do_distr_rank();

extern int halide_do_distr_send(const void *buf, halide_type_code_t type_code, int type_bits,
                                int count, int dest);
extern int halide_do_distr_isend(const void *buf, halide_type_code_t type_code, int type_bits,
                                 int count, int dest);
extern int halide_do_distr_isend_subarray(const void *buf, halide_type_code_t type_code,
                                          int type_bits, int ndims, int *sizes,
                                          int *subsizes, int *starts, int dest);

extern int halide_do_distr_recv(void *buf, halide_type_code_t type_code, int type_bits,
                                int count, int source);
extern int halide_do_distr_irecv(void *buf, halide_type_code_t type_code, int type_bits,
                                 int count, int source);
extern int halide_do_distr_irecv_subarray(void *buf, halide_type_code_t type_code,
                                          int type_bits, int ndims, int *sizes,
                                          int *subsizes, int *starts, int source);

extern int halide_do_distr_waitall_recvs(void *p);
extern int halide_do_distr_waitall_sends(void *p);
extern uint64_t halide_distr_time_ns(void *user_context, int id);

#ifdef __cplusplus
} // End extern "C"
#endif

#endif // HALIDE_HALIDERUNTIMEMPI_H
