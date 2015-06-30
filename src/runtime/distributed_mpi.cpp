#include "runtime_internal.h"

#include "HalideRuntime.h"

typedef int (*halide_task)(void *user_context, int, uint8_t *);

extern "C" {
extern int printf(const char *format, ...);;

WEAK int halide_do_task(void *user_context, halide_task f, int idx,
                        uint8_t *closure);

} // extern "C"

namespace Halide { namespace Runtime { namespace Internal {

WEAK int default_do_task(void *user_context, halide_task f, int idx,
                        uint8_t *closure) {
    return f(user_context, idx, closure);
}

WEAK int default_do_distr_for(void *user_context, halide_task f,
                            int min, int size, uint8_t *closure) {
    for (int x = min; x < min + size; x++) {
        int result = halide_do_task(user_context, f, x, closure);
        if (result) {
            return result;
        }
    }
    printf(">>> after do distr for\n");
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

} // extern "C"
