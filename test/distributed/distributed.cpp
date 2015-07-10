#include "Halide.h"
#include "mpi.h"
#include <stdarg.h>
#include <stdio.h>

using namespace Halide;

int mpi_printf(const char *format, ...) {
    int rank = 0, rc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        va_list args;
        va_start (args, format);
        rc = vprintf (format, args);
        va_end (args);
        return rc;
    } else {
        return 0;
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Var x, y;
    {
        Image<int> in(20);
        for (int i = 0; i < in.width(); i++) {
            in(i) = 2*i;
        }
        Func f;
        f(x) = in(x) + 1;

        f.compute_root().distribute(x);

        Image<int> im = f.realize(20);
        if (rank == 0) {
            for (int x = 0; x < im.width(); x++) {
                int correct = 2*x + 1;
                if (im(x) != correct) {
                    mpi_printf("im(%d) = %d instead of %d\n", x, im(x), correct);
                    MPI_Finalize();
                    return -1;
                }
            }
        }
    }

    mpi_printf("Success!\n");

    MPI_Finalize();
    return 0;
}
