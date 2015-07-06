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

    Var x, y;
    {
        Image<int> in(20);
        for (int i = 0; i < in.width(); i++) {
            in(i) = i;
        }
        Func f;
        f(x, y) = in(x) + in(y);

        f.compute_root().distribute(y);

        Image<int> im = f.realize(10, 20);
        for (int y = 0; y < im.height(); y++) {
            for (int x = 0; x < im.width(); x++) {
                int correct = x + y;
                if (im(x, y) != correct) {
                    mpi_printf("im(%d, %d) = %d instead of %d\n", x, y, im(x, y), correct);
                    return -1;
                }
            }
        }
    }

    mpi_printf("Success!\n");

    MPI_Finalize();
    return 0;
}
