#include "Halide.h"
#include "mpi.h"
#include <iomanip>
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

    // {
    //     // Cropping test. Really this should go into DistributeLoops.cpp.
    //     Image<int> in(10, 10);
    //     // Crop region:
    //     const int xoff = 4, yoff = 4, xsize = 3, ysize = 2;

    //     for (int y = 0; y < in.height(); y++) {
    //         for (int x = 0; x < in.width(); x++) {
    //             in(x, y) = x + y;
    //         }
    //     }

    //     uint8_t *data = (uint8_t *)&in(xoff, yoff);
    //     uint8_t *dest = new uint8_t[xsize * ysize * sizeof(int)];

    //     buffer_t ct = *in.raw_buffer();
    //     ct.host = dest;
    //     ct.extent[0] = xsize;
    //     ct.extent[1] = ysize;
    //     ct.stride[0] = 1;
    //     ct.stride[1] = xsize;
    //     Buffer cb(Int(32), &ct);
    //     Image<int> cropped(cb);

    //     for (int y = 0; y < ysize; y++) {
    //         uint8_t *dptr = dest + y * cropped.stride(1) * sizeof(int),
    //             *sptr = data + y * in.stride(1) * sizeof(int);
    //         unsigned nbytes = xsize * sizeof(int);
    //         memcpy(dptr, sptr, nbytes);
    //     }

    //     for (int y = 0; y < ysize; y++) {
    //         for (int x = 0; x < xsize; x++) {
    //             // int idx = (x + y * ysize) * sizeof(int);
    //             // int cval = *(int *)(dest + idx);
    //             int cval = cropped(x, y);
    //             int correct = in(xoff + x, yoff + y);
    //             if (cval != correct) {
    //                 mpi_printf("cropped(%d,%d) = %d instead of %d\n", x, y, cval, correct);
    //                 MPI_Finalize();
    //                 return -1;
    //             }
    //         }
    //     }

    //     delete[] dest;
    // }

    {
        Image<int> in(20);
        for (int i = 0; i < in.width(); i++) {
            in(i) = 2*i;
        }
        Func f;
        f(x) = in(x) + 1;
        f.compute_root().distribute(x);

        Image<int> out = f.realize(20);
        if (rank == 0) {
            for (int x = 0; x < out.width(); x++) {
                int correct = 2*x + 1;
                if (out(x) != correct) {
                    mpi_printf("out(%d) = %d instead of %d\n", x, out(x), correct);
                    MPI_Finalize();
                    return -1;
                }
            }
        }
    }

    {
        Image<int> in(10, 20);
        for (int y = 0; y < in.height(); y++) {
            for (int x = 0; x < in.width(); x++) {
                in(x, y) = x + y;
            }
        }

        Func f;
        f(x, y) = 2 * in(x, y);
        f.distribute(y);
        Image<int> out = f.realize(10, 20);
        if (rank == 0) {
            for (int y = 0; y < out.height(); y++) {
                for (int x = 0; x < out.width(); x++) {
                    int correct = 2*(x+y);
                    if (out(x,y) != correct) {
                        mpi_printf("out(%d,%d) = %d instead of %d\n", x, y,out(x,y), correct);
                        MPI_Finalize();
                        return -1;
                    }
                }
            }
        }
    }

    mpi_printf("Success!\n");

    MPI_Finalize();
    return 0;
}
