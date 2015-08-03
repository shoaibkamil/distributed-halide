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

    Var x, y, z;

    {
        DistributedImage<int> in(20);
        in.set_domain(x);
        in.placement().distribute(x);
        in.allocate();

        for (int x = 0; x < in.global_width(); x++) {
            if (!in.mine(x)) {
                continue;
            }
            in(x) = 2 * in.global(x);
        }

        Func f;
        f(x) = in(x) + 1;
        f.distribute(x);

        DistributedImage<int> out(20);
        out.set_domain(x);
        out.placement().distribute(x);
        out.allocate();
        f.realize(out.get_buffer());
        for (int x = 0; x < out.global_width(); x++) {
            if (!out.mine(x)) continue;
            int correct = 2 * out.global(x) + 1;
            if (out(x) != correct) {
                mpi_printf("out(%d) = %d instead of %d\n", x, out(x), correct);
                MPI_Finalize();
                return -1;
            }
        }
    }

    // {
    //     DistributedImage<int> in(20);
    //     in.set_domain(x);
    //     in.placement().distribute(x);
    //     in.allocate();

    //     for (int x = 0; x < in.width(); x++) {
    //         in(x) = 2 * in.global(x);
    //     }

    //     Expr clamped_x = clamp(x, 0, in.global_width()-1);
    //     Func clamped;
    //     clamped(x) = in(clamped_x);
    //     Func f;
    //     f(x) = clamped(x) + clamped(x+1) + 1;
    //     f.distribute(x);

    //     DistributedImage<int> out(20);
    //     out.set_domain(x);
    //     out.placement().distribute(x);
    //     out.allocate();
    //     f.realize(out.get_buffer());
    //     for (int x = 0; x < out.width(); x++) {
    //         int x1 = x, x2 = x == out.global_width() - 1 ? x : x+1;
    //         int correct = 2*out.global(x1) + 2*out.global(x2) + 1;
    //         if (out(x) != correct) {
    //             mpi_printf("out(%d) = %d instead of %d\n", x, out(x), correct);
    //             MPI_Finalize();
    //             return -1;
    //         }
    //     }
    // }

    // {
    //     Image<int> in(20);
    //     for (int i = 0; i < in.width(); i++) {
    //         in(i) = 2*i;
    //     }
    //     Func f;
    //     f(x) = in(x) + 1;
    //     f.compute_root().distribute(x);

    //     Image<int> out = f.realize(20);
    //     if (rank == 0) {
    //         for (int x = 0; x < out.width(); x++) {
    //             int correct = 2*x + 1;
    //             if (out(x) != correct) {
    //                 mpi_printf("out(%d) = %d instead of %d\n", x, out(x), correct);
    //                 MPI_Finalize();
    //                 return -1;
    //             }
    //         }
    //     }
    // }

    // {
    //     Image<int> in(10, 20);
    //     for (int y = 0; y < in.height(); y++) {
    //         for (int x = 0; x < in.width(); x++) {
    //             in(x, y) = x + y;
    //         }
    //     }

    //     Func f;
    //     f(x, y) = 2 * in(x, y);
    //     f.distribute(y);
    //     Image<int> out = f.realize(10, 20);
    //     if (rank == 0) {
    //         for (int y = 0; y < out.height(); y++) {
    //             for (int x = 0; x < out.width(); x++) {
    //                 int correct = 2*(x+y);
    //                 if (out(x,y) != correct) {
    //                     mpi_printf("out(%d,%d) = %d instead of %d\n", x, y,out(x,y), correct);
    //                     MPI_Finalize();
    //                     return -1;
    //                 }
    //             }
    //         }
    //     }
    // }

    // {
    //     Image<int> in(10, 20, 30);

    //     for (int z = 0; z < in.channels(); z++) {
    //         for (int y = 0; y < in.height(); y++) {
    //             for (int x = 0; x < in.width(); x++) {
    //                 in(x, y, z) = x + y + z;
    //             }
    //         }
    //     }

    //     Func f, g;
    //     f(x, y, z) = 2 * in(x, y, z);
    //     g(x, y, z) = 2 * in(x, y, z);
    //     f.distribute(z);
    //     g.distribute(y);
    //     Image<int> fout = f.realize(10, 20, 30);
    //     Image<int> gout = g.realize(10, 20, 30);
    //     if (rank == 0) {
    //         for (int z = 0; z < fout.channels(); z++) {
    //             for (int y = 0; y < fout.height(); y++) {
    //                 for (int x = 0; x < fout.width(); x++) {
    //                     int correct = 2*(x+y+z);
    //                     if (fout(x,y,z) != correct || gout(x,y,z) != correct) {
    //                         mpi_printf("out(%d,%d,%d) = %d instead of %d\n", x, y, z, fout(x,y,z), correct);
    //                         MPI_Finalize();
    //                         return -1;
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

    // {
    //     Image<int> in(10, 20, 30);

    //     for (int z = 0; z < in.channels(); z++) {
    //         for (int y = 0; y < in.height(); y++) {
    //             for (int x = 0; x < in.width(); x++) {
    //                 in(x, y, z) = x + y + z;
    //             }
    //         }
    //     }

    //     Func f, g;
    //     f(x, y, z) = 2 * in(x, y, z);
    //     g(x, y, z) = 2 * f(x, y, z);
    //     f.compute_root();
    //     g.distribute(z);

    //     Image<int> out = g.realize(10, 20, 30);
    //     if (rank == 0) {
    //         for (int z = 0; z < out.channels(); z++) {
    //             for (int y = 0; y < out.height(); y++) {
    //                 for (int x = 0; x < out.width(); x++) {
    //                     int correct = 4*(x+y+z);
    //                     if (out(x,y,z) != correct) {
    //                         mpi_printf("out(%d,%d,%d) = %d instead of %d\n", x, y, z, out(x,y,z), correct);
    //                         MPI_Finalize();
    //                         return -1;
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

    // {
    //     Image<int> in(10, 20);
    //     for (int y = 0; y < in.height(); y++) {
    //         for (int x = 0; x < in.width(); x++) {
    //             in(x, y) = x + y;
    //         }
    //     }

    //     Func f, g;
    //     f(x, y) = 2 * in(x, y);
    //     g(x, y) = 2 * f(x, y);

    //     f.compute_root().distribute(y);
    //     g.distribute(y);

    //     Image<int> out = g.realize(10, 20);
    //     if (rank == 0) {
    //         for (int y = 0; y < out.height(); y++) {
    //             for (int x = 0; x < out.width(); x++) {
    //                 int correct = 4*(x+y);
    //                 if (out(x,y) != correct) {
    //                     mpi_printf("out(%d,%d) = %d instead of %d\n", x, y,out(x,y), correct);
    //                     MPI_Finalize();
    //                     return -1;
    //                 }
    //             }
    //         }
    //     }
    // }

    // {
    //     Image<int> in(10, 20);
    //     for (int y = 0; y < in.height(); y++) {
    //         for (int x = 0; x < in.width(); x++) {
    //             in(x, y) = x + y;
    //         }
    //     }

    //     Func f, g, h, i;
    //     f(x, y) = 2 * in(x, y);
    //     g(x, y) = 2 * f(x, y);
    //     h(x, y) = 2 * f(x, y);
    //     i(x, y) = g(x, y) + h(x, y);

    //     f.compute_root().distribute(y);
    //     g.compute_root().distribute(y);

    //     Image<int> out = i.realize(10, 20);
    //     if (rank == 0) {
    //         for (int y = 0; y < out.height(); y++) {
    //             for (int x = 0; x < out.width(); x++) {
    //                 int correct = 4*(x+y) + 4*(x+y);
    //                 if (out(x,y) != correct) {
    //                     mpi_printf("out(%d,%d) = %d instead of %d\n", x, y,out(x,y), correct);
    //                     MPI_Finalize();
    //                     return -1;
    //                 }
    //             }
    //         }
    //     }
    // }

    mpi_printf("Success!\n");

    MPI_Finalize();
    return 0;
}
