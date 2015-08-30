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

template <class T>
void print_img1d(DistributedImage<T> &img) {
    for (int x = 0; x < img.width(); x++) {
        std::cout << std::setw(4) << img(x);
    }
    std::cout << "\n";
}

template <class T>
void print_img2d(DistributedImage<T> &img) {
    for (int y = 0; y < img.height(); y++) {
        for (int x = 0; x < img.width(); x++) {
            std::cout << std::setw(4) << img(x, y);
        }
        std::cout << "\n";
    }
}

namespace {
inline int intmin(int a, int b) { return a < b ? a : b; }
inline int intmax(int a, int b) { return a > b ? a : b; }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, numprocs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    Internal::distribute_loops_test();

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
        for (int x = 0; x < in.width(); x++) {
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
        for (int x = 0; x < out.width(); x++) {
            int correct = 2 * out.global(x) + 1;
            if (out(x) != correct) {
                mpi_printf("out(%d) = %d instead of %d\n", x, out(x), correct);
                MPI_Finalize();
                return -1;
            }
            //printf("Rank %d gets out(%d) = %d\n", rank, x, out(x));
        }
    }

    {
        DistributedImage<int> in(20);
        in.set_domain(x);
        in.placement().distribute(x);
        in.allocate();

        for (int x = 0; x < in.width(); x++) {
            in(x) = 2 * in.global(x);
        }

        Func f, g;
        f(x) = in(x) + 1;
        g(x) = f(x) + 1;
        f.compute_root().distribute(x);
        g.distribute(x);

        DistributedImage<int> out(20);
        out.set_domain(x);
        out.placement().distribute(x);
        out.allocate();
        g.realize(out.get_buffer());
        for (int x = 0; x < out.width(); x++) {
            int correct = 2 * out.global(x) + 2;
            if (out(x) != correct) {
                printf("[rank %d] out(%d) = %d instead of %d\n", rank, x, out(x), correct);
                MPI_Finalize();
                return -1;
            }
        }
    }

    {
        DistributedImage<int> in(20);
        in.set_domain(x);
        in.placement().distribute(x);
        in.allocate();

        for (int x = 0; x < in.width(); x++) {
            in(x) = 2 * in.global(x);
        }

        Expr clamped_x = clamp(x, 0, in.global_width()-1);
        Func clamped;
        clamped(x) = in(clamped_x);
        Func f;
        f(x) = clamped(x) + clamped(x+1) + 1;
        f.distribute(x);

        DistributedImage<int> out(20);
        out.set_domain(x);
        out.placement().distribute(x);
        out.allocate();
        f.realize(out.get_buffer());
        for (int x = 0; x < out.width(); x++) {
            const int xmax = out.global_width() - 1;
            const int xa = x;
            const int xb = out.global(x+1) >= xmax ? out.local(xmax) : x+1;
            const int correct = 2 * out.global(xa) + 2 * out.global(xb) + 1;
            if (out(x) != correct) {
                mpi_printf("out(%d) = %d instead of %d\n", x, out(x), correct);
                MPI_Finalize();
                return -1;
            }
        }
    }

    {
        DistributedImage<int> in(20);
        in.set_domain(x);
        in.placement().distribute(x);
        in.allocate();

        for (int x = 0; x < in.width(); x++) {
            in(x) = 2 * in.global(x);
        }

        Func clamped;
        clamped(x) = in(clamp(x, 0, in.global_width()-1));
        Func f;
        f(x) = clamped(x-1) + clamped(x+1) + 1;
        f.distribute(x);

        DistributedImage<int> out(20);
        out.set_domain(x);
        out.placement().distribute(x);
        out.allocate();
        f.realize(out.get_buffer());

        const int slice = (int)ceil(in.global_width() / (float)numprocs);
        assert(in.global(0, 0) == rank * slice);
        assert(in.local(0, rank * slice) == 0);
        assert(in.width() == (intmin((rank + 1) * slice - 1, in.global_width() - 1) - (rank * slice) + 1));
        assert(in.mine(0) == (rank == 0));
        assert(in.mine(in.global_width() - 1) == (rank == numprocs - 1));
        assert(in.mine(-1) == false);
        assert(in.mine(in.global_width()) == false);

        for (int x = 0; x < out.width(); x++) {
            const int xmax = out.global_width() - 1;
            const int gxp1 = out.global(0, x+1) >= xmax ? xmax : out.global(0, x+1),
                gxm1 = out.global(0, x) == 0 ? 0 : out.global(0, x-1);
            //const int gx = out.global(0, x), gy = out.global(1, y);
            const int correct = (2 * gxm1) + (2 * gxp1) + 1;
            if (out(x) != correct) {
                mpi_printf("out(%d) = %d instead of %d\n", x, out(x), correct);
                MPI_Finalize();
                return -1;
            }
        }
    }

    {
        DistributedImage<int> in(20);
        in.set_domain(x);
        in.placement().distribute(x);
        in.allocate();

        for (int x = 0; x < in.width(); x++) {
            in(x) = 2 * in.global(x);
        }

        Expr clamped_x = clamp(x, 0, in.global_width()-1);
        Func clamped;
        clamped(x) = in(clamped_x);
        Func f("f"), g("g");
        f(x) = clamped(x) + clamped(x+1) + 1;
        g(x) = f(x) + f(x+1) + 1;
        f.compute_root().distribute(x);
        g.distribute(x);

        DistributedImage<int> out(20);
        out.set_domain(x);
        out.placement().distribute(x);
        out.allocate();
        g.realize(out.get_buffer());
        for (int x = 0; x < out.width(); x++) {
            const int xmax = out.global_width() - 1;
            const int xp1 = out.global(x+1) >= xmax ? out.local(xmax) : x+1,
                xp2 = out.global(x+2) >= xmax ? out.local(xmax) : x+2;
            const int correct = (2 * out.global(x) + 2 * out.global(xp1) + 1) +
                (2 * out.global(xp1) + 2 * out.global(xp2) + 1) + 1;
            if (out(x) != correct) {
                printf("[rank %d] out(%d) = %d instead of %d\n", rank, x, out(x), correct);
                MPI_Finalize();
                return -1;
            }
        }
    }

    {
        DistributedImage<int> in(10, 20);
        in.set_domain(x, y);
        in.placement().distribute(y);
        in.allocate();

        for (int y = 0; y < in.height(); y++) {
            for (int x = 0; x < in.width(); x++) {
                in(x, y) = in.global(0, x) + in.global(1, y);
            }
        }

        Expr clamped_x = clamp(x, 0, in.global_width()-1),
            clamped_y = clamp(y, 0, in.global_height()-1);
        Func clamped;
        clamped(x, y) = in(clamped_x, clamped_y);
        Func f;
        f(x, y) = clamped(x, y) + clamped(x, y+1) + 1;
        f.distribute(y);

        DistributedImage<int> out(10, 20);
        out.set_domain(x, y);
        out.placement().distribute(y);
        out.allocate();
        f.realize(out.get_buffer());
        for (int y = 0; y < out.height(); y++) {
            for (int x = 0; x < out.width(); x++) {
                const int max = out.global_height() - 1;
                const int clamp = out.global(1, y+1) >= max ? out.local(1, max) : y+1;
                const int correct = out.global(0, x) + out.global(1, y) + out.global(0, x) + out.global(1, clamp) + 1;
                if (out(x, y) != correct) {
                    printf("[rank %d] out(%d,%d) = %d instead of %d\n", rank, x, y, out(x, y), correct);
                    MPI_Finalize();
                    return -1;
                }
            }
        }
    }

    {
        DistributedImage<int> in(10, 20);
        in.set_domain(x, y);
        in.placement().distribute(y);
        in.allocate();

        for (int y = 0; y < in.height(); y++) {
            for (int x = 0; x < in.width(); x++) {
                in(x, y) = in.global(0, x) + in.global(1, y);
            }
        }

        Expr clamped_x = clamp(x, 0, in.global_width()-1),
            clamped_y = clamp(y, 0, in.global_height()-1);
        Func clamped;
        clamped(x, y) = in(clamped_x, clamped_y);
        Func f, g;
        f(x, y) = clamped(x, y) + clamped(x, y+1) + 1;
        g(x, y) = f(x, y) + f(x, y+1) + 1;
        f.compute_at(g, y);
        g.distribute(y);

        DistributedImage<int> out(10, 20);
        out.set_domain(x, y);
        out.placement().distribute(y);
        out.allocate();
        g.realize(out.get_buffer());
        for (int y = 0; y < out.height(); y++) {
            for (int x = 0; x < out.width(); x++) {
                const int max = out.global_height() - 1;
                const int yp1 = out.global(1, y+1) >= max ? out.local(1, max) : y+1,
                    yp2 = out.global(1, y+2) >= max ? out.local(1, max) : y+2;
                const int correct = (out.global(0, x) + out.global(1, y) + out.global(0, x) + out.global(1, yp1) + 1) +
                    (out.global(0, x) + out.global(1, yp1) + out.global(0, x) + out.global(1, yp2) + 1) + 1;
                if (out(x, y) != correct) {
                    printf("[rank %d] out(%d,%d) = %d instead of %d\n", rank, x, y, out(x, y), correct);
                    MPI_Finalize();
                    return -1;
                }
            }
        }
    }

    {
        DistributedImage<int> in(50, 60);
        in.set_domain(x, y);
        in.placement().distribute(y);
        in.allocate();

        for (int y = 0; y < in.height(); y++) {
            for (int x = 0; x < in.width(); x++) {
                in(x, y) = in.global(0, x) + in.global(1, y);
            }
        }

        Expr clamped_x = clamp(x, 0, in.global_width()-1),
            clamped_y = clamp(y, 0, in.global_height()-1);
        Func clamped;
        clamped(x, y) = in(clamped_x, clamped_y);
        Func f, g;
        Var xo("xo"), xi("xi"), fused("fused");
        f(x, y) = clamped(x, y) + clamped(x, y+1) + 1;
        g(x, y) = f(x, y) + f(x, y+1) + 1;
        f.compute_root();
        f.split(x, xo, xi, 5);
        g.split(x, xo, xi, 5);
        g.fuse(xo, y, fused);
        f.distribute(xo);
        g.distribute(fused);

        DistributedImage<int> out(50, 60);
        out.set_domain(x, y);
        out.placement().distribute(y);
        out.allocate();
        g.realize(out.get_buffer());
        for (int y = 0; y < out.height(); y++) {
            for (int x = 0; x < out.width(); x++) {
                const int max = out.global_height() - 1;
                const int yp1 = out.global(1, y+1) >= max ? out.local(1, max) : y+1,
                    yp2 = out.global(1, y+2) >= max ? out.local(1, max) : y+2;
                const int correct = (out.global(0, x) + out.global(1, y) + out.global(0, x) + out.global(1, yp1) + 1) +
                    (out.global(0, x) + out.global(1, yp1) + out.global(0, x) + out.global(1, yp2) + 1) + 1;
                if (out(x, y) != correct) {
                    printf("[rank %d] out(%d,%d) = %d instead of %d\n", rank, x, y, out(x, y), correct);
                    MPI_Finalize();
                    return -1;
                }
            }
        }
    }

    {
        DistributedImage<int> in(10, 20);
        in.set_domain(x, y);
        in.placement().distribute(y);
        in.allocate();

        for (int y = 0; y < in.height(); y++) {
            for (int x = 0; x < in.width(); x++) {
                in(x, y) = in.global(0, x) + in.global(1, y);
            }
        }

        Func f, g;
        f(x, y) = in(x, y) + in(x, y) + 1;
        g(x, y) = 2 * f(x, y);
        f.compute_root().distribute(y);
        g.distribute(y);

        DistributedImage<int> out(10, 20);
        out.set_domain(x, y);
        out.placement().distribute(y);
        out.allocate();
        g.realize(out.get_buffer());
        for (int y = 0; y < out.height(); y++) {
            for (int x = 0; x < out.width(); x++) {
                int gx = out.global(0, x), gy = out.global(1, y);
                const int correct = 2*(gx + gy + gx + gy + 1);
                if (out(x, y) != correct) {
                    printf("[rank %d] out(%d,%d) = %d instead of %d\n", rank, x, y, out(x, y), correct);
                    MPI_Finalize();
                    return -1;
                }
            }
        }
    }

    {
        DistributedImage<int> in(10, 20, 30);
        in.set_domain(x, y, z);
        in.placement().distribute(z);
        in.allocate();

        for (int z = 0; z < in.channels(); z++) {
            for (int y = 0; y < in.height(); y++) {
                for (int x = 0; x < in.width(); x++) {
                    int gx = in.global(0, x), gy = in.global(1, y), gz = in.global(2, z);
                    in(x, y, z) = gx + gy + gz;
                }
            }
        }

        Func f, g;
        f(x, y, z) = 2 * in(x, y, z);
        g(x, y, z) = 2 * f(x, y, z);
        f.compute_root().distribute(z);
        g.distribute(z);

        DistributedImage<int> out(10, 20, 30);
        out.set_domain(x, y, z);
        out.placement().distribute(z);
        out.allocate();
        g.realize(out.get_buffer());
        for (int z = 0; z < out.channels(); z++) {
            for (int y = 0; y < out.height(); y++) {
                for (int x = 0; x < out.width(); x++) {
                    int gx = out.global(0, x), gy = out.global(1, y), gz = out.global(2, z);
                    int correct = 4*(gx+gy+gz);
                    if (out(x,y,z) != correct) {
                        mpi_printf("out(%d,%d,%d) = %d instead of %d\n", x, y, z, out(x,y,z), correct);
                        MPI_Finalize();
                        return -1;
                    }
                }
            }
        }
    }

    {
        DistributedImage<int> in(10, 20);
        in.set_domain(x, y);
        in.placement().distribute(y);
        in.allocate();

        for (int y = 0; y < in.height(); y++) {
            for (int x = 0; x < in.width(); x++) {
                in(x, y) = in.global(0, x) + in.global(1, y);
            }
        }

        Func f, g;
        f(x, y) = 2 * in(x, y);
        g(x, y) = 2 * f(x, y);
        f.compute_root().distribute(x);
        g.distribute(y);

        DistributedImage<int> out(10, 20);
        out.set_domain(x, y);
        out.placement().distribute(y);
        out.allocate();
        g.realize(out.get_buffer());
        for (int y = 0; y < out.height(); y++) {
            for (int x = 0; x < out.width(); x++) {
                int gx = out.global(0, x), gy = out.global(1, y);
                const int correct = 4*(gx + gy);
                if (out(x, y) != correct) {
                    printf("[rank %d] out(%d,%d) = %d instead of %d\n", rank, x, y, out(x, y), correct);
                    MPI_Finalize();
                    return -1;
                }
            }
        }
    }

    {
        DistributedImage<int> in(10, 20, 30);
        in.set_domain(x, y, z);
        in.placement().distribute(z);
        in.allocate();

        for (int z = 0; z < in.channels(); z++) {
            for (int y = 0; y < in.height(); y++) {
                for (int x = 0; x < in.width(); x++) {
                    int gx = in.global(0, x), gy = in.global(1, y), gz = in.global(2, z);
                    in(x, y, z) = gx + gy + gz;
                }
            }
        }

        Func f, g;
        f(x, y, z) = 2 * in(x, y, z);
        g(x, y, z) = 2 * f(x, y, z);
        f.compute_root().distribute(y);
        g.distribute(z);

        DistributedImage<int> out(10, 20, 30);
        out.set_domain(x, y, z);
        out.placement().distribute(z);
        out.allocate();
        g.realize(out.get_buffer());
        for (int z = 0; z < out.channels(); z++) {
            for (int y = 0; y < out.height(); y++) {
                for (int x = 0; x < out.width(); x++) {
                    int gx = out.global(0, x), gy = out.global(1, y), gz = out.global(2, z);
                    int correct = 4*(gx+gy+gz);
                    if (out(x,y,z) != correct) {
                        printf("[rank %d] out(%d,%d,%d) = %d instead of %d\n", rank, x, y, z, out(x,y,z), correct);
                        MPI_Finalize();
                        return -1;
                    }
                }
            }
        }
    }

    {
        DistributedImage<int> in(10, 20);
        in.set_domain(x, y);
        in.placement().distribute(y);
        in.allocate();

        for (int y = 0; y < in.height(); y++) {
            for (int x = 0; x < in.width(); x++) {
                in(x, y) = in.global(0, x) + in.global(1, y);
            }
        }

        Func f, g, h, i;
        f(x, y) = 2 * in(x, y);
        g(x, y) = 2 * f(x, y);
        h(x, y) = 2 * f(x, y);
        i(x, y) = g(x, y) + h(x, y);

        f.compute_root().distribute(y);
        g.compute_root().distribute(y);
        i.distribute(y);

        DistributedImage<int> out(10, 20);
        out.set_domain(x, y);
        out.placement().distribute(y);
        out.allocate();
        i.realize(out.get_buffer());
        for (int y = 0; y < out.height(); y++) {
            for (int x = 0; x < out.width(); x++) {
                int gx = out.global(0, x), gy = out.global(1, y);
                int correct = 4*(gx+gy) + 4*(gx+gy);
                if (out(x,y) != correct) {
                    mpi_printf("out(%d,%d) = %d instead of %d\n", x, y,out(x,y), correct);
                    MPI_Finalize();
                    return -1;
                }
            }
        }
    }

    {
        DistributedImage<int> in(10, 20);
        in.set_domain(x, y);
        in.placement().distribute(y);
        in.allocate();

        for (int y = 0; y < in.height(); y++) {
            for (int x = 0; x < in.width(); x++) {
                in(x, y) = in.global(0, x) + in.global(1, y);
            }
        }

        Func f, g;
        f(x, y) = in(x, y) + in(x, y) + 1;
        g(x, y) = f(x, y);
        f.compute_root().distribute(x);
        g.distribute(x);

        DistributedImage<int> out(10, 20);
        out.set_domain(x, y);
        out.placement().distribute(x);
        out.allocate();
        g.realize(out.get_buffer());
        for (int y = 0; y < out.height(); y++) {
            for (int x = 0; x < out.width(); x++) {
                int gx = out.global(0, x), gy = out.global(1, y);
                const int correct = (gx + gy + gx + gy + 1);
                if (out(x, y) != correct) {
                    printf("[rank %d] out(%d,%d) = %d instead of %d\n", rank, x, y, out(x, y), correct);
                    MPI_Finalize();
                    return -1;
                }
            }
        }
    }

    {
        DistributedImage<int> in(100, 100);
        in.set_domain(x, y);
        in.placement().distribute(y);
        in.allocate();

        for (int y = 0; y < in.height(); y++) {
            for (int x = 0; x < in.width(); x++) {
                in(x, y) = in.global(0, x) + in.global(1, y);
            }
        }

        Expr clamped_x = clamp(x, 0, in.global_width()-1),
            clamped_y = clamp(y, 0, in.global_height()-1);
        Func clamped;
        clamped(x, y) = in(clamped_x, clamped_y);
        Func blurx, blury;
        blurx(x, y) = (clamped(x-1, y) + clamped(x, y) + clamped(x+1, y)) / 3;
        blury(x, y) = (blurx(x, y-1) + blurx(x, y) + blurx(x, y+1)) / 3;
        blurx.compute_root().distribute(y);
        blury.distribute(y);

        DistributedImage<int> out(100, 100);
        out.set_domain(x, y);
        out.placement().distribute(y);
        out.allocate();
        blury.realize(out.get_buffer());
        for (int y = 0; y < out.height(); y++) {
            for (int x = 0; x < out.width(); x++) {
                const int xmax = out.global_width() - 1, ymax = out.global_height() - 1;
                const int gxp1 = out.global(0, x+1) >= xmax ? xmax : out.global(0, x+1),
                    gxm1 = out.global(0, x) == 0 ? 0 : out.global(0, x-1);
                const int gyp1 = out.global(1, y+1) >= ymax ? ymax : out.global(1, y+1),
                    gym1 = out.global(1, y) == 0 ? 0 : out.global(1, y-1);
                const int gx = out.global(0, x), gy = out.global(1, y);
                const int correct = (((gxm1 + gym1 + gx + gym1 + gxp1 + gym1)/3) +
                                     ((gxm1 + gy + gx + gy + gxp1 + gy)/3) +
                                     ((gxm1 + gyp1 + gx + gyp1 + gxp1 + gyp1)/3)) / 3;
                if (out(x, y) != correct) {
                    printf("[rank %d] out(%d,%d) = %d instead of %d\n", rank, x, y, out(x, y), correct);
                    MPI_Finalize();
                    return -1;
                }
            }
        }
    }

    {
        DistributedImage<int> in(11, 113);
        in.set_domain(x, y);
        in.placement().distribute(y);
        in.allocate();

        for (int y = 0; y < in.height(); y++) {
            for (int x = 0; x < in.width(); x++) {
                in(x, y) = in.global(0, x) + in.global(1, y);
            }
        }

        Expr clamped_x = clamp(x, 0, in.global_width()-1),
            clamped_y = clamp(y, 0, in.global_height()-1);
        Func clamped;
        clamped(x, y) = in(clamped_x, clamped_y);
        Func blurx, blury;
        blurx(x, y) = (clamped(x-1, y) + clamped(x, y) + clamped(x+1, y)) / 3;
        blury(x, y) = (blurx(x, y-1) + blurx(x, y) + blurx(x, y+1)) / 3;

        // First tile, then fuse the tile indices and distribute
        // across the tiles.
        Var x_outer, y_outer, x_inner, y_inner, tile_index;
        blurx.tile(x, y, x_outer, y_outer, x_inner, y_inner, 2, 2);
        blurx.fuse(x_outer, y_outer, tile_index);
        blurx.compute_root().distribute(tile_index).parallel(tile_index);
        blury.distribute(y);

        DistributedImage<int> out(11, 113);
        out.set_domain(x, y);
        out.placement().distribute(y);
        out.allocate();
        blury.realize(out.get_buffer());
        for (int y = 0; y < out.height(); y++) {
            for (int x = 0; x < out.width(); x++) {
                const int xmax = out.global_width() - 1, ymax = out.global_height() - 1;
                const int gxp1 = out.global(0, x+1) >= xmax ? xmax : out.global(0, x+1),
                    gxm1 = out.global(0, x) == 0 ? 0 : out.global(0, x-1);
                const int gyp1 = out.global(1, y+1) >= ymax ? ymax : out.global(1, y+1),
                    gym1 = out.global(1, y) == 0 ? 0 : out.global(1, y-1);
                const int gx = out.global(0, x), gy = out.global(1, y);
                const int correct = (((gxm1 + gym1 + gx + gym1 + gxp1 + gym1)/3) +
                                     ((gxm1 + gy + gx + gy + gxp1 + gy)/3) +
                                     ((gxm1 + gyp1 + gx + gyp1 + gxp1 + gyp1)/3)) / 3;
                if (out(x, y) != correct) {
                    printf("[rank %d] out(%d,%d) = %d instead of %d\n", rank, x, y, out(x, y), correct);
                    MPI_Finalize();
                    return -1;
                }
            }
        }
    }

    {
        DistributedImage<int> in(20);
        in.set_domain(x);
        in.placement().distribute(x);
        in.allocate();

        for (int x = 0; x < in.width(); x++) {
            in(x) = 2 * in.global(x);
        }

        Func f, g;
        f(x) = in(x) + 1;
        g(x) = f(x) + 1;
        Var xo("xo"), xi("xi");
        f.compute_root().split(x, xo, xi, 8).distribute(xo);
        g.distribute(x);

        DistributedImage<int> out(20);
        out.set_domain(x);
        out.placement().distribute(x);
        out.allocate();
        g.realize(out.get_buffer());
        for (int x = 0; x < out.width(); x++) {
            int correct = 2 * out.global(x) + 2;
            if (out(x) != correct) {
                printf("[rank %d] out(%d) = %d instead of %d\n", rank, x, out(x), correct);
                MPI_Finalize();
                return -1;
            }
        }
    }

    {
        DistributedImage<int> in(50, 50);
        in.set_domain(x, y);
        in.placement().distribute(y);
        in.allocate();

        for (int y = 0; y < in.height(); y++) {
            for (int x = 0; x < in.width(); x++) {
                in(x, y) = in.global(0, x) + in.global(1, y);
            }
        }

        RDom k(0, in.global_height());
        Func f;
        f(x, y) = 2 * in(x, y);
        f(x, y) += in(x, k) + 1;

        f.distribute(y);
        f.update().distribute(y);

        DistributedImage<int> out(50, 50);
        out.set_domain(x, y);
        out.placement().distribute(y);
        out.allocate();
        f.realize(out.get_buffer());
        for (int y = 0; y < out.height(); y++) {
            for (int x = 0; x < out.width(); x++) {
                int gx = out.global(0, x), gy = out.global(1, y);
                int correct = 2 * (gx + gy);
                for (int k = 0; k < out.global_height(); k++) {
                    correct += (gx + k) + 1;
                }
                if (out(x,y) != correct) {
                    mpi_printf("out(%d,%d) = %d instead of %d\n", x, y,out(x,y), correct);
                    MPI_Finalize();
                    return -1;
                }
            }
        }
    }

    printf("Rank %d Success!\n", rank);

    MPI_Finalize();
    return 0;
}
