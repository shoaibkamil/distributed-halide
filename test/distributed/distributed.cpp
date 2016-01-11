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

template <class T>
inline bool vec_eq(const std::vector<T> &a, const std::vector<T> &b) {
    return a == b;
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

    Var x, y, z, c;

    {
        DistributedImage<int> in(20);
        Func f;
        f(x) = in(x) + 1;
        f.distribute(x);

        DistributedImage<int> out(20);
        out.set_domain(x);
        out.placement().distribute(x);
        out.allocate();
        in.set_domain(x);
        in.placement().distribute(x);
        in.allocate(f, out);
        for (int x = 0; x < in.width(); x++) {
            in(x) = 2 * in.global(x);
        }

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

        Func f, g;
        f(x) = in(x) + 1;
        g(x) = f(x) + 1;
        f.compute_root().distribute(x);
        g.distribute(x);

        DistributedImage<int> out(20);
        out.set_domain(x);
        out.placement().distribute(x);
        out.allocate();
        in.set_domain(x);
        in.placement().distribute(x);
        in.allocate(g, out);

        for (int x = 0; x < in.width(); x++) {
            in(x) = 2 * in.global(x);
        }
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
        in.set_domain(x);
        in.placement().distribute(x);
        in.allocate(f, out);

        for (int x = 0; x < in.width(); x++) {
            in(x) = 2 * in.global(x);
        }

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

        Func clamped;
        clamped(x) = in(clamp(x, 0, in.global_width()-1));
        Func f;
        f(x) = clamped(x-1) + clamped(x+1) + 1;
        f.distribute(x);

        DistributedImage<int> out(20);
        out.set_domain(x);
        out.placement().distribute(x);
        out.allocate();
        in.set_domain(x);
        in.placement().distribute(x);
        in.allocate(f, out);

        for (int x = 0; x < in.width(); x++) {
            in(x) = 2 * in.global(x);
        }

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
        in.set_domain(x);
        in.placement().distribute(x);
        in.allocate(g, out);

        for (int x = 0; x < in.width(); x++) {
            in(x) = 2 * in.global(x);
        }

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
        in.set_domain(x, y);
        in.placement().distribute(y);
        in.allocate(f, out);

        for (int y = 0; y < in.height(); y++) {
            for (int x = 0; x < in.width(); x++) {
                in(x, y) = in.global(0, x) + in.global(1, y);
            }
        }

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
        in.set_domain(x, y);
        in.placement().distribute(y);
        in.allocate(g, out);

        for (int y = 0; y < in.height(); y++) {
            for (int x = 0; x < in.width(); x++) {
                in(x, y) = in.global(0, x) + in.global(1, y);
            }
        }

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
        in.set_domain(x, y);
        in.placement().distribute(y);
        in.allocate(g, out);

        for (int y = 0; y < in.height(); y++) {
            for (int x = 0; x < in.width(); x++) {
                in(x, y) = in.global(0, x) + in.global(1, y);
            }
        }

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

        Func f, g;
        f(x, y) = in(x, y) + in(x, y) + 1;
        g(x, y) = 2 * f(x, y);
        f.compute_root().distribute(y);
        g.distribute(y);

        DistributedImage<int> out(10, 20);
        out.set_domain(x, y);
        out.placement().distribute(y);
        out.allocate();
        in.set_domain(x, y);
        in.placement().distribute(y);
        in.allocate(g, out);

        for (int y = 0; y < in.height(); y++) {
            for (int x = 0; x < in.width(); x++) {
                in(x, y) = in.global(0, x) + in.global(1, y);
            }
        }
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

        Func f, g;
        f(x, y, z) = 2 * in(x, y, z);
        g(x, y, z) = 2 * f(x, y, z);
        f.compute_root().distribute(z);
        g.distribute(z);

        DistributedImage<int> out(10, 20, 30);
        out.set_domain(x, y, z);
        out.placement().distribute(z);
        out.allocate();
        in.set_domain(x, y, z);
        in.placement().distribute(z);
        in.allocate(g, out);

        for (int z = 0; z < in.channels(); z++) {
            for (int y = 0; y < in.height(); y++) {
                for (int x = 0; x < in.width(); x++) {
                    int gx = in.global(0, x), gy = in.global(1, y), gz = in.global(2, z);
                    in(x, y, z) = gx + gy + gz;
                }
            }
        }

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

        Func f, g;
        f(x, y) = 2 * in(x, y);
        g(x, y) = 2 * f(x, y);
        f.compute_root().distribute(x);
        g.distribute(y);

        DistributedImage<int> out(10, 20);
        out.set_domain(x, y);
        out.placement().distribute(y);
        out.allocate();
        in.set_domain(x, y);
        in.placement().distribute(y);
        in.allocate(g, out);

        for (int y = 0; y < in.height(); y++) {
            for (int x = 0; x < in.width(); x++) {
                in(x, y) = in.global(0, x) + in.global(1, y);
            }
        }

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

        Func f, g;
        f(x, y, z) = 2 * in(x, y, z);
        g(x, y, z) = 2 * f(x, y, z);
        f.compute_root().distribute(y);
        g.distribute(z);

        DistributedImage<int> out(10, 20, 30);
        out.set_domain(x, y, z);
        out.placement().distribute(z);
        out.allocate();
        in.set_domain(x, y, z);
        in.placement().distribute(z);
        in.allocate(g, out);

        for (int z = 0; z < in.channels(); z++) {
            for (int y = 0; y < in.height(); y++) {
                for (int x = 0; x < in.width(); x++) {
                    int gx = in.global(0, x), gy = in.global(1, y), gz = in.global(2, z);
                    in(x, y, z) = gx + gy + gz;
                }
            }
        }

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
        in.set_domain(x, y);
        in.placement().distribute(y);
        in.allocate(i, out);

        for (int y = 0; y < in.height(); y++) {
            for (int x = 0; x < in.width(); x++) {
                in(x, y) = in.global(0, x) + in.global(1, y);
            }
        }

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

        Func f, g;
        f(x, y) = in(x, y) + in(x, y) + 1;
        g(x, y) = f(x, y);
        f.compute_root().distribute(x);
        g.distribute(x);

        DistributedImage<int> out(10, 20);
        out.set_domain(x, y);
        out.placement().distribute(x);
        out.allocate();
        in.set_domain(x, y);
        in.placement().distribute(y);
        in.allocate(g, out);

        for (int y = 0; y < in.height(); y++) {
            for (int x = 0; x < in.width(); x++) {
                in(x, y) = in.global(0, x) + in.global(1, y);
            }
        }

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
        in.set_domain(x, y);
        in.placement().distribute(y);
        in.allocate(blury, out);

        for (int y = 0; y < in.height(); y++) {
            for (int x = 0; x < in.width(); x++) {
                in(x, y) = in.global(0, x) + in.global(1, y);
            }
        }

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
        in.set_domain(x, y);
        in.placement().distribute(y);
        in.allocate(blury, out);

        for (int y = 0; y < in.height(); y++) {
            for (int x = 0; x < in.width(); x++) {
                in(x, y) = in.global(0, x) + in.global(1, y);
            }
        }

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
        DistributedImage<int> in(100);

        Func f, g;
        f(x) = in(x) + 1;
        g(x) = f(x) + 1;
        Var xo("xo"), xi("xi");
        f.compute_root().split(x, xo, xi, 8).distribute(xo);
        g.distribute(x);

        DistributedImage<int> out(100);
        out.set_domain(x);
        out.placement().distribute(x);
        out.allocate();
        in.set_domain(x);
        in.placement().distribute(x);
        in.allocate(g, out);

        for (int x = 0; x < in.width(); x++) {
            in(x) = 2 * in.global(x);
        }

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
        in.set_domain(x, y);
        in.placement().distribute(y);
        in.allocate(f, out);

        for (int y = 0; y < in.height(); y++) {
            for (int x = 0; x < in.width(); x++) {
                in(x, y) = in.global(0, x) + in.global(1, y);
            }
        }

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

    {
        DistributedImage<int> in(50, 50);
        Image<int> aux(1, 50);

        for (int y = 0; y < aux.height(); y++) {
            for (int x = 0; x < aux.width(); x++) {
                aux(x, y) = x + y + 1;
            }
        }

        Expr clamped_x = clamp(x, 0, in.global_width()-1),
            clamped_y = clamp(y, 0, in.global_height()-1);
        Func clamped;
        clamped(x, y) = in(clamped_x, clamped_y);
        Func f, g;
        f(x, y) = clamped(x, y) + clamped(x, y+1) + 1;
        g(x, y) = f(x, y) + f(x, y+1) + 1 + aux(0, y);
        f.compute_root().distribute(y);
        g.distribute(y);

        DistributedImage<int> out(50, 50);
        out.set_domain(x, y);
        out.placement().distribute(y);
        out.allocate();
        in.set_domain(x, y);
        in.placement().distribute(y);
        in.allocate(g, out);

        for (int y = 0; y < in.height(); y++) {
            for (int x = 0; x < in.width(); x++) {
                in(x, y) = in.global(0, x) + in.global(1, y);
            }
        }

        g.realize(out.get_buffer());
        for (int y = 0; y < out.height(); y++) {
            for (int x = 0; x < out.width(); x++) {
                const int max = out.global_height() - 1;
                const int yp1 = out.global(1, y+1) >= max ? out.local(1, max) : y+1,
                    yp2 = out.global(1, y+2) >= max ? out.local(1, max) : y+2;
                const int correct = (out.global(0, x) + out.global(1, y) + out.global(0, x) + out.global(1, yp1) + 1) +
                    (out.global(0, x) + out.global(1, yp1) + out.global(0, x) + out.global(1, yp2) + 1) + 1 +
                    out.global(1, y) + 1;
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

        Expr clamped_x = clamp(x, 0, in.global_width()-1),
            clamped_y = clamp(y, 0, in.global_height()-1);
        Func clamped;
        clamped(x, y) = in(clamped_x, clamped_y);
        Func blurx, blury;
        blurx(x, y) = (clamped(x-1, y) + clamped(x, y) + clamped(x+1, y)) / 3;
        blury(x, y) = (blurx(x, y-1) + blurx(x, y) + blurx(x, y+1)) / 3;

        Var yi;
        blurx.compute_rank().parallel(y, 2).vectorize(x, 8);
        blury.distribute(y);

        DistributedImage<int> out(100, 100);
        out.set_domain(x, y);
        out.placement().distribute(y);
        out.allocate();
        in.set_domain(x, y);
        in.placement().distribute(y);
        in.allocate(blury, out);

        for (int y = 0; y < in.height(); y++) {
            for (int x = 0; x < in.width(); x++) {
                in(x, y) = in.global(0, x) + in.global(1, y);
            }
        }

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
        DistributedImage<int> in(100, 100);

        Expr clamped_x = clamp(x, 0, in.global_width()-1),
            clamped_y = clamp(y, 0, in.global_height()-1);
        Func clamped;
        clamped(x, y) = in(clamped_x, clamped_y);
        Func f, g, h;
        f(x, y) = clamped(x, y) + clamped(x, y+1) + 1;
        g(x, y) = f(x, y) + f(x, y+1) + 1;
        h(x, y) = g(x, y) * 2;

        f.compute_rank().parallel(y, 2).vectorize(x, 8);
        g.compute_root().parallel(y, 2).vectorize(x, 8).distribute(y);
        h.parallel(y, 2).vectorize(x, 8).distribute(y);

        DistributedImage<int> out(100, 100);
        out.set_domain(x, y);
        out.placement().parallel(y, 2).distribute(y);
        out.allocate();
        in.set_domain(x, y);
        in.placement().distribute(y);
        in.allocate(h, out);

        for (int y = 0; y < in.height(); y++) {
            for (int x = 0; x < in.width(); x++) {
                in(x, y) = in.global(0, x) + in.global(1, y);
            }
        }

        h.realize(out.get_buffer());
        for (int y = 0; y < out.height(); y++) {
            for (int x = 0; x < out.width(); x++) {
                const int max = out.global_height() - 1;
                const int yp1 = out.global(1, y+1) >= max ? out.local(1, max) : y+1,
                    yp2 = out.global(1, y+2) >= max ? out.local(1, max) : y+2;
                const int correct = ((out.global(0, x) + out.global(1, y) + out.global(0, x) + out.global(1, yp1) + 1) +
                                     (out.global(0, x) + out.global(1, yp1) + out.global(0, x) + out.global(1, yp2) + 1) + 1) * 2;
                if (out(x, y) != correct) {
                    printf("[rank %d] out(%d,%d) = %d instead of %d\n", rank, x, y, out(x, y), correct);
                    MPI_Finalize();
                    return -1;
                }
            }
        }
    }

    {
        DistributedImage<int16_t> in(100, 100);

        Expr clamped_x = clamp(x, 0, in.global_width()-1),
            clamped_y = clamp(y, 0, in.global_height()-1);
        Func clamped;
        clamped(x, y) = in(clamped_x, clamped_y);

        const int J = 2;
        Var k;
        Func ll;
        Func gPyramid[J], lPyramid[J], outGPyramid[J];

        for (int j = 0; j < J; j++) {
            gPyramid[j](x, y, k) = clamped(x, y) + k;
        }

        lPyramid[J-1](x, y, k) = gPyramid[J-1](x, y, k);
        for (int j = J-2; j >= 0; j--) {
            lPyramid[j](x, y, k) = gPyramid[j](x, y, k) - gPyramid[j+1](x, y, k);
        }

        outGPyramid[J-1](x, y) = lPyramid[J-1](x, y, 0);
        for (int j = J-2; j >= 0; j--) {
            Expr li = clamp(clamped(x, y), 0, J);
            outGPyramid[j](x, y) = lPyramid[j](x, y, li);
        }

        ll(x, y) = outGPyramid[0](x, y);

        // Schedule
        ll.distribute(y);
        for (int j = 0; j < 1; j++) {
            outGPyramid[j].compute_root().distribute(y);
        }
        for (int j = 1; j < J; j++) {
            gPyramid[j].compute_rank();
        }

        DistributedImage<int> out(100, 100);
        out.set_domain(x, y);
        out.placement().parallel(y).distribute(y);
        out.allocate();
        in.set_domain(x, y);
        in.placement().distribute(y);
        in.allocate(ll, out);

        for (int y = 0; y < in.height(); y++) {
            for (int x = 0; x < in.width(); x++) {
                in(x, y) = in.global(0, x) + in.global(1, y);
            }
        }

        ll.realize(out.get_buffer());
        // Don't bother checking output.
    }

    {
        DistributedImage<int> in(100, 100, "in");
        ImageParam inp(Int(32), 2, "in");

        std::vector<std::pair<Expr, Expr>>
            global_bounds = {std::make_pair(0, 100), std::make_pair(0, 100)};

        Func accessor;
        accessor(x, y) = inp(x, y);
        Func clamped = BoundaryConditions::repeat_image(accessor, global_bounds);
        Func f, g, h;
        f(x, y) = clamped(x, y) + clamped(x+1, y);

        f.distribute(x);

        DistributedImage<int> out(100, 100);
        out.set_domain(x, y);
        out.placement().distribute(x);
        out.allocate();
        in.set_domain(x, y);
        in.placement().distribute(x);
        in.allocate(f, out);

        for (int y = 0; y < in.height(); y++) {
            for (int x = 0; x < in.width(); x++) {
                in(x, y) = in.global(0, x) + in.global(1, y);
            }
        }

        inp.set(in);

        f.realize(out.get_buffer());
        for (int y = 0; y < out.height(); y++) {
            for (int x = 0; x < out.width(); x++) {
                const int max = out.global_width() - 1;
                const int xp1 = out.global(0, x+1) > max ? 0 : out.global(0, x+1);

                const int correct = (out.global(0, x) + out.global(1, y)) + (xp1 + out.global(1, y));
                if (out(x, y) != correct) {
                    printf("[rank %d] out(%d,%d) = %d instead of %d\n", rank, x, y, out(x, y), correct);
                    MPI_Finalize();
                    return -1;
                }
            }
        }
    }

    {
        DistributedImage<int> in(10, 10);

        in.set_domain(x, y);
        in.placement().distribute(y);
        in.allocate();

        RDom r(0, in.width(), 0, in.height());
        Func f("ff"), g("gg");
        g(x, y) = in(x, y);
        g.compute_root();
        f() = maximum(r, g(r.x, r.y));

        for (int y = 0; y < in.height(); y++) {
            for (int x = 0; x < in.width(); x++) {
                in(x, y) = in.global(0, x) + in.global(1, y);
            }
        }

        int local_result = evaluate<int>(f);
        int global_result = 0;
        MPI_Allreduce(&local_result, &global_result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        const int local_correct = in.global(0, in.width() - 1) + in.global(1, in.height() - 1);
        if (local_result != local_correct) {
            printf("[rank %d] local result = %d instead of %d\n", rank, local_result, local_correct);
            MPI_Finalize();
            return -1;
        }

        const int global_correct = (in.global_width() - 1) + (in.global_height() - 1);
        if (global_result != global_correct) {
            printf("[rank %d] result = %d instead of %d\n", rank, global_result, global_correct);
            MPI_Finalize();
            return -1;
        }
    }

    {
        DistributedImage<double> U(10, 10, 10, 5, "U");

        Var cc;
        U.set_domain(x, y, z, cc);
        U.placement().distribute(z);
        U.allocate();

        Func Q("Q");
        Q(x, y, z, cc) = Expr(1.0) + U(x, y, z, cc);
        Q.bound(cc, 0, 5).unroll(cc);
        Q.compute_root().distribute(z).parallel(z);

        RDom r(0, U.global_extent(0), 0, U.global_extent(1), 0, U.global_extent(2));

        Expr c     = sqrt(Expr(3.0)*Q(r.x,r.y,r.z,4)/Q(r.x,r.y,r.z,0));
        Expr courx = Q(r.x,r.y,r.z,1);
        Expr coury = Q(r.x,r.y,r.z,2);
        Expr courz = Q(r.x,r.y,r.z,3);

        Expr huge = 100;
        Expr courmx = max( Expr(-huge), courx );
        Expr courmy = max( Expr(-huge), coury );
        Expr courmz = max( Expr(-huge), courz );

        Expr maxcourpt = max(courmx, max(courmy, courmz));

        Func helper("courno");
        helper() = maximum(r, maxcourpt);

        double local_result = evaluate<double>(helper);
        assert(local_result >= 0);
    }

    {
        const int w = 10;
        DistributedImage<int> in(w);
        Image<int> in_correct(w);

        // Under test
        Func clamped;
        clamped(x) = in(clamp(x, 0, in.global_width()-1));
        Func f, buffer;
        buffer(x) = clamped(x);
        buffer.compute_root().distribute(x);
        f(x) = buffer(x-1) + buffer(x+1);
        f.distribute(x);

        // Correct
        Func clamped1;
        clamped1(x) = in_correct(clamp(x, 0, in_correct.width()-1));
        Func f1, buffer1;
        buffer1(x) = clamped1(x);
        buffer1.compute_root();
        f1(x) = buffer1(x-1) + buffer1(x+1);

        in.set_domain(x);
        in.placement().distribute(x);
        in.allocate(f, in);

        for (int x = 0; x < in_correct.width(); x++) {
            in_correct(x) = x;
            if (in.mine(x)) {
                const int lx = in.local(0, x);
                in(lx) = x;
            }
        }

        const int niters = 5;
        for (int i = 0; i < niters; i++) {
            f.realize(in);
            f1.realize(in_correct);
        }

        for (int x = 0; x < in.width(); x++) {
            const int gx = in.global(0, x);
            const int correct = in_correct(gx);
            if (in(x) != correct) {
                printf("[rank %d] in(%d) = %d instead of %d\n", rank, x, in(x), correct);
                MPI_Finalize();
                return -1;
            }
        }

    }

    {
        const int w = 10;
        DistributedImage<int> in(w), out(w);

        Func clamped;
        clamped(x) = in(clamp(x, 0, in.global_width()-1));
        Func f, g;
        f(x) = clamped(x);
        f(x) += 1;
        g(x) = f(x);
        f.compute_rank().vectorize(x, 2);
        f.update().vectorize(x, 2);
        g.compute_root().distribute(x);

        out.set_domain(x);
        out.placement().distribute(x);
        out.allocate();

        in.set_domain(x);
        in.placement().distribute(x);
        in.allocate(g, out);

        for (int x = 0; x < in.width(); x++) {
            in(x) = in.global(0, x);
        }

        g.realize(out);

        for (int x = 0; x < out.width(); x++) {
            const int gx = out.global(0, x);
            const int correct = gx + 1;
            if (out(x) != correct) {
                printf("[rank %d] out(%d) = %d instead of %d\n", rank, x, out(x), correct);
                MPI_Finalize();
                return -1;
            }
        }
    }

    if (numprocs == 4) {
        DistributedImage<int> in(16, 16);

        Func clamped;
        clamped(x, y) = in(clamp(x, 0, in.global_width()-1),
                           clamp(y, 0, in.global_height()-1));
        Func f;
        f(x, y) = clamped(x, y);
        // Distribute f onto a grid of 2x2 processors:
        f.distribute(x, y, 2, 2);

        DistributedImage<int> out(16, 16);
        out.set_domain(x, y);
        out.placement().distribute(x, y, 2, 2);
        out.allocate();
        in.set_domain(x, y);
        in.placement().distribute(x, y, 2, 2);
        in.allocate(f, out);

        for (int y = 0; y < in.height(); y++) {
            for (int x = 0; x < in.width(); x++) {
                in(x, y) = in.global(0, x) + in.global(1, y);
            }
        }

        const float sqrtn = sqrtf(numprocs);
        const int isqrtn = (int)sqrtn;
        const int xslice = (int)ceil(in.global_width() / sqrtn);
        const int yslice = (int)ceil(in.global_height() / sqrtn);

        for (int r = 0; r < numprocs; r++) {
            if (rank == r) {
                printf("[rank %d] input from (%d,%d) to (%d,%d)\n",
                       rank, in.global(0, 0), in.global(1, 0),
                       in.global(0, in.width()-1), in.global(1, in.height()-1));
                fflush(stdout);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        assert(in.global(0, 0) == (rank % isqrtn) * xslice);
        assert(in.global(0, in.width()-1) == (rank % isqrtn) * xslice + (xslice-1));
        assert(in.local(0, (rank % isqrtn)*xslice) == 0);

        assert(in.global(1, 0) == (rank / isqrtn) * yslice);
        assert(in.global(1, in.height()-1) == (rank / isqrtn) * yslice + (yslice-1));
        assert(in.local(1, (rank / isqrtn)*yslice) == 0);

        assert(in.mine(0, 0) == (rank == 0));
        assert(in.mine(in.global_width() - 1, in.global_height() - 1) == (rank == numprocs - 1));
        assert(in.mine(-1, 0) == false);
        assert(in.mine(in.global_width(), in.global_height()) == false);
        if (numprocs > 1) {
            assert(in.mine(xslice, 0) == (rank == 1));
        }

        assert(out.global(0, 0) == (rank % isqrtn) * xslice);
        assert(out.global(0, out.width()-1) == (rank % isqrtn) * xslice + (xslice-1));
        assert(out.local(0, (rank % isqrtn)*xslice) == 0);

        assert(out.global(1, 0) == (rank / isqrtn) * yslice);
        assert(out.global(1, out.height()-1) == (rank / isqrtn) * yslice + (yslice-1));
        assert(out.local(1, (rank / isqrtn)*yslice) == 0);

        assert(out.mine(0, 0) == (rank == 0));
        assert(out.mine(out.global_width() - 1, out.global_height() - 1) == (rank == numprocs - 1));
        assert(out.mine(-1, 0) == false);
        assert(out.mine(out.global_width(), out.global_height()) == false);
        if (numprocs > 1) {
            assert(out.mine(xslice, 0) == (rank == 1));
        }

        f.realize(out.get_buffer());

        for (int y = 0; y < out.height(); y++) {
            for (int x = 0; x < out.width(); x++) {
                const int gx = out.global(0, x), gy = out.global(1, y);
                const int correct = gx + gy;
                if (out(x, y) != correct) {
                    printf("[rank %d] out(%d,%d) = %d instead of %d\n", rank, x, y, out(x, y), correct);
                    MPI_Abort(MPI_COMM_WORLD, -1);
                    return -1;
                }
            }
        }
    }

    {
        const auto s0 = approx_factors_near_sqrt(0), s1 = approx_factors_near_sqrt(1),
            s2 = approx_factors_near_sqrt(2), s4 = approx_factors_near_sqrt(4),
            s8 = approx_factors_near_sqrt(8), s9 = approx_factors_near_sqrt(9),
            s10 = approx_factors_near_sqrt(10), s17 = approx_factors_near_sqrt(17),
            s24 = approx_factors_near_sqrt(24), s85 = approx_factors_near_sqrt(85),
            s99 = approx_factors_near_sqrt(99);

        assert(s0 == std::make_pair(0, 0));
        assert(s1 == std::make_pair(1, 1));
        assert(s2 == std::make_pair(1, 2));
        assert(s4 == std::make_pair(2, 2));
        assert(s8 == std::make_pair(2, 4));
        assert(s9 == std::make_pair(3, 3));
        assert(s10 == std::make_pair(3, 3));
        assert(s17 == std::make_pair(4, 4));
        assert(s24 == std::make_pair(4, 6));
        assert(s85 == std::make_pair(9, 9));
        assert(s99 == std::make_pair(9, 11));

        const auto c0 = approx_factors_near_cubert(0), c1 = approx_factors_near_cubert(1),
            c2 = approx_factors_near_cubert(2), c4 = approx_factors_near_cubert(4),
            c8 = approx_factors_near_cubert(8), c16 = approx_factors_near_cubert(16),
            c17 = approx_factors_near_cubert(17), c27 = approx_factors_near_cubert(27),
            c85 = approx_factors_near_cubert(85), c99 = approx_factors_near_cubert(99),
            c100 = approx_factors_near_cubert(100);

        assert(vec_eq(c0, {0, 0, 0}));
        assert(vec_eq(c1, {1, 1, 1}));
        assert(vec_eq(c2, {1, 1, 2}));
        assert(vec_eq(c4, {1, 2, 2}));
        assert(vec_eq(c8, {2, 2, 2}));
        assert(vec_eq(c16, {2, 2, 4}));
        assert(vec_eq(c17, {2, 2, 4}));
        assert(vec_eq(c27, {3, 3, 3}));
        assert(vec_eq(c85, {4, 4, 5}));
        assert(vec_eq(c99, {4, 4, 6}));
        assert(vec_eq(c100, {4, 5, 5}));
    }

    {
        DistributedImage<int> in(100, 100);

        Func clamped;
        clamped(x, y) = in(clamp(x, 0, in.global_width()-1),
                           clamp(y, 0, in.global_height()-1));
        Func f;
        f(x, y) = clamped(x, y);
        // Use the helper function to determine the best process grid arrangement.
        std::pair<int, int> proc_grid = approx_factors_near_sqrt(numprocs);
        const int p = proc_grid.first, q = proc_grid.second;
        f.distribute(x, y, p, q);

        mpi_printf("Using process grid %dx%d\n", p, q);

        DistributedImage<int> out(100, 100);
        out.set_domain(x, y);
        out.placement().distribute(x, y, p, q);
        out.allocate();
        in.set_domain(x, y);
        in.placement().distribute(x, y, p, q);
        in.allocate(f, out);

        for (int y = 0; y < in.height(); y++) {
            for (int x = 0; x < in.width(); x++) {
                in(x, y) = in.global(0, x) + in.global(1, y);
            }
        }

        f.realize(out.get_buffer());
        for (int y = 0; y < out.height(); y++) {
            for (int x = 0; x < out.width(); x++) {
                const int gx = out.global(0, x), gy = out.global(1, y);
                const int correct = gx + gy;
                if (out(x, y) != correct) {
                    printf("[rank %d] out(%d,%d) = %d instead of %d\n", rank, x, y, out(x, y), correct);
                    MPI_Abort(MPI_COMM_WORLD, -1);
                    return -1;
                }
            }
        }
    }

    {
        DistributedImage<int> in(100, 100);

        Func clamped;
        clamped(x, y) = in(clamp(x, 0, in.global_width()-1),
                           clamp(y, 0, in.global_height()-1));
        Func f, g;
        f(x, y) = clamped(x, y);
        g(x, y) = f(x, y-1) + f(x, y+1);
        // Use the helper function to determine the best process grid arrangement.
        std::pair<int, int> proc_grid = approx_factors_near_sqrt(numprocs);
        const int p = proc_grid.first, q = proc_grid.second;
        f.compute_root().distribute(x, y, p, q);
        g.compute_root().distribute(x, y, p, q);

        DistributedImage<int> out(100, 100);
        out.set_domain(x, y);
        out.placement().distribute(x, y, p, q);
        out.allocate();
        in.set_domain(x, y);
        in.placement().distribute(x, y, p, q);
        in.allocate(g, out);

        for (int y = 0; y < in.height(); y++) {
            for (int x = 0; x < in.width(); x++) {
                in(x, y) = in.global(0, x) + in.global(1, y);
            }
        }

        g.realize(out.get_buffer());
        for (int y = 0; y < out.height(); y++) {
            for (int x = 0; x < out.width(); x++) {
                const int ymax = out.global_height() - 1;
                const int gyp1 = out.global(1, y+1) >= ymax ? ymax : out.global(1, y+1),
                    gym1 = out.global(1, y) == 0 ? 0 : out.global(1, y-1);
                const int gx = out.global(0, x);
                const int correct = gx + gym1 + gx + gyp1;
                if (out(x, y) != correct) {
                    printf("[rank %d] out(%d,%d) = %d instead of %d\n", rank, x, y, out(x, y), correct);
                    MPI_Abort(MPI_COMM_WORLD, -1);
                    return -1;
                }
            }
        }
    }

    {
        DistributedImage<int> in(10);

        Func clamped;
        clamped(x) = in(clamp(x, 0, in.global_width()-1));
        Func f;
        f(x) = clamped(x-1);
        f.distribute(x);
        Func init;
        init(x) = x;
        init.distribute(x);

        DistributedImage<int> out(10);
        out.set_domain(x);
        out.placement().distribute(x);
        out.allocate();
        in.set_domain(x);
        in.placement().distribute(x);
        in.allocate(f, out);

        init.realize(in);
        for (int x = 0; x < in.width(); x++) {
            const int correct = in.global(0, x);
            if (in(x) != correct) {
                printf("[rank %d] in(%d) = %d instead of %d\n", rank, x, in(x), correct);
                MPI_Abort(MPI_COMM_WORLD, -1);
                return -1;
            }
        }

        f.realize(out);
        for (int x = 0; x < out.width(); x++) {
            const int gxm1 = out.global(0, x) == 0 ? 0 : out.global(0, x-1);
            const int correct = gxm1;
            if (out(x) != correct) {
                printf("[rank %d] out(%d) = %d instead of %d\n", rank, x, out(x), correct);
                MPI_Finalize();
                return -1;
            }
        }
    }

    {
        DistributedImage<int> in(100, 100, 100);

        Func clamped;
        clamped(x, y, z) = in(clamp(x, 0, in.global_width()-1),
                           clamp(y, 0, in.global_height()-1),
                           clamp(z, 0, in.global_channels()-1));
        Func f, g;
        f(x, y, z) = clamped(x, y, z);
        g(x, y, z) = f(x, y-1, z) + f(x, y+1, z) + f(x, y, z-1);

        std::vector<int> proc_grid = approx_factors_near_cubert(numprocs);
        const int p = proc_grid[0], q = proc_grid[1], r = proc_grid[2];
        mpi_printf("Using process grid %dx%dx%d\n", p, q, r);

        f.compute_root().distribute(x, y, z, p, q, r);
        g.compute_root().distribute(x, y, z, p, q, r);

        DistributedImage<int> out(100, 100, 100);
        out.set_domain(x, y, z);
        out.placement().distribute(x, y, z, p, q, r);
        out.allocate();
        in.set_domain(x, y, z);
        in.placement().distribute(x, y, z, p, q, r);
        in.allocate(g, out);

        for (int z = 0; z < in.channels(); z++) {
            for (int y = 0; y < in.height(); y++) {
                for (int x = 0; x < in.width(); x++) {
                    in(x, y, z) = in.global(0, x) + in.global(1, y) + in.global(2, z);
                }
            }
        }

        g.realize(out.get_buffer());
        for (int z = 0; z < out.channels(); z++) {
            for (int y = 0; y < out.height(); y++) {
                for (int x = 0; x < out.width(); x++) {
                    const int ymax = out.global_height() - 1;
                    const int gy = out.global(1, y),
                        gyp1 = out.global(1, y+1) >= ymax ? ymax : out.global(1, y+1),
                        gym1 = out.global(1, y) == 0 ? 0 : out.global(1, y-1);
                    const int gz = out.global(2, z),
                        gzm1 = out.global(2, z) == 0 ? 0 : out.global(2, z-1);
                    const int gx = out.global(0, x);
                    const int correct = gx + gym1 + gz + gx + gyp1 + gz + gx + gy + gzm1;
                    if (out(x, y, z) != correct) {
                        printf("[rank %d] out(%d,%d,%d) = %d instead of %d\n", rank, x, y, z, out(x, y, z), correct);
                        MPI_Abort(MPI_COMM_WORLD, -1);
                        return -1;
                    }
                }
            }
        }
    }
#if 0
    {
        DistributedImage<int> in(512, 512, 512, 5);

        Func clamped;
        clamped(x, y, z, c) = in(clamp(x, 0, in.global_width()-1),
                                 clamp(y, 0, in.global_height()-1),
                                 clamp(z, 0, in.global_channels()-1),
                                 clamp(c, 0, in.global_extent(3)-1));
        Func f, g;
        f(x, y, z, c) = clamped(x, y, z, c);
        g(x, y, z, c) = f(x, y-1, z, c) + f(x, y+1, z, c) + f(x, y, z-1, c);

        std::vector<int> proc_grid = approx_factors_near_cubert(numprocs);
        const int p = proc_grid[0], q = proc_grid[1], r = proc_grid[2];
        mpi_printf("Using process grid %dx%dx%d\n", p, q, r);

        f.compute_root().distribute(x, y, z, p, q, r).parallel(z).vectorize(x, 4);
        g.compute_root().distribute(x, y, z, p, q, r).parallel(z).vectorize(x, 4);

        DistributedImage<int> out(512, 512, 512, 5);
        out.set_domain(x, y, z, c);
        out.placement().distribute(x, y, z, p, q, r).vectorize(x, 4);
        out.allocate();
        in.set_domain(x, y, z, c);
        in.placement().distribute(x, y, z, p, q, r).vectorize(x, 4);
        in.allocate(g, out);

        for (int c = 0; c < in.extent(3); c++) {
            for (int z = 0; z < in.channels(); z++) {
                for (int y = 0; y < in.height(); y++) {
                    for (int x = 0; x < in.width(); x++) {
                        in(x, y, z, c) = in.global(0, x) + in.global(1, y) + in.global(2, z) + in.global(3, c);
                    }
                }
            }
        }

        g.realize(out.get_buffer());
        for (int c = 0; c < out.extent(3); c++) {
            for (int z = 0; z < out.channels(); z++) {
                for (int y = 0; y < out.height(); y++) {
                    for (int x = 0; x < out.width(); x++) {
                        const int ymax = out.global_height() - 1;
                        const int gy = out.global(1, y),
                            gyp1 = out.global(1, y+1) >= ymax ? ymax : out.global(1, y+1),
                            gym1 = out.global(1, y) == 0 ? 0 : out.global(1, y-1);
                        const int gz = out.global(2, z),
                            gzm1 = out.global(2, z) == 0 ? 0 : out.global(2, z-1);
                        const int gx = out.global(0, x);
                        const int gc = out.global(3, c);
                        const int correct = gx + gym1 + gz + gc + gx + gyp1 + gz + gc + gx + gy + gzm1 + gc;
                        if (out(x, y, z, c) != correct) {
                            printf("[rank %d] out(%d,%d,%d,%d) = %d instead of %d\n", rank, x, y, z, c, out(x, y, z, c), correct);
                            MPI_Abort(MPI_COMM_WORLD, -1);
                            return -1;
                        }
                    }
                }
            }
        }
    }
#endif
    {
        const int w = 100;
        DistributedImage<int> in(w+2, w+2);

        // Test without clamp boundary condition
        Func accessor;
        accessor(x, y) = in(x+1, y+1);

        Func f;
        f(x, y) = accessor(x, y);
        f.compute_root().distribute(y);

        DistributedImage<int> out(w, w);
        out.set_domain(x, y);
        out.placement().distribute(y);
        out.allocate();
        in.set_domain(x, y);
        in.placement().distribute(y);
        in.allocate(f, out);

        // Fill in ghost zone.
        for (int y = 0; y < in.global_height(); y++) {
            for (int x = 0; x < in.global_width(); x++) {
                int vx = x - 1, vy = y - 1;
                if (x == 0) vx = 0;
                if (x == in.global_width() - 1) vx = w-1;
                if (y == 0) vy = 0;
                if (y == in.global_height() - 1) vy = w-1;

                if (in.mine(x, y)) {
                    in(in.local(0, x), in.local(1, y)) = vx + vy;
                }
            }
        }

        f.realize(out);
        for (int y = 0; y < out.height(); y++) {
            for (int x = 0; x < out.width(); x++) {
                const int correct = out.global(0, x) + out.global(1, y);
                if (out(x, y) != correct) {
                    printf("[rank %d] out(%d,%d) = %d instead of %d\n", rank, x, y, out(x, y), correct);
                    MPI_Finalize();
                    return -1;
                }
            }
        }
    }

    {
        const int w = 100;
        DistributedImage<int> in(w+2, w+2);

        Func accessor;
        accessor(x, y) = in(x+1, y+1);

        Func f, g;
        f(x, y) = (accessor(x-1, y) + accessor(x, y) + accessor(x+1, y)) / 3;
        g(x, y) = (f(x, y-1) + f(x, y) + f(x, y+1)) / 3;

        f.compute_root().distribute(y);
        g.compute_root().distribute(y);

        DistributedImage<int> out(w, w);
        out.set_domain(x, y);
        out.placement().distribute(y);
        out.allocate();
        in.set_domain(x, y);
        in.placement().distribute(y);
        in.allocate(g, out);

        for (int y = 0; y < in.global_height(); y++) {
            for (int x = 0; x < in.global_width(); x++) {
                int vx = x - 1, vy = y - 1;
                if (x == 0) vx = 0;
                if (x == in.global_width() - 1) vx = w-1;
                if (y == 0) vy = 0;
                if (y == in.global_height() - 1) vy = w-1;

                if (in.mine(x, y)) {
                    in(in.local(0, x), in.local(1, y)) = vx + vy;
                }
            }
        }

        g.realize(out);

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

    printf("Rank %d Success!\n", rank);

    MPI_Finalize();
    return 0;
}
