#include "Halide.h"
#include "mpi.h"
#include <stdio.h>

using namespace Halide;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    Var x, y;
    {
        Func f;
        f(x, y) = x + y;

        f.compute_root().distribute(x);

        Image<int> im = f.realize(10, 10);
        for (int y = 0; y < im.height(); y++) {
            for (int x = 0; x < im.width(); x++) {
                int correct = x + y;
                if (im(x, y) != correct) {
                    printf("im(%d, %d) = %d instead of %d\n", x, y, im(x, y), correct);
                    return -1;
                }
            }
        }
    }

    printf("Success!\n");

    MPI_Finalize();
    return 0;
}
