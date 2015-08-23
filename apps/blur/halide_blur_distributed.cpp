#include <sys/time.h>
#include "Halide.h"
using namespace Halide;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, numprocs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    MPI_Barrier(MPI_COMM_WORLD);
    timeval e2e_start, e2e_end;
    gettimeofday(&e2e_start, NULL);

    const int w = std::stoi(argv[1]), h = std::stoi(argv[2]);
    Func blur_x("blur_x"), blur_y("blur_y");
    Var x("x"), y("y"), xi("xi"), yi("yi");

    // Declare our input and output in global width and height.
    DistributedImage<int> input(w, h);
    DistributedImage<int> output(w, h);
    // Set domain and data distribution of input buffer.
    input.set_domain(x, y);
    input.placement().distribute(y);
    input.allocate();

    // Initialize my (local) input. We use global coordinates so that
    // it is clear if the data is distributed.
    for (int y = 0; y < input.height(); y++) {
        for (int x = 0; x < input.width(); x++) {
            input(x, y) = input.global(0, x) + input.global(1, y);
        }
    }

    // Boundary conditions: don't go beyond global image bounds.
    Func clamped;
    clamped(x, y) = input(clamp(x, 0, input.global_width() - 1),
                          clamp(y, 0, input.global_height() - 1));
    // The algorithm
    blur_x(x, y) = (clamped(x-1, y) + clamped(x, y) + clamped(x+1, y))/3;
    blur_y(x, y) = (blur_x(x, y-1) + blur_x(x, y) + blur_x(x, y+1))/3;

    // How to schedule it
    blur_y.split(y, y, yi, 8).parallel(y).vectorize(x, 8);
    blur_x.store_at(blur_y, y).compute_at(blur_y, yi).vectorize(x, 8);
    blur_y.distribute(y);

    // Set domain and data distribution of output buffer. A current
    // limitation is that the distribution of the output buffer must
    // match the distribution of the last pipeline stage.
    output.set_domain(x, y);
    output.placement().split(y, y, yi, 8).distribute(y);
    output.allocate();
    // Run the program and test output for correctness
    blur_y.realize(output.get_buffer());
    timeval t1, t2;
    const int niters = 10;
    gettimeofday(&t1, NULL);
    for (int i = 0; i < niters; i++) {
        blur_y.realize(output.get_buffer());
    }
    gettimeofday(&t2, NULL);
    float sum = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0f;

    //printf("[rank %d] avg time\t%.3f\tseconds\n", rank, sum/niters);

    for (int y = 0; y < output.height(); y++) {
        for (int x = 0; x < output.width(); x++) {
            const int xmax = output.global_width() - 1, ymax = output.global_height() - 1;
            const int gxp1 = output.global(0, x+1) >= xmax ? xmax : output.global(0, x+1),
                gxm1 = output.global(0, x) == 0 ? 0 : output.global(0, x-1);
            const int gyp1 = output.global(1, y+1) >= ymax ? ymax : output.global(1, y+1),
                gym1 = output.global(1, y) == 0 ? 0 : output.global(1, y-1);
            const int gx = output.global(0, x), gy = output.global(1, y);
            const int correct = (((gxm1 + gym1 + gx + gym1 + gxp1 + gym1)/3) +
                                 ((gxm1 + gy + gx + gy + gxp1 + gy)/3) +
                                 ((gxm1 + gyp1 + gx + gyp1 + gxp1 + gyp1)/3)) / 3;
            if (output(x, y) != correct) {
                printf("[rank %d] output(%d,%d) = %d instead of %d\n", rank, x, y, output(x, y), correct);
                MPI_Finalize();
                return -1;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    gettimeofday(&e2e_end, NULL);
    float e2e_sec = (e2e_end.tv_sec - e2e_start.tv_sec) + (e2e_end.tv_usec - e2e_start.tv_usec) / 1000000.0f;
    float total = e2e_sec;
    if (rank == 0) {
        for (int i = 1; i < numprocs; i++) {
            float t;
            MPI_Recv(&t, 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total += t;
        }
    } else {
        MPI_Send(&e2e_sec, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("Blur test succeeded!\n");
        printf("Timing: <%d> ranks <%.3f> seconds\n", numprocs, total/numprocs);
    }
    MPI_Finalize();
    return 0;
}
