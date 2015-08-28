#include "Halide.h"
#include "mpi_timing.h"

#include <stdio.h>
#include <memory>

using namespace Halide;

std::default_random_engine generator(0);
std::uniform_real_distribution<float> distribution(10, 500);
Var x("x"), y("y");

float rndflt() {
    return distribution(generator);
}

bool float_eq(float a, float b) {
    const float thresh = 1e-5;
    return a == b || (std::abs(a - b) / b) < thresh;
}

Func build(Func input, bool distributed) {
    Func block, block_transpose, output;
    block(x, y) = input(x, y);
    block_transpose(x, y) = block(y, x);
    output(x, y) = block_transpose(x, y);

    Var xi, yi;

    if (distributed) {
        block.compute_root().distribute(y);
        block_transpose.compute_root().distribute(y);
    } else {
        // Do 8 vectorized loads from the input.
        block.compute_at(output, x).vectorize(x).unroll(y);
        block_transpose.compute_at(output, x).vectorize(y).unroll(x);
        output.tile(x, y, xi, yi, 8, 8).vectorize(xi).unroll(yi);
    }

    return output;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, numprocs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    const int w = std::stoi(argv[1]), h = std::stoi(argv[2]);

    Image<float> global_input(w, h), global_output(w, h);
    DistributedImage<float> input(w, h), output(w, h);
    input.set_domain(x, y);
    input.placement().distribute(y);
    input.allocate();
    output.set_domain(x, y);
    output.placement().distribute(y);
    output.allocate();

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float v = rndflt();
            if (input.mine(x, y)) {
                int lx = input.local(0, x), ly = input.local(1, y);
                input(lx, ly) = v;
            }
            global_input(x, y) = v;
        }
    }

    Func accessor, global_accessor;
    accessor(x, y) = input(x, y);
    global_accessor(x, y) = global_input(x, y);

    Func transpose_correct = build(global_accessor, false);
    Func transpose_distributed = build(accessor, true);

    transpose_distributed.distribute(y);

    transpose_correct.realize(global_output);
    transpose_distributed.realize(output.get_buffer());

    const int niters = 10;
    MPITiming timing(MPI_COMM_WORLD);
    timing.barrier();
    timeval t1, t2;
    for (int i = 0; i < niters; i++) {
        gettimeofday(&t1, NULL);
        transpose_distributed.realize(output.get_buffer());
        gettimeofday(&t2, NULL);
        float t = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0f;
        timing.record(t);
    }
    timing.reduce(MPITiming::Median);

    for (int y = 0; y < output.height(); y++) {
        for (int x = 0; x < output.width(); x++) {
            int gx = output.global(0, x), gy = output.global(1, y);
            if (!float_eq(output(x, y), global_output(gx, gy))) {
                printf("[rank %d] output(%d,%d) = %f instead of %f\n", rank, x, y, output(x, y), global_output(gx, gy));
                MPI_Abort(MPI_COMM_WORLD, 1);
                MPI_Finalize();
                return -1;
            }
        }
    }

    timing.gather(MPITiming::Max);
    timing.report();

    if (rank == 0) {
        printf("Transpose test succeeded!\n");
    }
    MPI_Finalize();
    return 0;
}
