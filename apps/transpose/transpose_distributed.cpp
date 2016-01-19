#include "Halide.h"
#include "mpi_timing.h"

#include <stdio.h>
#include <memory>

using namespace Halide;

std::default_random_engine generator(0);
std::uniform_real_distribution<float> distribution(10, 500);
Var x("x"), y("y"), xi, yi, tile;

float rndflt() {
    return distribution(generator);
}

bool float_eq(float a, float b) {
    const float thresh = 1e-5;
    return a == b || (std::abs(a - b) / b) < thresh;
}

// Hack to make sure we use the best non-distributed schedule for the
// baseline when running scalability tests.
bool actually_distributed() {
    char *e = getenv("HL_DISABLE_DISTRIBUTED");
    // disable = 0 => return true
    // disable != 0 => return false
    return !e || atoi(e) == 0;
}

Func build(Func input, bool distributed) {
    Func block, block_transpose, output;
    block(x, y) = input(x, y);
    block_transpose(x, y) = block(y, x);
    output(x, y) = block_transpose(x, y);

    if (distributed && actually_distributed()) {
        output.split(x, x, xi, 16).vectorize(xi);
        output.parallel(y);
        output.distribute(y);
    } else {
        output.tile(x, y, xi, yi, 16, 16).vectorize(xi).unroll(yi);
        output.parallel(y);
    }

    return output;
}

int main(int argc, char **argv) {
    int req = MPI_THREAD_MULTIPLE, prov;
    MPI_Init_thread(&argc, &argv, req, &prov);
    assert(prov == req);
    int rank = 0, numprocs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    const int w = std::stoi(argv[1]), h = std::stoi(argv[2]);

    // Image<float> global_input(w, h), global_output(w, h);
    DistributedImage<float> input(w, h), output(h, w);

    Func accessor, global_accessor;
    accessor(x, y) = input(x, y);
    // global_accessor(x, y) = global_input(x, y);

    // Func transpose_correct = build(global_accessor, false);
    Func transpose_distributed = build(accessor, true);

    output.set_domain(x, y);
    //output.placement().tile(x, y, xi, yi, 16, 16).distribute(y);
    output.placement().distribute(y);
    output.allocate();
    input.set_domain(x, y);
    input.placement().distribute(x);
    input.allocate(transpose_distributed, output);

    for (int y = 0; y < input.height(); y++) {
        for (int x = 0; x < input.width(); x++) {
            int gx = input.global(0, x), gy = input.global(1, y);
            float v = gx+gy; //rndflt();
            input(x, y) = v;
        }
    }

    // transpose_correct.realize(global_output);
    transpose_distributed.realize(output.get_buffer());

    const int niters = 50;
#ifdef USE_MPIP
    MPI_Pcontrol(1);
#endif
    MPITiming timing(MPI_COMM_WORLD);
    timing.barrier();
    for (int i = 0; i < niters; i++) {
        timing.start();
        transpose_distributed.realize(output.get_buffer());
        MPITiming::timing_t t = timing.stop();
        timing.record(t);
    }
    timing.reduce(MPITiming::Median);

    // for (int y = 0; y < output.height(); y++) {
    //     for (int x = 0; x < output.width(); x++) {
    //         int gx = output.global(0, x), gy = output.global(1, y);
    //         if (!float_eq(output(x, y), global_output(gx, gy))) {
    //             printf("[rank %d] output(%d,%d) = %f instead of %f\n", rank, x, y, output(x, y), global_output(gx, gy));
    //             MPI_Abort(MPI_COMM_WORLD, 1);
    //             MPI_Finalize();
    //             return -1;
    //         }
    //     }
    // }

    timing.gather(MPITiming::Max);
    timing.report();

    if (rank == 0) {
        printf("Transpose test succeeded!\n");
    }
    MPI_Finalize();
    return 0;
}
