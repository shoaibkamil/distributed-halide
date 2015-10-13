#include "Halide.h"
#include "mpi_timing.h"

using namespace Halide;
#include "image_io.h"
#include <iostream>
#include <limits>

#include <sys/time.h>

std::default_random_engine generator(0);
std::uniform_real_distribution<float> distribution(0, 1);

DistributedImage<float> input, output;
Image<float> global_input, global_output;
Var x("x"), y("y"), c("c"), k, xi, yo, yi;

bool float_eq(float a, float b) {
    const float thresh = 1e-5;
    return a == b || (std::abs(a - b) / b) < thresh;
}

float rndflt() {
    return distribution(generator);
}

Func build(bool distributed) {
    Func clamped;
    if (distributed) {
        clamped(x, y) = input(clamp(x, 0, input.global_width() - 1),
                              clamp(y, 0, input.global_height() - 1));
    } else {
        clamped(x, y) = global_input(clamp(x, 0, global_input.width() - 1),
                                     clamp(y, 0, global_input.height() - 1));
    }
    Func upsample_x("upsample_x"), upsample_y("upsample_y");
    upsample_x(x, y) = (clamped((x-1)/2, y) + clamped((x+1)/2, y)) / 2;
    upsample_y(x, y) = (upsample_x(x, (y-1)/2) + upsample_x(x, (y+1)/2)) / 2;
    Func resized("resized");
    resized(x, y) = upsample_y(x, y);

    // upsample_x.compute_root().split(y, y, yi, 16).parallel(y).distribute(x);
    // resized.parallel(y).distribute(x);
    resized.parallel(y);
    if (distributed) {
        resized.distribute(y);
    }
    return resized;
}

int main(int argc, char **argv) {
    int req = MPI_THREAD_MULTIPLE, prov;
    MPI_Init_thread(&argc, &argv, req, &prov);
    assert(prov == req);
    int rank = 0, numprocs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    // global_input = load<float>(argv[1]);
    // global_output = Image<float>(global_input.width() * 2, global_input.height() * 2);
    // const int w = global_input.width(), h = global_input.height();

    const int w = std::stoi(argv[1]), h = std::stoi(argv[2]);
    const int ow = w*2, oh = h*2;

    input = DistributedImage<float>(w, h);
    output = DistributedImage<float>(ow, oh);
    // global_input = Image<float>(w, h);
    // global_output = Image<float>(ow, oh);

    Func resize_distributed = build(true);
    // Func resize_correct = build(false);

    output.set_domain(x, y);
    output.placement().distribute(y);
    output.allocate();
    input.set_domain(x, y);
    input.placement();
    input.allocate(resize_distributed, output);

    // for (int y = 0; y < h; y++) {
    //     for (int x = 0; x < w; x++) {
    //         if (input.mine(x, y)) {
    //             int lx = input.local(0, x), ly = input.local(1, y);
    //             input(lx, ly) = global_input(x, y);
    //         }
    //     }
    // }
    for (int y = 0; y < input.height(); y++) {
        for (int x = 0; x < input.width(); x++) {
            input(x, y) = x + y;
        }
    }

    //resize_correct.realize(global_output);
    resize_distributed.realize(output.get_buffer());
    //save(global_output, argv[2]);

    const int niters = 50;
    MPITiming timing(MPI_COMM_WORLD);
    timing.barrier();
    for (int i = 0; i < niters; i++) {
        timing.start();
        resize_distributed.realize(output.get_buffer());
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
        printf("Upsample test succeeded!\n");
    }

    MPI_Finalize();
    return 0;
}
