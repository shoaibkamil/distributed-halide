#include "Halide.h"
#include "mpi_timing.h"

using namespace Halide;

#include <halide_image_io.h>
#include <iostream>
#include <limits>
#include <sys/time.h>

using std::vector;

std::default_random_engine generator(0);
std::uniform_real_distribution<float> distribution(0, 1);

DistributedImage<float> input, output;
Image<float> global_input, global_output;
Var x("x"), y("y"), c("c"), xi, yi;

bool float_eq(float a, float b) {
    const float thresh = 1e-5;
    return a == b || (std::abs(a - b) / b) < thresh;
}

float rndflt() {
    return distribution(generator);
}

Func build(bool distributed) {
    const int levels = 10;
    Func downsampled[levels];
    Func downx[levels];
    Func interpolated[levels];
    Func upsampled[levels];
    Func upsampledx[levels];

    //Func clamped = BoundaryConditions::repeat_edge(input);
    Func clamped;
    if (distributed) {
        clamped(x, y, c) = input(clamp(x, 0, input.global_width() - 1),
                                 clamp(y, 0, input.global_height() - 1),
                                 clamp(c, 0, input.global_channels() - 1));
    } else {
        clamped(x, y, c) = global_input(clamp(x, 0, global_input.width() - 1),
                                        clamp(y, 0, global_input.height() - 1),
                                        clamp(c, 0, global_input.channels() - 1));
    }

    Expr width, height;
    if (distributed) {
        width = input.global_width();
        height = input.global_height();
    } else {
        width = global_input.width();
        height = global_input.height();
    }

    // This triggers a bug in llvm 3.3 (3.2 and trunk are fine), so we
    // rewrite it in a way that doesn't trigger the bug. The rewritten
    // form assumes the input alpha is zero or one.
    // downsampled[0](x, y, c) = select(c < 3, clamped(x, y, c) * clamped(x, y, 3), clamped(x, y, 3));
    downsampled[0](x, y, c) = clamped(x, y, c) * clamped(x, y, 3);

    for (int l = 1; l < levels; ++l) {
        Func prev = downsampled[l-1];

        if (l == 4) {
            // Also add a boundary condition at a middle pyramid level
            // to prevent the footprint of the downsamplings to extend
            // too far off the base image. Otherwise we look 512
            // pixels off each edge.
            Expr w = width/(1 << l);
            Expr h = height/(1 << l);
            prev = lambda(x, y, c, prev(clamp(x, 0, w), clamp(y, 0, h), c));
        }

        downx[l](x, y, c) = (prev(x*2-1, y, c) +
                             2.0f * prev(x*2, y, c) +
                             prev(x*2+1, y, c)) * 0.25f;
        downsampled[l](x, y, c) = (downx[l](x, y*2-1, c) +
                                   2.0f * downx[l](x, y*2, c) +
                                   downx[l](x, y*2+1, c)) * 0.25f;
    }
    interpolated[levels-1](x, y, c) = downsampled[levels-1](x, y, c);
    for (int l = levels-2; l >= 0; --l) {
        upsampledx[l](x, y, c) = (interpolated[l+1](x/2, y, c) +
                                  interpolated[l+1]((x+1)/2, y, c)) / 2.0f;
        upsampled[l](x, y, c) =  (upsampledx[l](x, y/2, c) +
                                  upsampledx[l](x, (y+1)/2, c)) / 2.0f;
        interpolated[l](x, y, c) = downsampled[l](x, y, c) + (1.0f - downsampled[l](x, y, 3)) * upsampled[l](x, y, c);
    }

    Func normalize("normalize");
    normalize(x, y, c) = interpolated[0](x, y, c) / interpolated[0](x, y, 3);

    Func final("final");
    final(x, y, c) = normalize(x, y, c);

    const int sched = 2;

    switch (sched) {
    case 0:
    {
        std::cout << "Flat schedule." << std::endl;
        for (int l = 0; l < levels; ++l) {
            downsampled[l].compute_root();
            interpolated[l].compute_root();
        }
        final.compute_root();
        break;
    }
    case 1:
    {
        std::cout << "Flat schedule with vectorization." << std::endl;
        for (int l = 0; l < levels; ++l) {
            downsampled[l].compute_root().vectorize(x,4);
            interpolated[l].compute_root().vectorize(x,4);
        }
        final.compute_root();
        break;
    }
    case 2:
    {
        Var xi, yi;
        std::cout << "Flat schedule with parallelization + vectorization." << std::endl;
        for (int l = 1; l < levels-1; ++l) {
            // if (l > 0) downsampled[l].compute_root().parallel(y).reorder(c, x, y).reorder_storage(c, x, y).vectorize(c, 4);
            // interpolated[l].compute_root().parallel(y).reorder(c, x, y).reorder_storage(c, x, y).vectorize(c, 4);
            // interpolated[l].unroll(x, 2).unroll(y, 2);
            if (l > 0) downsampled[l].compute_root().parallel(y).reorder(c, x, y).vectorize(c, 4);
            interpolated[l].compute_root().parallel(y).reorder(c, x, y).vectorize(c, 4);
            interpolated[l].unroll(x, 2).unroll(y, 2);
        }
        final.reorder(c, x, y).bound(c, 0, 3).parallel(y);
        final.tile(x, y, xi, yi, 2, 2).unroll(xi).unroll(yi);
        final.bound(x, 0, width);
        final.bound(y, 0, height);
        break;
    }
    case 3:
    {
        std::cout << "Flat schedule with vectorization sometimes." << std::endl;
        for (int l = 0; l < levels; ++l) {
            if (l + 4 < levels) {
                Var yo,yi;
                downsampled[l].compute_root().vectorize(x,4);
                interpolated[l].compute_root().vectorize(x,4);
            } else {
                downsampled[l].compute_root();
                interpolated[l].compute_root();
            }
        }
        final.compute_root();
        break;
    }
    default:
        assert(0 && "No schedule with this number.");
    }

    if (distributed) {
        for (int l = 1; l < 5; ++l) {
            downsampled[l].distribute(y);
            interpolated[l].distribute(y);
        }
        final.distribute(y);
    }

    return final;
}

int main(int argc, char **argv) {
    int req = MPI_THREAD_MULTIPLE, prov;
    MPI_Init_thread(&argc, &argv, req, &prov);
    assert(prov == req);
    int rank = 0, numprocs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    const int w = std::stoi(argv[1]), h = std::stoi(argv[2]), d = 4;
    const int ow = w, oh = h, od = 3;

    input = DistributedImage<float>(w, h, d);
    output = DistributedImage<float>(ow, oh, od);
    // global_input = Image<float>(w, h, d);
    // global_output = Image<float>(ow, oh, od);

    // Func interpolated_correct = build(false);
    Func interpolated_distributed = build(true);

    output.set_domain(x, y, c);
    output.placement().tile(x, y, xi, yi, 2, 2).distribute(y);
    output.allocate();
    input.set_domain(x, y, c);
    input.placement().distribute(y);
    input.allocate(interpolated_distributed, output);

    assert(input.channels() == 4);

    for (int z = 0; z < input.channels(); z++) {
        for (int y = 0; y < input.height(); y++) {
            for (int x = 0; x < input.width(); x++) {
                int gx = input.global(0, x), gy = input.global(1, y), gz = input.global(2, z);
                float v = gx+gy+gz;
                input(x, y, z) = v;
            }
        }
    }

    // JIT compile the pipeline eagerly, so we don't interfere with timing
    Target target = get_target_from_environment();
    interpolated_distributed.compile_jit(target);
    // interpolated_correct.realize(global_output);

    const int niters = 50;
#ifdef USE_MPIP
    MPI_Pcontrol(1);
#endif
    MPITiming timing(MPI_COMM_WORLD);
    timing.barrier();
    for (int i = 0; i < niters; i++) {
        timing.start();
        interpolated_distributed.realize(output.get_buffer());
        MPITiming::timing_t t = timing.stop();
        timing.record(t);
    }
    timing.reduce(MPITiming::Median);

    // for (int c = 0; c < output.channels(); c++) {
    //     for (int y = 0; y < output.height(); y++) {
    //         for (int x = 0; x < output.width(); x++) {
    //             int gx = output.global(0, x), gy = output.global(1, y), gc = output.global(2, c);
    //             if (!float_eq(output(x, y, c), global_output(gx, gy, gc))) {
    //                 printf("[rank %d] output(%d,%d,%d) = %f instead of %f\n", rank, x, y, c, output(x, y, c), global_output(gx, gy, gc));
    //                 MPI_Abort(MPI_COMM_WORLD, 1);
    //                 MPI_Finalize();
    //                 return -1;
    //             }
    //         }
    //     }
    // }

    timing.gather(MPITiming::Max);
    timing.report();
    if (rank == 0) {
        printf("Interpolate test succeeded!\n");
    }

    MPI_Finalize();
    return 0;
}
