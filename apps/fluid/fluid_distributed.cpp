#include "Halide.h"
#include "mpi_timing.h"
using namespace Halide;

int main(int argc, char **argv) {
    int req = MPI_THREAD_MULTIPLE, prov;
    MPI_Init_thread(&argc, &argv, req, &prov);
    assert(prov == req);
    int rank = 0, numprocs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    const int w = std::stoi(argv[1]), h = std::stoi(argv[2]), d = std::stoi(argv[3]);
    Var x("x"), y("y"), z("z"), xi("xi"), yi("yi");

    // Declare our input and output in global width and height.
    DistributedImage<float> input(w, h, d);
    DistributedImage<float> output(w, h, d);

    Func clamped;
    clamped(x, y, z) = input(clamp(x, 0, input.global_width() - 1),
                             clamp(y, 0, input.global_height() - 1),
                             clamp(z, 0, input.global_channels() - 1));
    Func heat3d;
    Expr c0 = 0.5f, c1 = -0.25f;
    heat3d(x, y, z) = c0 * clamped(x, y, z) + c1 * (clamped(x-1, y, z) + clamped(x+1, y, z) +
                                                  clamped(x, y-1, z) + clamped(x, y+1, z) +
                                                  clamped(x, y, z-1) + clamped(x, y, z+1));
    heat3d
        .tile(x, y, xi, yi, 8, 8).vectorize(xi).unroll(yi)
        .parallel(z)
        .distribute(z);

    output.set_domain(x, y, z);
    output.placement().distribute(z);
    output.allocate();

    input.set_domain(x, y, z);
    input.placement().distribute(z);
    input.allocate(heat3d, output);

    for (int z = 0; z < input.channels(); z++) {
        for (int y = 0; y < input.height(); y++) {
            for (int x = 0; x < input.width(); x++) {
                input(x, y, z) = input.global(0, x) + input.global(1, y) + input.global(2, z);
            }
        }
    }

    // Realize once to compile and allocate
    heat3d.realize(input.get_buffer());
    // Run the program and test output for correctness
    const int niters = 100;
    MPITiming timing(MPI_COMM_WORLD);
    timing.barrier();
    for (int i = 0; i < niters; i++) {
        timing.start();
        heat3d.realize(output.get_buffer());
        MPITiming::timing_t t = timing.stop();
        timing.record(t);
    }
    timing.reduce(MPITiming::Median);

    timing.gather(MPITiming::Max);
    timing.report();
    MPI_Finalize();
    return 0;
}
