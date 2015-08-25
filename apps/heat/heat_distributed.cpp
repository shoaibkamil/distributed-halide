#include "Halide.h"
#include "mpi_timing.h"
using namespace Halide;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, numprocs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    const int w = std::stoi(argv[1]), h = std::stoi(argv[2]), d = std::stoi(argv[3]);
    Var x("x"), y("y"), z("z"), xi("xi"), yi("yi");

    // Declare our input and output in global width and height.
    DistributedImage<double> input(w, h, d);
    DistributedImage<double> output(w, h, d);
    input.set_domain(x, y, z);
    input.placement().distribute(z);
    input.allocate();
    for (int z = 0; z < input.channels(); z++) {
        for (int y = 0; y < input.height(); y++) {
            for (int x = 0; x < input.width(); x++) {
                input(x, y) = input.global(0, x) + input.global(1, y);
            }
        }
    }
    output.set_domain(x, y, z);
    output.placement().distribute(z);
    output.allocate();

    Func clamped;
    clamped(x, y, z) = input(clamp(x, 0, input.global_width() - 1),
                             clamp(y, 0, input.global_height() - 1),
                             clamp(z, 0, input.global_channels() - 1));
    Func heat3d("heat3d"), buffered;
    Expr c0 = cast<double>(0.5f), c1 = cast<double>(-0.25f);
    buffered(x, y, z) = clamped(x, y, z);
    heat3d(x, y, z) = c0 * buffered(x, y, z) + c1 * (buffered(x-1, y, z) + buffered(x+1, y, z) +
                                                  buffered(x, y-1, z) + buffered(x, y+1, z) +
                                                  buffered(x, y, z-1) + buffered(x, y, z+1));
    // This has the effect of double-buffering the input:
    buffered.compute_root().distribute(z);
    heat3d.distribute(z);

    // Realize once to compile
    heat3d.realize(input.get_buffer());
    // Run the program and test output for correctness
    const int niters = 100;
    MPITiming timing(MPI_COMM_WORLD);
    timing.barrier();
    timeval t1, t2;
    gettimeofday(&t1, NULL);
    for (int i = 0; i < niters; i++) {
        heat3d.realize(input.get_buffer());
    }
    gettimeofday(&t2, NULL);
    float t = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0f;
    timing.record(t);
    timing.reduce(MPITiming::Median);

    timing.gather(MPITiming::Max);
    timing.report();
    if (rank == 0) {
        printf("Heat test succeeded!\n");
    }
    MPI_Finalize();
    return 0;
}
