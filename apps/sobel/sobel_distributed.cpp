#include "Halide.h"
using namespace Halide;

#include "halide_image_io.h"
#include "mpi_timing.h"

DistributedImage<float> input, output;
Image<float> global_input, global_output;

Var x("x"), y("y");

std::default_random_engine generator(0);
std::uniform_real_distribution<float> distribution(0, 1);

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
    Func sobelx, sobely;
    sobelx(x, y) = -1*clamped(x-1, y-1) + clamped(x+1, y-1) +
        -2*clamped(x-1, y) + 2*clamped(x+1, y) +
        -1*clamped(x-1, y+1) + clamped(x+1, y+1);

    sobely(x, y) = -1*clamped(x-1, y-1) + -2*clamped(x, y-1) + -1*clamped(x+1,y-1) +
        clamped(x-1, y+1) + 2*clamped(x, y+1) + clamped(x+1, y+1);

    Func sobel;
    sobel(x, y) = sqrt(sobelx(x, y) * sobelx(x, y) + sobely(x, y) * sobely(x, y));

    sobelx.vectorize(x, 8).compute_at(sobel, y);
    sobely.vectorize(x, 8).compute_at(sobel, y);
    sobel.parallel(y);

    if (distributed) {
        sobel.distribute(y);
    }
    return sobel;
}

int main(int argc, char **argv) {
    int req = MPI_THREAD_MULTIPLE, prov;
    MPI_Init_thread(&argc, &argv, req, &prov);
    assert(prov == req);
    int rank = 0, numprocs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    const int w = std::stoi(argv[1]), h = std::stoi(argv[2]);

    // Image<float> global_input = load<float>(argv[1]);
    // Image<float> global_output(global_input.width(), global_input.height());

    // const int w = global_input.width(), h = global_input.height();


    input = DistributedImage<float>(w, h);
    output = DistributedImage<float>(w, h);
    // global_input = Image<float>(w, h);
    // global_output = Image<float>(w, h);

    Func sobel_distributed = build(true);
    // Func sobel_correct = build(false);

    output.set_domain(x, y);
    output.placement().distribute(y);
    output.allocate();

    input.set_domain(x, y);
    input.placement().distribute(y);
    input.allocate(sobel_distributed, output);

    // for (int y = 0; y < h; y++) {
    //     for (int x = 0; x < w; x++) {
    //         float v = x + y;
    //         if (input.mine(x, y)) {
    //             int lx = input.local(0, x), ly = input.local(1, y);
    //             //input(lx, ly) = global_input(x, y);
    //             input(lx, ly) = v;
    //         }
    //         //global_input(x, y) = v;
    //     }
    // }

    for (int y = 0; y < input.height(); y++) {
        for (int x = 0; x < input.width(); x++) {
            input(x, y) = input.global(0, x) + input.global(1, y);
        }
    }

    sobel_distributed.realize(output.get_buffer());
    // sobel_correct.realize(global_output);

    const int niters = 50;
#ifdef USE_MPIP
    MPI_Pcontrol(1);
#endif
    MPITiming timing(MPI_COMM_WORLD);
    timing.barrier();
    for (int i = 0; i < niters; i++) {
        timing.start();
        sobel_distributed.realize(output.get_buffer());
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
        printf("Sobel test succeeded!\n");
    }

    // if (rank == 0) {
    //     if (numprocs > 1) {
    //         for (int r = 1; r < numprocs; r++) {
    //             MPI_Status status;
    //             int numrows, rowwidth, startx, starty;
    //             MPI_Recv(&numrows, 1, MPI_INT, r, 0, MPI_COMM_WORLD, &status);
    //             MPI_Recv(&rowwidth, 1, MPI_INT, r, 0, MPI_COMM_WORLD, &status);
    //             MPI_Recv(&startx, 1, MPI_INT, r, 0, MPI_COMM_WORLD, &status);
    //             MPI_Recv(&starty, 1, MPI_INT, r, 0, MPI_COMM_WORLD, &status);

    //             int maxcount = output.global_width();
    //             float *data = new float[maxcount];
    //             for (int i = 0; i < numrows; i++) {
    //                 MPI_Recv(data, maxcount, MPI_FLOAT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //                 int gx = startx, gy = starty + i;
    //                 for (int i = 0; i < rowwidth; gx++, i++) {
    //                     global_output(gx, gy) = data[i];
    //                 }
    //             }
    //             delete[] data;
    //         }
    //     }

    //     for (int y = 0; y < h; y++) {
    //         for (int x = 0; x < w; x++) {
    //             if (output.mine(x, y)) {
    //                 int lx = output.local(0, x), ly = output.local(1, y);
    //                 global_output(x, y) = output(lx, ly);
    //             }
    //         }
    //     }
    //     save(global_output, argv[2]);
    // } else {
    //     int numrows = output.height(), rowwidth = output.width(),
    //         startx = output.global(0, 0), starty = output.global(1, 0);
    //     MPI_Send(&numrows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    //     MPI_Send(&rowwidth, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    //     MPI_Send(&startx, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    //     MPI_Send(&starty, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    //     for (int y = 0; y < output.height(); y++) {
    //         int count = output.width();
    //         float *data = new float[count];
    //         for (int x = 0; x < output.width(); x++) {
    //             data[x] = output(x, y);
    //         }
    //         MPI_Send(data, count, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    //         delete[] data;
    //     }
    // }

    MPI_Finalize();
    return 0;
}
