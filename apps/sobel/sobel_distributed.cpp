#include "Halide.h"
using namespace Halide;

#include "image_io.h"
#include "mpi_timing.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, numprocs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    //const int w = std::stoi(argv[1]), h = std::stoi(argv[2]);
    Var x("x"), y("y");

    Image<float> global_input = load<float>(argv[1]);
    Image<float> global_output(global_input.width(), global_input.height());

    const int w = global_input.width(), h = global_input.height();

    DistributedImage<float> input(w, h);
    DistributedImage<float> output(w, h);
    Func clamped;
    clamped(x, y) = input(clamp(x, 0, input.global_width() - 1),
                          clamp(y, 0, input.global_height() - 1));

    Func sobelx, sobely;
    sobelx(x, y) = -1*clamped(x-1, y-1) + clamped(x+1, y-1) +
        -2*clamped(x-1, y) + 2*clamped(x+1, y) +
        -1*clamped(x-1, y+1) + clamped(x+1, y+1);

    sobely(x, y) = -1*clamped(x-1, y-1) + -2*clamped(x, y-1) + -1*clamped(x+1,y-1) +
        clamped(x-1, y+1) + 2*clamped(x, y+1) + clamped(x+1, y+1);

    Func sobel;
    sobel(x, y) = sqrt(sobelx(x, y) * sobelx(x, y) + sobely(x, y) * sobely(x, y));

    sobel.distribute(y);

    output.set_domain(x, y);
    output.placement().distribute(y);
    output.allocate();

    input.set_domain(x, y);
    input.placement().distribute(y);
    input.allocate(sobel, output);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (input.mine(x, y)) {
                int lx = input.local(0, x), ly = input.local(1, y);
                input(lx, ly) = global_input(x, y);
            }
        }
    }

    sobel.realize(output.get_buffer());

    if (rank == 0) {
        if (numprocs > 1) {
            for (int r = 1; r < numprocs; r++) {
                MPI_Status status;
                int numrows, rowwidth, startx, starty;
                MPI_Recv(&numrows, 1, MPI_INT, r, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(&rowwidth, 1, MPI_INT, r, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(&startx, 1, MPI_INT, r, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(&starty, 1, MPI_INT, r, 0, MPI_COMM_WORLD, &status);

                int maxcount = output.global_width();
                float *data = new float[maxcount];
                for (int i = 0; i < numrows; i++) {
                    MPI_Recv(data, maxcount, MPI_FLOAT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    int gx = startx, gy = starty + i;
                    for (int i = 0; i < rowwidth; gx++, i++) {
                        global_output(gx, gy) = data[i];
                    }
                }
                delete[] data;
            }
        }

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (output.mine(x, y)) {
                    int lx = output.local(0, x), ly = output.local(1, y);
                    global_output(x, y) = output(lx, ly);
                }
            }
        }
        save(global_output, argv[2]);
    } else {
        int numrows = output.height(), rowwidth = output.width(),
            startx = output.global(0, 0), starty = output.global(1, 0);
        MPI_Send(&numrows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&rowwidth, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&startx, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&starty, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        for (int y = 0; y < output.height(); y++) {
            int count = output.width();
            float *data = new float[count];
            for (int x = 0; x < output.width(); x++) {
                data[x] = output(x, y);
            }
            MPI_Send(data, count, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
            delete[] data;
        }
    }
    MPI_Finalize();
    return 0;
}
