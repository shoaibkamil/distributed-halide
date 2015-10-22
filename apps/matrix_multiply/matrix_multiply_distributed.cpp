#include "Halide.h"
#include "mpi_timing.h"
#include <stdio.h>
#include <assert.h>

using namespace Halide;

std::default_random_engine generator(0);
std::uniform_real_distribution<float> distribution(10, 500);
Var x("x"), y("y"), xi("xi"), xo("xo"), yo("yo"), yi("yo"), yii("yii"), xii("xii");
const int block_size = 32;
int matrix_size;

float rndflt() {
    return distribution(generator);
}

bool float_eq(float a, float b) {
    const float thresh = 1e-5;
    return a == b || (std::abs(a - b) / b) < thresh;
}

Func build(Func A, Func B, bool distributed) {
    Func matrix_mul("matrix_mul");

    RDom k(0, matrix_size);
    RVar ki;

    matrix_mul(x, y) = 0.0f;
    matrix_mul(x, y) += A(k, y) * B(x, k);

    matrix_mul.split(x, x, xi, block_size).split(xi, xi, xii, 8)
        .split(y, y, yi, block_size).split(yi, yi, yii, 4)
        .reorder(xii, yii, xi, yi, x, y)
        .parallel(y).vectorize(xii).unroll(xi).unroll(yii);

    matrix_mul.update()
        .split(x, x, xi, block_size).split(xi, xi, xii, 8)
        .split(y, y, yi, block_size).split(yi, yi, yii, 4)
        .split(k, k, ki, block_size)
        .reorder(xii, yii, xi, ki, yi, k, x, y)
        .parallel(y).vectorize(xii).unroll(xi).unroll(yii);

    // matrix_mul
    //     .bound(x, 0, matrix_size)
    //     .bound(y, 0, matrix_size);

    if (distributed) {
        matrix_mul.distribute(y);
        matrix_mul.update().distribute(y);
    }

    return matrix_mul;
}

int main(int argc, char **argv) {
    int req = MPI_THREAD_MULTIPLE, prov;
    MPI_Init_thread(&argc, &argv, req, &prov);
    assert(prov == req);
    int rank = 0, numprocs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    const int w = std::stoi(argv[1]), h = std::stoi(argv[2]);
    assert(w == h && "Non-square matrices unimplemented");

    // Compute C = A*B
    Image<float> global_A(w, h), global_B(w, h), global_C(w, h);
    DistributedImage<float> A(w, h), B(w, h), C(w, h);
    A.set_domain(x, y);
    A.placement();
    A.allocate();
    B.set_domain(x, y);
    B.placement();
    B.allocate();
    C.set_domain(x, y);
    C.placement().split(x, x, xi, block_size).split(xi, xi, xii, 8)
        .split(y, y, yi, block_size).split(yi, yi, yii, 4)
        .distribute(y);
    C.allocate();

    matrix_size = w;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float v = rndflt();
            if (A.mine(x, y)) {
                int lx = A.local(0, x), ly = A.local(1, y);
                A(lx, ly) = v;
            }
            global_A(x, y) = v;

            v = rndflt();
            if (B.mine(x, y)) {
                int lx = B.local(0, x), ly = B.local(1, y);
                B(lx, ly) = v;
            }
            global_B(x, y) = v;
        }
    }

    Func accessor_A, global_accessor_A, accessor_B, global_accessor_B;
    accessor_A(x, y) = A(x, y);
    global_accessor_A(x, y) = global_A(x, y);
    accessor_B(x, y) = B(x, y);
    global_accessor_B(x, y) = global_B(x, y);

    Func mm_correct = build(global_accessor_A, global_accessor_B, false);
    Func mm_distributed = build(accessor_A, accessor_B, true);

    mm_correct.realize(global_C);
    mm_distributed.realize(C.get_buffer());

    const int niters = 10;
    MPITiming timing(MPI_COMM_WORLD);
    timing.barrier();
    timeval t1, t2;
    for (int i = 0; i < niters; i++) {
        gettimeofday(&t1, NULL);
        mm_distributed.realize(C.get_buffer());
        gettimeofday(&t2, NULL);
        float t = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0f;
        timing.record(t);
    }
    timing.reduce(MPITiming::Median);

    for (int y = 0; y < C.height(); y++) {
        for (int x = 0; x < C.width(); x++) {
            int gx = C.global(0, x), gy = C.global(1, y);
            if (!float_eq(C(x, y), global_C(gx, gy))) {
                printf("[rank %d] C(%d,%d) = %f instead of %f\n", rank, x, y, C(x, y), global_C(gx, gy));
                MPI_Abort(MPI_COMM_WORLD, 1);
                MPI_Finalize();
                return -1;
            }
        }
    }

    timing.gather(MPITiming::Max);
    timing.report();

    if (rank == 0) {
        printf("Matrix multiply test succeeded!\n");
    }
    MPI_Finalize();

    return 0;
}
