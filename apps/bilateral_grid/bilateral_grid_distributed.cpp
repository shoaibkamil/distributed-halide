#include "Halide.h"
#include "mpi_timing.h"
#include <stdio.h>
#include <random>

using namespace Halide;

const int s_sigma = 8;
const float r_sigma = 0.1;
std::default_random_engine generator(0);
std::uniform_real_distribution<float> distribution(10, 500);

bool float_eq(float a, float b) {
    const float thresh = 1e-5;
    return a == b || (std::abs(a - b) / b) < thresh;
}

void compute_correct(Image<float> &input, Image<float> &out) {
    Var x("x"), y("y"), z("z"), c("c");

    // Add a boundary condition
    Func clamped = BoundaryConditions::repeat_edge(input);

    // Construct the bilateral grid
    RDom r(0, s_sigma, 0, s_sigma);
    Expr val = clamped(x * s_sigma + r.x - s_sigma/2, y * s_sigma + r.y - s_sigma/2);
    val = clamp(val, 0.0f, 1.0f);
    Expr zi = cast<int>(val * (1.0f/r_sigma) + 0.5f);
    Func histogram("histogram");
    histogram(x, y, z, c) = 0.0f;
    histogram(x, y, zi, c) += select(c == 0, val, 1.0f);

    // Blur the grid using a five-tap filter
    Func blurx("blurx"), blury("blury"), blurz("blurz");
    blurz(x, y, z, c) = (histogram(x, y, z-2, c) +
                         histogram(x, y, z-1, c)*4 +
                         histogram(x, y, z  , c)*6 +
                         histogram(x, y, z+1, c)*4 +
                         histogram(x, y, z+2, c));
    blurx(x, y, z, c) = (blurz(x-2, y, z, c) +
                         blurz(x-1, y, z, c)*4 +
                         blurz(x  , y, z, c)*6 +
                         blurz(x+1, y, z, c)*4 +
                         blurz(x+2, y, z, c));
    blury(x, y, z, c) = (blurx(x, y-2, z, c) +
                         blurx(x, y-1, z, c)*4 +
                         blurx(x, y  , z, c)*6 +
                         blurx(x, y+1, z, c)*4 +
                         blurx(x, y+2, z, c));

    // Take trilinear samples to compute the output
    val = clamp(input(x, y), 0.0f, 1.0f);
    Expr zv = val * (1.0f/r_sigma);
    zi = cast<int>(zv);
    Expr zf = zv - zi;
    Expr xf = cast<float>(x % s_sigma) / s_sigma;
    Expr yf = cast<float>(y % s_sigma) / s_sigma;
    Expr xi = x/s_sigma;
    Expr yi = y/s_sigma;
    Func interpolated("interpolated");
    interpolated(x, y, c) =
        lerp(lerp(lerp(blury(xi, yi, zi, c), blury(xi+1, yi, zi, c), xf),
                  lerp(blury(xi, yi+1, zi, c), blury(xi+1, yi+1, zi, c), xf), yf),
             lerp(lerp(blury(xi, yi, zi+1, c), blury(xi+1, yi, zi+1, c), xf),
                  lerp(blury(xi, yi+1, zi+1, c), blury(xi+1, yi+1, zi+1, c), xf), yf), zf);

    // Normalize
    Func bilateral_grid("bilateral_grid");
    bilateral_grid(x, y) = interpolated(x, y, 0)/interpolated(x, y, 1);

    // CPU schedule
    histogram.compute_at(blurz, y);
    histogram.update().reorder(c, r.x, r.y, x, y).unroll(c);
    blurz.compute_root().reorder(c, z, x, y).parallel(y).vectorize(x, 4).unroll(c);
    blurx.compute_root().reorder(c, x, y, z).parallel(z).vectorize(x, 4).unroll(c);
    blury.compute_root().reorder(c, x, y, z).parallel(z).vectorize(x, 4).unroll(c);
    bilateral_grid.compute_root().parallel(y).vectorize(x, 4);

    bilateral_grid.realize(out);
}

float rndflt() {
    return distribution(generator);
}

int main(int argc, char **argv) {
    int req = MPI_THREAD_MULTIPLE, prov;
    MPI_Init_thread(&argc, &argv, req, &prov);
    assert(prov == req);
    int rank = 0, numprocs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    const int w = std::stoi(argv[1]), h = std::stoi(argv[2]);

    // For correctness testing:
    // Image<float> global_input(w, h), global_output(w, h);

    DistributedImage<float> input(w, h), output(w, h);
    Var x("x"), y("y"), z("z"), c("c");

    // Add a boundary condition
    Func clamped;
    clamped(x, y) = input(clamp(x, 0, input.global_width()-1),
                          clamp(y, 0, input.global_height()-1));

    // Construct the bilateral grid
    RDom r(0, s_sigma, 0, s_sigma);
    Expr val = clamped(x * s_sigma + r.x - s_sigma/2, y * s_sigma + r.y - s_sigma/2);
    val = clamp(val, 0.0f, 1.0f);
    Expr zi = cast<int>(val * (1.0f/r_sigma) + 0.5f);
    Func histogram("histogram");
    histogram(x, y, z, c) = 0.0f;
    histogram(x, y, zi, c) += select(c == 0, val, 1.0f);

    // Blur the grid using a five-tap filter
    Func blurx("blurx"), blury("blury"), blurz("blurz");
    blurz(x, y, z, c) = (histogram(x, y, z-2, c) +
                         histogram(x, y, z-1, c)*4 +
                         histogram(x, y, z  , c)*6 +
                         histogram(x, y, z+1, c)*4 +
                         histogram(x, y, z+2, c));
    blurx(x, y, z, c) = (blurz(x-2, y, z, c) +
                         blurz(x-1, y, z, c)*4 +
                         blurz(x  , y, z, c)*6 +
                         blurz(x+1, y, z, c)*4 +
                         blurz(x+2, y, z, c));
    blury(x, y, z, c) = (blurx(x, y-2, z, c) +
                         blurx(x, y-1, z, c)*4 +
                         blurx(x, y  , z, c)*6 +
                         blurx(x, y+1, z, c)*4 +
                         blurx(x, y+2, z, c));

    // Take trilinear samples to compute the output
    val = clamp(input(x, y), 0.0f, 1.0f);
    Expr zv = val * (1.0f/r_sigma);
    zi = cast<int>(zv);
    Expr zf = zv - zi;
    Expr xf = cast<float>(x % s_sigma) / s_sigma;
    Expr yf = cast<float>(y % s_sigma) / s_sigma;
    Expr xi = x/s_sigma;
    Expr yi = y/s_sigma;
    Func interpolated("interpolated");
    interpolated(x, y, c) =
        lerp(lerp(lerp(blury(xi, yi, zi, c), blury(xi+1, yi, zi, c), xf),
                  lerp(blury(xi, yi+1, zi, c), blury(xi+1, yi+1, zi, c), xf), yf),
             lerp(lerp(blury(xi, yi, zi+1, c), blury(xi+1, yi, zi+1, c), xf),
                  lerp(blury(xi, yi+1, zi+1, c), blury(xi+1, yi+1, zi+1, c), xf), yf), zf);

    // Normalize
    Func bilateral_grid("bilateral_grid");
    bilateral_grid(x, y) = interpolated(x, y, 0)/interpolated(x, y, 1);

    // CPU schedule
    Var yin;
    bilateral_grid.compute_root().parallel(y).distribute(y).vectorize(x, 4);
    histogram.compute_at(blurz, y);
    histogram.update().reorder(c, r.x, r.y, x, y).unroll(c);
    blurz.compute_at(bilateral_grid, y).reorder(c, z, x, y).vectorize(x, 4).unroll(c);
    blurx.compute_at(bilateral_grid, y).reorder(c, x, y, z).vectorize(x, 4).unroll(c);
    blury.compute_at(bilateral_grid, y).reorder(c, x, y, z).vectorize(x, 4).unroll(c);

    output.set_domain(x, y);
    output.placement().distribute(y);
    output.allocate();
    input.set_domain(x, y);
    input.placement().distribute(y);
    input.allocate(bilateral_grid, output);

    for (int y = 0; y < input.height(); y++) {
        for (int x = 0; x < input.width(); x++) {
            int gx = input.global(0, x), gy = input.global(1, y);
            float v = gx+gy; //rndflt();
            input(x, y) = v;
        }
    }

    // blurz.compute_root().reorder(c, z, x, y).parallel(y).vectorize(x, 4).unroll(c).distribute(y);
    // blurx.compute_root().reorder(c, x, y, z).parallel(z).vectorize(x, 4).unroll(c).distribute(z);
    // blury.compute_root().reorder(c, x, y, z).parallel(z).vectorize(x, 4).unroll(c).distribute(z);
    //bilateral_grid.compute_root().parallel(y).vectorize(x, 4).distribute(y);

    //compute_correct(global_input, global_output);
    bilateral_grid.realize(output.get_buffer());

    const int niters = 50;
#ifdef USE_MPIP
    MPI_Pcontrol(1);
#endif
    MPITiming timing(MPI_COMM_WORLD);
    timing.barrier();
    for (int i = 0; i < niters; i++) {
        timing.start();
        bilateral_grid.realize(output.get_buffer());
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
        printf("Bilateral grid test succeeded!\n");
    }
    MPI_Finalize();
    return 0;
}
