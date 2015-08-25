#include "Halide.h"
using namespace Halide;
#include "mpi_timing.h"

Var x("x"), y("y"), c("c"), k("k");
const int J = 8;
const int maxJ = 20;
const int levels = 8;
const int alpha = 1;
const int beta = 1;

// Downsample with a 1 3 3 1 filter
Func downsample(Func f) {
    Func downx, downy;
    downx(x, y, _) = (f(2*x-1, y, _) + 3.0f * (f(2*x, y, _) + f(2*x+1, y, _)) + f(2*x+2, y, _)) / 8.0f;
    downy(x, y, _) = (downx(x, 2*y-1, _) + 3.0f * (downx(x, 2*y, _) + downx(x, 2*y+1, _)) + downx(x, 2*y+2, _)) / 8.0f;
    return downy;
}

// Upsample using bilinear interpolation
Func upsample(Func f) {
    Func upx, upy;
    upx(x, y, _) = 0.25f * f((x/2) - 1 + 2*(x % 2), y, _) + 0.75f * f(x/2, y, _);
    upy(x, y, _) = 0.25f * upx(x, (y/2) - 1 + 2*(y % 2), _) + 0.75f * upx(x, y/2, _);
    return upy;
}

void compute_correct(Image<uint16_t> &input, Image<uint16_t> &out) {
    // Make the remapping function as a lookup table.
    Func remap;
    Expr fx = cast<float>(x) / 256.0f;
    remap(x) = alpha*fx*exp(-fx*fx/2.0f);

    // Set a boundary condition
    Func clamped = BoundaryConditions::repeat_edge(input);

    // Convert to floating point
    Func floating;
    floating(x, y, c) = clamped(x, y, c) / 65535.0f;

    // Get the luminance channel
    Func gray;
    gray(x, y) = 0.299f * floating(x, y, 0) + 0.587f * floating(x, y, 1) + 0.114f * floating(x, y, 2);

    // Make the processed Gaussian pyramid.
    Func gPyramid[maxJ];
    // Do a lookup into a lut with 256 entires per intensity level
    Expr level = k * (1.0f / (levels - 1));
    Expr idx = gray(x, y)*cast<float>(levels-1)*256.0f;
    idx = clamp(cast<int>(idx), 0, (levels-1)*256);
    gPyramid[0](x, y, k) = beta*(gray(x, y) - level) + level + remap(idx - 256*k);
    for (int j = 1; j < J; j++) {
        gPyramid[j](x, y, k) = downsample(gPyramid[j-1])(x, y, k);
    }

    // Get its laplacian pyramid
    Func lPyramid[maxJ];
    lPyramid[J-1](x, y, k) = gPyramid[J-1](x, y, k);
    for (int j = J-2; j >= 0; j--) {
        lPyramid[j](x, y, k) = gPyramid[j](x, y, k) - upsample(gPyramid[j+1])(x, y, k);
    }

    // Make the Gaussian pyramid of the input
    Func inGPyramid[maxJ];
    inGPyramid[0](x, y) = gray(x, y);
    for (int j = 1; j < J; j++) {
        inGPyramid[j](x, y) = downsample(inGPyramid[j-1])(x, y);
    }

    // Make the laplacian pyramid of the output
    Func outLPyramid[maxJ];
    for (int j = 0; j < J; j++) {
        // Split input pyramid value into integer and floating parts
        Expr level = inGPyramid[j](x, y) * cast<float>(levels-1);
        Expr li = clamp(cast<int>(level), 0, levels-2);
        Expr lf = level - cast<float>(li);
        // Linearly interpolate between the nearest processed pyramid levels
        outLPyramid[j](x, y) = (1.0f - lf) * lPyramid[j](x, y, li) + lf * lPyramid[j](x, y, li+1);
    }

    // Make the Gaussian pyramid of the output
    Func outGPyramid[maxJ];
    outGPyramid[J-1](x, y) = outLPyramid[J-1](x, y);
    for (int j = J-2; j >= 0; j--) {
        outGPyramid[j](x, y) = upsample(outGPyramid[j+1])(x, y) + outLPyramid[j](x, y);
    }

    // Reintroduce color (Connelly: use eps to avoid scaling up noise w/ apollo3.png input)
    Func color;
    float eps = 0.01f;
    color(x, y, c) = outGPyramid[0](x, y) * (floating(x, y, c)+eps) / (gray(x, y)+eps);

    Func local_laplacian("local_laplacian");
    // Convert back to 16-bit
    local_laplacian(x, y, c) = cast<uint16_t>(clamp(color(x, y, c), 0.0f, 1.0f) * 65535.0f);

    /* THE SCHEDULE */
    remap.compute_root();

    // cpu schedule
    Var yi;
    local_laplacian.parallel(y, 32).vectorize(x, 8);
    gray.compute_root().parallel(y, 32).vectorize(x, 8);
    for (int j = 0; j < 4; j++) {
        if (j > 0) {
            inGPyramid[j]
                .compute_root().parallel(y, 32).vectorize(x, 8);
            gPyramid[j]
                .compute_root().reorder_storage(x, k, y)
                .reorder(k, y).parallel(y, 8).vectorize(x, 8);
        }
        outGPyramid[j].compute_root().parallel(y, 32).vectorize(x, 8);
    }
    for (int j = 4; j < J; j++) {
        inGPyramid[j].compute_root();
        gPyramid[j].compute_root().parallel(k);
        outGPyramid[j].compute_root();
    }

    local_laplacian.realize(out);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, numprocs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    const int w = std::stoi(argv[1]), h = std::stoi(argv[2]), d = 4;

    // For correctness testing:
    Image<uint16_t> global_input(w, h, d), global_output(w, h, d);
    DistributedImage<uint16_t> input(w, h, d), output(w, h, d);

    input.set_domain(x, y, c);
    input.placement().distribute(y);
    input.allocate();
    output.set_domain(x, y, c);
    output.placement().distribute(y);
    output.allocate();

    for (int c = 0; c < d; c++) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                uint16_t v = rand() & 0xfff;
                if (input.mine(x, y, c)) {
                    int lx = input.local(0, x), ly = input.local(1, y), lc = input.local(2, c);
                    input(lx, ly, lc) = v;
                }
                global_input(x, y, c) = v;
            }
        }
    }

    // Make the remapping function as a lookup table.
    Func remap;
    Expr fx = cast<float>(x) / 256.0f;
    remap(x) = alpha*fx*exp(-fx*fx/2.0f);

    // Set a boundary condition
    Func clamped;
    clamped(x, y, c) = input(clamp(x, 0, input.global_width() - 1),
                             clamp(y, 0, input.global_height() - 1),
                             clamp(c, 0, input.global_channels() - 1));

    // Convert to floating point
    Func floating;
    floating(x, y, c) = clamped(x, y, c) / 65535.0f;

    // Get the luminance channel
    Func gray;
    gray(x, y) = 0.299f * floating(x, y, 0) + 0.587f * floating(x, y, 1) + 0.114f * floating(x, y, 2);

    // Make the processed Gaussian pyramid.
    Func gPyramid[maxJ];
    // Do a lookup into a lut with 256 entires per intensity level
    Expr level = k * (1.0f / (levels - 1));
    Expr idx = gray(x, y)*cast<float>(levels-1)*256.0f;
    idx = clamp(cast<int>(idx), 0, (levels-1)*256);
    gPyramid[0](x, y, k) = beta*(gray(x, y) - level) + level + remap(idx - 256*k);
    for (int j = 1; j < J; j++) {
        gPyramid[j](x, y, k) = downsample(gPyramid[j-1])(x, y, k);
    }

    // Get its laplacian pyramid
    Func lPyramid[maxJ];
    lPyramid[J-1](x, y, k) = gPyramid[J-1](x, y, k);
    for (int j = J-2; j >= 0; j--) {
        lPyramid[j](x, y, k) = gPyramid[j](x, y, k) - upsample(gPyramid[j+1])(x, y, k);
    }

    // Make the Gaussian pyramid of the input
    Func inGPyramid[maxJ];
    inGPyramid[0](x, y) = gray(x, y);
    for (int j = 1; j < J; j++) {
        inGPyramid[j](x, y) = downsample(inGPyramid[j-1])(x, y);
    }

    // Make the laplacian pyramid of the output
    Func outLPyramid[maxJ];
    for (int j = 0; j < J; j++) {
        // Split input pyramid value into integer and floating parts
        Expr level = inGPyramid[j](x, y) * cast<float>(levels-1);
        Expr li = clamp(cast<int>(level), 0, levels-2);
        Expr lf = level - cast<float>(li);
        // Linearly interpolate between the nearest processed pyramid levels
        outLPyramid[j](x, y) = (1.0f - lf) * lPyramid[j](x, y, li) + lf * lPyramid[j](x, y, li+1);
    }

    // Make the Gaussian pyramid of the output
    Func outGPyramid[maxJ];
    outGPyramid[J-1](x, y) = outLPyramid[J-1](x, y);
    for (int j = J-2; j >= 0; j--) {
        outGPyramid[j](x, y) = upsample(outGPyramid[j+1])(x, y) + outLPyramid[j](x, y);
    }

    // Reintroduce color (Connelly: use eps to avoid scaling up noise w/ apollo3.png input)
    Func color;
    float eps = 0.01f;
    color(x, y, c) = outGPyramid[0](x, y) * (floating(x, y, c)+eps) / (gray(x, y)+eps);

    Func local_laplacian("local_laplacian");
    // Convert back to 16-bit
    local_laplacian(x, y, c) = cast<uint16_t>(clamp(color(x, y, c), 0.0f, 1.0f) * 65535.0f);

    /* THE SCHEDULE */
    Var yi;
    local_laplacian.parallel(y).vectorize(x, 8).distribute(y);
    remap.compute_root();
    gray.compute_root().parallel(y, 32).vectorize(x, 8).distribute(y);
    for (int j = 0; j < 4; j++) {
        if (j > 0) {
            inGPyramid[j]
                .compute_root().parallel(y, 32).vectorize(x, 8).distribute(y);
            gPyramid[j]
                .compute_root()
                .reorder(k, y).parallel(y, 8).vectorize(x, 8).distribute(y);
        }
        outGPyramid[j].compute_root().parallel(y, 32).vectorize(x, 8).distribute(y);
    }
    for (int j = 4; j < J; j++) {
        inGPyramid[j].compute_root().distribute(y);
        gPyramid[j].compute_root().parallel(k).distribute(y);
        outGPyramid[j].compute_root().distribute(y);
    }

    compute_correct(global_input, global_output);
    local_laplacian.realize(output.get_buffer());

    const int niters = 10;
    MPITiming timing(MPI_COMM_WORLD);
    timing.barrier();
    timeval t1, t2;
    for (int i = 0; i < niters; i++) {
        gettimeofday(&t1, NULL);
        local_laplacian.realize(output.get_buffer());
        gettimeofday(&t2, NULL);
        float t = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0f;
        timing.record(t);
    }
    timing.reduce(MPITiming::Median);

    for (int c = 0; c < output.channels(); c++) {
        for (int y = 0; y < output.height(); y++) {
            for (int x = 0; x < output.width(); x++) {
                int gx = output.global(0, x), gy = output.global(1, y), gc = output.global(2, c);
                if (output(x, y, c) != global_output(gx, gy, gc)) {
                    printf("[rank %d] output(%d,%d) = %u instead of %u\n", rank, x, y, output(x, y, c), global_output(gx, gy, gc));
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    MPI_Finalize();
                    return -1;
                }
            }
        }
    }
    timing.gather(MPITiming::Max);
    timing.report();
    if (rank == 0) {
        printf("Bilateral grid test succeeded!\n");
    }
    MPI_Finalize();

    return 0;
}
