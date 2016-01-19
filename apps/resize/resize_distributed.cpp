#include "Halide.h"
#include "mpi_timing.h"

using namespace Halide;

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

enum InterpolationType {
    BOX, LINEAR, CUBIC, LANCZOS
};

Expr kernel_box(Expr x) {
    Expr xx = abs(x);
    return select(xx <= 0.5f, 1.0f, 0.0f);
}

Expr kernel_linear(Expr x) {
    Expr xx = abs(x);
    return select(xx < 1.0f, 1.0f - xx, 0.0f);
}

Expr kernel_cubic(Expr x) {
    Expr xx = abs(x);
    Expr xx2 = xx * xx;
    Expr xx3 = xx2 * xx;
    float a = -0.5f;

    return select(xx < 1.0f, (a + 2.0f) * xx3 - (a + 3.0f) * xx2 + 1,
                  select (xx < 2.0f, a * xx3 - 5 * a * xx2 + 8 * a * xx - 4.0f * a,
                          0.0f));
}

Expr sinc(Expr x) {
    return sin(float(M_PI) * x) / x;
}

Expr kernel_lanczos(Expr x) {
    Expr value = sinc(x) * sinc(x/3);
    value = select(x == 0.0f, 1.0f, value); // Take care of singularity at zero
    value = select(x > 3 || x < -3, 0.0f, value); // Clamp to zero out of bounds
    return value;
}

struct KernelInfo {
    const char *name;
    float size;
    Expr (*kernel)(Expr);
};

static KernelInfo kernelInfo[] = {
    { "box", 0.5f, kernel_box },
    { "linear", 1.0f, kernel_linear },
    { "cubic", 2.0f, kernel_cubic },
    { "lanczos", 3.0f, kernel_lanczos }
};


InterpolationType interpolationType = CUBIC;
const float scaleFactor = 1.3f;
const int schedule = 3;

Func build(bool distributed) {
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


    // For downscaling, widen the interpolation kernel to perform lowpass
    // filtering.
    float kernelScaling = std::min(scaleFactor, 1.0f);
    float kernelSize = kernelInfo[interpolationType].size / kernelScaling;

    // source[xy] are the (non-integer) coordinates inside the source image
    Expr sourcex = (x + 0.5f) / scaleFactor;
    Expr sourcey = (y + 0.5f) / scaleFactor;

    // Initialize interpolation kernels. Since we allow an arbitrary
    // scaling factor, the filter coefficients are different for each x
    // and y coordinate.
    Func kernelx("kernelx"), kernely("kernely");
    Expr beginx = cast<int>(sourcex - kernelSize + 0.5f);
    Expr beginy = cast<int>(sourcey - kernelSize + 0.5f);
    RDom domx(0, static_cast<int>(2.0f * kernelSize) + 1, "domx");
    RDom domy(0, static_cast<int>(2.0f * kernelSize) + 1, "domy");
    {
        const KernelInfo &info = kernelInfo[interpolationType];
        Func kx, ky;
        kx(x, k) = info.kernel((k + beginx - sourcex) * kernelScaling);
        ky(y, k) = info.kernel((k + beginy - sourcey) * kernelScaling);
        kernelx(x, k) = kx(x, k) / sum(kx(x, domx));
        kernely(y, k) = ky(y, k) / sum(ky(y, domy));
    }

    // Perform separable resizing
    Func resized_x("resized_x");
    Func resized_y("resized_y");
    resized_x(x, y, c) = sum(kernelx(x, domx) * cast<float>(clamped(domx + beginx, y, c)));
    resized_y(x, y, c) = sum(kernely(y, domy) * resized_x(x, domy + beginy, c));

    Func final("final");
    final(x, y, c) = clamp(resized_y(x, y, c), 0.0f, 1.0f);

    // Scheduling
    bool parallelize = (schedule >= 2);
    bool vectorize = (schedule == 1 || schedule == 3);

    kernelx.compute_root();
    kernely.compute_at(final, y);

    if (vectorize) {
        resized_x.vectorize(x, 4);
        final.vectorize(x, 4);
    }

    if (parallelize) {
        final.split(y, yo, y, 8).parallel(yo);
        resized_x.store_at(final, yo).compute_at(final, y);
    } else {
        resized_x.store_at(final, c).compute_at(final, y);
    }

    if (distributed) {
        final.distribute(yo);
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

    const int w = std::stoi(argv[1]), h = std::stoi(argv[2]), d = 3;
    const int ow = (int)(w * scaleFactor), oh = (int)(h * scaleFactor), od = 3;

    input = DistributedImage<float>(w, h, d);
    output = DistributedImage<float>(ow, oh, od);
    // global_input = Image<float>(w, h, d);
    // global_output = Image<float>(ow, oh, od);

    //Func resize_correct = build(false);
    Func resize_distributed = build(true);

    output.set_domain(x, y, c);
    output.placement().split(y, yo, y, 8).distribute(yo);
    output.allocate();
    input.set_domain(x, y, c);
    input.placement().split(y, yo, y, 8).distribute(yo);
    input.allocate(resize_distributed, output);

    for (int c = 0; c < input.channels(); c++) {
        for (int y = 0; y < input.height(); y++) {
            for (int x = 0; x < input.width(); x++) {
                int gx = input.global(0, x), gy = input.global(1, y), gc = input.global(2, c);
                float v = gx + gy + gc;
                input(x, y, c) = v;
            }
        }
    }

    resize_distributed.compile_jit();

    // resize_correct.realize(global_output);

    const int niters = 50;
#ifdef USE_MPIP
    MPI_Pcontrol(1);
#endif
    MPITiming timing(MPI_COMM_WORLD);
    timing.barrier();
    for (int i = 0; i < niters; i++) {
        timing.start();
        resize_distributed.realize(output.get_buffer());
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
        printf("Resize test succeeded!\n");
    }

    MPI_Finalize();
    return 0;
}
