#include "Halide.h"
#include "mpi_timing.h"
#include <stdint.h>

using namespace Halide;

const int bit_width = 8, schedule = 1;

Var x, y, tx("tx"), ty("ty"), c("c"), xi, yi;

// Average two positive values rounding up
Expr avg(Expr a, Expr b) {
    Type wider = a.type().with_bits(a.type().bits() * 2);
    return cast(a.type(), (cast(wider, a) + b + 1)/2);
}

Func hot_pixel_suppression(Func input) {
    Expr a = max(max(input(x-2, y), input(x+2, y)),
                 max(input(x, y-2), input(x, y+2)));
    Expr b = min(min(input(x-2, y), input(x+2, y)),
                 min(input(x, y-2), input(x, y+2)));

    Func denoised;
    denoised(x, y) = clamp(input(x, y), b, a);

    return denoised;
}

Func interleave_x(Func a, Func b) {
    Func out;
    out(x, y) = select((x%2)==0, a(x/2, y), b(x/2, y));
    return out;
}

Func interleave_y(Func a, Func b) {
    Func out;
    out(x, y) = select((y%2)==0, a(x, y/2), b(x, y/2));
    return out;
}

Func deinterleave(Func raw) {
    // Deinterleave the color channels
    Func deinterleaved;

    deinterleaved(x, y, c) = select(c == 0, raw(2*x, 2*y),
                                    select(c == 1, raw(2*x+1, 2*y),
                                           select(c == 2, raw(2*x, 2*y+1),
                                                  raw(2*x+1, 2*y+1))));
    return deinterleaved;
}

Func demosaic(Func deinterleaved, Func processed) {
    // These are the values we already know from the input
    // x_y = the value of channel x at a site in the input of channel y
    // gb refers to green sites in the blue rows
    // gr refers to green sites in the red rows

    // Give more convenient names to the four channels we know
    Func r_r, g_gr, g_gb, b_b;
    g_gr(x, y) = deinterleaved(x, y, 0);
    r_r(x, y)  = deinterleaved(x, y, 1);
    b_b(x, y)  = deinterleaved(x, y, 2);
    g_gb(x, y) = deinterleaved(x, y, 3);

    // These are the ones we need to interpolate
    Func b_r, g_r, b_gr, r_gr, b_gb, r_gb, r_b, g_b;

    // First calculate green at the red and blue sites

    // Try interpolating vertically and horizontally. Also compute
    // differences vertically and horizontally. Use interpolation in
    // whichever direction had the smallest difference.
    Expr gv_r  = avg(g_gb(x, y-1), g_gb(x, y));
    Expr gvd_r = absd(g_gb(x, y-1), g_gb(x, y));
    Expr gh_r  = avg(g_gr(x+1, y), g_gr(x, y));
    Expr ghd_r = absd(g_gr(x+1, y), g_gr(x, y));

    g_r(x, y)  = select(ghd_r < gvd_r, gh_r, gv_r);

    Expr gv_b  = avg(g_gr(x, y+1), g_gr(x, y));
    Expr gvd_b = absd(g_gr(x, y+1), g_gr(x, y));
    Expr gh_b  = avg(g_gb(x-1, y), g_gb(x, y));
    Expr ghd_b = absd(g_gb(x-1, y), g_gb(x, y));

    g_b(x, y)  = select(ghd_b < gvd_b, gh_b, gv_b);

    // Next interpolate red at gr by first interpolating, then
    // correcting using the error green would have had if we had
    // interpolated it in the same way (i.e. add the second derivative
    // of the green channel at the same place).
    Expr correction;
    correction = g_gr(x, y) - avg(g_r(x, y), g_r(x-1, y));
    r_gr(x, y) = correction + avg(r_r(x-1, y), r_r(x, y));

    // Do the same for other reds and blues at green sites
    correction = g_gr(x, y) - avg(g_b(x, y), g_b(x, y-1));
    b_gr(x, y) = correction + avg(b_b(x, y), b_b(x, y-1));

    correction = g_gb(x, y) - avg(g_r(x, y), g_r(x, y+1));
    r_gb(x, y) = correction + avg(r_r(x, y), r_r(x, y+1));

    correction = g_gb(x, y) - avg(g_b(x, y), g_b(x+1, y));
    b_gb(x, y) = correction + avg(b_b(x, y), b_b(x+1, y));

    // Now interpolate diagonally to get red at blue and blue at
    // red. Hold onto your hats; this gets really fancy. We do the
    // same thing as for interpolating green where we try both
    // directions (in this case the positive and negative diagonals),
    // and use the one with the lowest absolute difference. But we
    // also use the same trick as interpolating red and blue at green
    // sites - we correct our interpolations using the second
    // derivative of green at the same sites.

    correction = g_b(x, y)  - avg(g_r(x, y), g_r(x-1, y+1));
    Expr rp_b  = correction + avg(r_r(x, y), r_r(x-1, y+1));
    Expr rpd_b = absd(r_r(x, y), r_r(x-1, y+1));

    correction = g_b(x, y)  - avg(g_r(x-1, y), g_r(x, y+1));
    Expr rn_b  = correction + avg(r_r(x-1, y), r_r(x, y+1));
    Expr rnd_b = absd(r_r(x-1, y), r_r(x, y+1));

    r_b(x, y)  = select(rpd_b < rnd_b, rp_b, rn_b);


    // Same thing for blue at red
    correction = g_r(x, y)  - avg(g_b(x, y), g_b(x+1, y-1));
    Expr bp_r  = correction + avg(b_b(x, y), b_b(x+1, y-1));
    Expr bpd_r = absd(b_b(x, y), b_b(x+1, y-1));

    correction = g_r(x, y)  - avg(g_b(x+1, y), g_b(x, y-1));
    Expr bn_r  = correction + avg(b_b(x+1, y), b_b(x, y-1));
    Expr bnd_r = absd(b_b(x+1, y), b_b(x, y-1));

    b_r(x, y)  =  select(bpd_r < bnd_r, bp_r, bn_r);

    // Interleave the resulting channels
    Func r = interleave_y(interleave_x(r_gr, r_r),
                          interleave_x(r_b, r_gb));
    Func g = interleave_y(interleave_x(g_gr, g_r),
                          interleave_x(g_b, g_gb));
    Func b = interleave_y(interleave_x(b_gr, b_r),
                          interleave_x(b_b, b_gb));

    Func output;
    output(x, y, c) = select(c == 0, r(x, y),
                             c == 1, g(x, y),
                                     b(x, y));


    /* THE SCHEDULE */
    if (schedule == 0) {
        // optimized for ARM
        // Compute these in chunks over tiles, vectorized by 8
        g_r.compute_at(processed, tx).vectorize(x, 8);
        g_b.compute_at(processed, tx).vectorize(x, 8);
        r_gr.compute_at(processed, tx).vectorize(x, 8);
        b_gr.compute_at(processed, tx).vectorize(x, 8);
        r_gb.compute_at(processed, tx).vectorize(x, 8);
        b_gb.compute_at(processed, tx).vectorize(x, 8);
        r_b.compute_at(processed, tx).vectorize(x, 8);
        b_r.compute_at(processed, tx).vectorize(x, 8);
        // These interleave in y, so unrolling them in y helps
        output.compute_at(processed, tx)
            .vectorize(x, 8)
            .unroll(y, 2)
            .reorder(c, x, y).bound(c, 0, 3).unroll(c);
    } else if (schedule == 1) {
        // optimized for X86
        // Don't vectorize, because sse is bad at 16-bit interleaving
        g_r.compute_at(processed, tx);
        g_b.compute_at(processed, tx);
        r_gr.compute_at(processed, tx);
        b_gr.compute_at(processed, tx);
        r_gb.compute_at(processed, tx);
        b_gb.compute_at(processed, tx);
        r_b.compute_at(processed, tx);
        b_r.compute_at(processed, tx);
        // These interleave in x and y, so unrolling them helps
        output.compute_at(processed, tx).unroll(x, 2).unroll(y, 2)
            .reorder(c, x, y).bound(c, 0, 3).unroll(c);

    } else {
        // Basic naive schedule
        g_r.compute_root();
        g_b.compute_root();
        r_gr.compute_root();
        b_gr.compute_root();
        r_gb.compute_root();
        b_gb.compute_root();
        r_b.compute_root();
        b_r.compute_root();
        output.compute_root();
    }
    return output;
}


Func color_correct(Func input, ImageParam matrix_3200, ImageParam matrix_7000, Param<float> kelvin) {
    // Get a color matrix by linearly interpolating between two
    // calibrated matrices using inverse kelvin.

    Func matrix;
    Expr alpha = (1.0f/kelvin - 1.0f/3200) / (1.0f/7000 - 1.0f/3200);
    Expr val =  (matrix_3200(x, y) * alpha + matrix_7000(x, y) * (1 - alpha));
    matrix(x, y) = cast<int32_t>(val * 256.0f); // Q8.8 fixed point
    matrix.compute_root();

    Func corrected;
    Expr ir = cast<int32_t>(input(x, y, 0));
    Expr ig = cast<int32_t>(input(x, y, 1));
    Expr ib = cast<int32_t>(input(x, y, 2));

    Expr r = matrix(3, 0) + matrix(0, 0) * ir + matrix(1, 0) * ig + matrix(2, 0) * ib;
    Expr g = matrix(3, 1) + matrix(0, 1) * ir + matrix(1, 1) * ig + matrix(2, 1) * ib;
    Expr b = matrix(3, 2) + matrix(0, 2) * ir + matrix(1, 2) * ig + matrix(2, 2) * ib;

    r = cast<int16_t>(r/256);
    g = cast<int16_t>(g/256);
    b = cast<int16_t>(b/256);
    corrected(x, y, c) = select(c == 0, r,
                                select(c == 1, g, b));

    return corrected;
}


Func apply_curve(Func input, Type result_type, Param<float> gamma, Param<float> contrast) {
    // copied from FCam
    Func curve("curve");

    Expr xf = clamp(cast<float>(x)/1024.0f, 0.0f, 1.0f);
    Expr g = pow(xf, 1.0f/gamma);
    Expr b = 2.0f - pow(2.0f, contrast/100.0f);
    Expr a = 2.0f - 2.0f*b;
    Expr z = select(g > 0.5f,
                    1.0f - (a*(1.0f-g)*(1.0f-g) + b*(1.0f-g)),
                    a*g*g + b*g);

    Expr val = cast(result_type, clamp(z*256.0f, 0.0f, 255.0f));
    curve(x) = val;
    curve.compute_root(); // It's a LUT, compute it once ahead of time.

    Func curved;
    curved(x, y, c) = curve(input(x, y, c));

    return curved;
}

Func process(Func raw, Type result_type,
             ImageParam matrix_3200, ImageParam matrix_7000, Param<float> color_temp,
             Param<float> gamma, Param<float> contrast, bool distributed) {
    Func processed;

    Func denoised = hot_pixel_suppression(raw);
    Func deinterleaved = deinterleave(denoised);
    Func demosaiced = demosaic(deinterleaved, processed);
    Func corrected = color_correct(demosaiced, matrix_3200, matrix_7000, color_temp);
    Func curved = apply_curve(corrected, result_type, gamma, contrast);

    processed(tx, ty, c) = curved(tx, ty, c);

    // Schedule
    processed.bound(c, 0, 3); // bound color loop 0-3, properly
    if (schedule == 0) {
        // Compute in chunks over tiles, vectorized by 8
        denoised.compute_at(processed, tx).vectorize(x, 8);
        deinterleaved.compute_at(processed, tx).vectorize(x, 8).reorder(c, x, y).unroll(c);
        corrected.compute_at(processed, tx).vectorize(x, 4).reorder(c, x, y).unroll(c);
        processed.tile(tx, ty, xi, yi, 32, 32).reorder(xi, yi, c, tx, ty);
        processed.parallel(ty);
    } else if (schedule == 1) {
        // Same as above, but don't vectorize (sse is bad at interleaved 16-bit ops)
        denoised.compute_at(processed, tx);
        deinterleaved.compute_at(processed, tx);
        corrected.compute_at(processed, tx);
        processed.tile(tx, ty, xi, yi, 32, 32).reorder(xi, yi, c, tx, ty);
        processed.parallel(ty);
    } else {
        denoised.compute_root();
        deinterleaved.compute_root();
        corrected.compute_root();
        processed.compute_root();
    }

    if (distributed) {
        processed.distribute(ty);
    }

    return processed;
}

int main(int argc, char **argv) {
    int req = MPI_THREAD_MULTIPLE, prov;
    MPI_Init_thread(&argc, &argv, req, &prov);
    assert(prov == req);
    int rank = 0, numprocs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    Param<float> color_temp(3700.0f), gamma(2.0f), contrast(50.0f);

    // These color matrices are for the sensor in the Nokia N900 and are
    // taken from the FCam source.
    float _matrix_3200[][4] = {{ 1.6697f, -0.2693f, -0.4004f, -42.4346f},
                                {-0.3576f,  1.0615f,  1.5949f, -37.1158f},
                                {-0.2175f, -1.8751f,  6.9640f, -26.6970f}};
    float _matrix_7000[][4] = {{ 2.2997f, -0.4478f,  0.1706f, -39.0923f},
                                {-0.3826f,  1.5906f, -0.2080f, -25.4311f},
                                {-0.0888f, -0.7344f,  2.2832f, -20.0826f}};
    DistributedImage<float> matrix_3200_img(4, 3), matrix_7000_img(4, 3);
    matrix_3200_img.set_domain(x, y);
    matrix_3200_img.allocate();
    matrix_7000_img.set_domain(x, y);
    matrix_7000_img.allocate();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            matrix_3200_img(j, i) = _matrix_3200[i][j];
            matrix_7000_img(j, i) = _matrix_7000[i][j];
        }
    }
    ImageParam matrix_3200(Float(32), 2), matrix_7000(Float(32), 2);
    matrix_3200.set(matrix_3200_img);
    matrix_7000.set(matrix_7000_img);

    // For a faithful comparison, make input 2592x1968.
    const int w = std::stoi(argv[1]), h = std::stoi(argv[2]);
    const int ow = ((w - 32)/32)*32, oh = ((h - 24)/32)*32;

    // Image<uint16_t> global_input(w, h);
    // Image<uint8_t> global_output(ow, oh, 3);
    DistributedImage<uint16_t> input(w, h);
    DistributedImage<uint8_t> output(ow, oh, 3);

    // shift things inwards to give us enough padding on the
    // boundaries so that we don't need to check bounds. We're going
    // to make a 2560x1920 output image, just like the FCam pipe, so
    // shift by 16, 12
    Func shifted, shifted_global;
    shifted(x, y) = input(x+16, y+12);
    // shifted_global(x, y) = global_input(x+16, y+12);

    Type result_type = UInt(bit_width);

    // Build the pipeline
    // Func processed_correct = process(shifted_global, result_type, matrix_3200, matrix_7000, color_temp, gamma, contrast, false);
    Func processed_distributed = process(shifted, result_type, matrix_3200, matrix_7000, color_temp, gamma, contrast, true);

    output.set_domain(tx, ty, c);
    output.placement().tile(tx, ty, xi, yi, 32, 32).distribute(ty);
    output.allocate();
    input.set_domain(x, y);
    input.placement().distribute(y);
    input.allocate(processed_distributed, output);

    for (int y = 0; y < input.height(); y++) {
        for (int x = 0; x < input.width(); x++) {
            int gx = input.global(0, x), gy = input.global(1, y);
            uint16_t v = (gx + gy) & 0xfff;
            input(x, y) = v;
        }
    }

    // We can generate slightly better code if we know the output is a whole number of tiles.
    // Expr out_width = processed_correct.output_buffer().width();
    // Expr out_height = processed_correct.output_buffer().height();
    // processed_correct
    //     .bound(tx, 0, (out_width/32)*32)
    //     .bound(ty, 0, (out_height/32)*32);
    //processed_correct.realize(global_output);

    processed_distributed.realize(output.get_buffer());
    const int niters = 50;
#ifdef USE_MPIP
    MPI_Pcontrol(1);
#endif
    MPITiming timing(MPI_COMM_WORLD);
    timing.barrier();
    for (int i = 0; i < niters; i++) {
        timing.start();
        processed_distributed.realize(output.get_buffer());
        MPITiming::timing_t t = timing.stop();
        timing.record(t);
    }
    timing.reduce(MPITiming::Median);

    // for (int c = 0; c < output.channels(); c++) {
    //     for (int y = 0; y < output.height(); y++) {
    //         for (int x = 0; x < output.width(); x++) {
    //             int gx = output.global(0, x), gy = output.global(1, y), gc = output.global(2, c);
    //             if (output(x, y, c) != global_output(gx, gy, gc)) {
    //                 printf("[rank %d] output(%d,%d,%d) = %u instead of %u\n", rank, x, y, c, output(x, y, c), global_output(gx, gy, gc));
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
        printf("Camera pipe test succeeded!\n");
    }

    MPI_Finalize();
    return 0;
}
