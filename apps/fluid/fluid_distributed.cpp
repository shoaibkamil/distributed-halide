#include "Halide.h"
#include "mpi_timing.h"
using namespace Halide;

// Implements Compressible Navier-Stokes (CNS) simulation.
// Based on ExACT CNS: http://portal.nersc.gov/project/CAL/exact.htm

namespace {
int rank = 0, numprocs = 0;

// Input parameters
const int DM = 3;
const int nsteps = 5;
const int plot_int = 5;
const int n_cell = 128;
const int max_grid_size = 64;
const double cfl = 0.5;
const double eta = 1.8e-4;
const double alam = 1.5e2;

const double prob_lo     = -0.1e0;
const double prob_hi     =  0.1e0;
const int lo = 0;
const int hi = n_cell-1;

const int irho = 0;
const int imx  = 1;
const int imy  = 2;
const int imz  = 3;
const int iene = 4;
const int nc = 5;

const double dx = (prob_hi - prob_lo)/n_cell;

void init_data(DistributedImage<double> &data) {
    const double twopi = 2.0 * 3.141592653589793238462643383279502884197;
    const double scale = (prob_hi - prob_lo)/twopi;
    for (int z = 0; z < data.extent(2); z++) {
        const double zloc = (double)z * dx/scale;
        for (int y = 0; y < data.extent(1); y++) {
            const double yloc = (double)y * dx/scale;
            for (int x = 0; x < data.extent(0); x++) {
                const double xloc = (double)x * dx/scale;

                const double uvel   = 1.1e4*sin(1*xloc)*sin(2*yloc)*sin(3*zloc);
                const double vvel   = 1.0e4*sin(2*xloc)*sin(4*yloc)*sin(1*zloc);
                const double wvel   = 1.2e4*sin(3*xloc)*cos(2*yloc)*sin(2*zloc);
                const double rholoc = 1.0e-3 + 1.0e-5*sin(1*xloc)*cos(2*yloc)*cos(3*zloc);
                const double eloc   = 2.5e9  + 1.0e-3*sin(2*xloc)*cos(2*yloc)*sin(2*zloc);

                // Conserved variables
                data(x, y, z, irho) = rholoc;
                data(x, y, z, imx)  = rholoc*uvel;
                data(x, y, z, imy)  = rholoc*vvel;
                data(x, y, z, imz)  = rholoc*wvel;
                data(x, y, z, iene) = rholoc*(eloc + (pow(uvel, 2)+pow(vvel,2)+pow(wvel,2))/2);
            }
        }
    }
}

inline bool parallel_IOProcessor() {
    return rank == 0;
}

void parallel_reduce(double &a, double &b, MPI_Op op) {

}

template <typename T>
T max(T a, T b) {
    return a > b ? a : b;
}

template <typename T>
T abs(T a) {
    return a < 0 ? -a : a;
}


double Huge() {
    return std::numeric_limits<double>::max();
}

void ctoprim(DistributedImage<double> &U, DistributedImage<double> &Q, double &courno) {
    const double GAMMA = 1.4;
    const double CV    = 8.3333333333e6;
    double CVinv = 1.0 / CV;

    // XXX: Need to handle boundary conditions here. Is it easier to
    // just implement in Halide now? These aren't terribly difficult
    // loops here. Or should we verify these two C loops to make sure
    // we have indexing/ordering correct first?
    for (int z = 0; z < Q.extent(2); z++) {
        for (int y = 0; y < Q.extent(1); y++) {
            for (int x = 0; x < Q.extent(0); x++) {
                const double rhoinv = 1.0/U(x,y,z,0);
                Q(x,y,z,0) = U(x,y,z,0);
                Q(x,y,z,1) = U(x,y,z,1)*rhoinv;
                Q(x,y,z,2) = U(x,y,z,2)*rhoinv;
                Q(x,y,z,3) = U(x,y,z,3)*rhoinv;

                double eint = U(x,y,z,4)*rhoinv - 0.5*(pow(Q(x,y,z,1),2) + pow(Q(x,y,z,2),2) + pow(Q(x,y,z,3),2));

                Q(x,y,z,4) = (GAMMA-1.0)*eint*U(x,y,z,0);
                Q(x,y,z,5) = eint * CVinv;
            }
        }
    }

    double courmx, courmy, courmz;
    courmx = -Huge(); courmy = -Huge(); courmz = -Huge();

    // XXX: Need to handle boundary conditions here
    for (int z = 0; z < Q.extent(2); z++) {
        for (int y = 0; y < Q.extent(1); y++) {
            for (int x = 0; x < Q.extent(0); x++) {
                const double dxinv = 1.0 / dx;
                const double c     = sqrt(GAMMA*Q(x,y,z,4)/Q(x,y,z,0));
                double courx, coury, courz;
                courx = ( c+abs(Q(x,y,z,1)) ) * dxinv;
                coury = ( c+abs(Q(x,y,z,2)) ) * dxinv;
                courz = ( c+abs(Q(x,y,z,3)) ) * dxinv;

                courmx = max( courmx, courx );
                courmy = max( courmy, coury );
                courmz = max( courmz, courz );

            }
        }
    }
    courno = max(courmx, max(courmy, max(courmz, courno)));
}

void advance(DistributedImage<double> &U, DistributedImage<double> &Q, double &dt) {
    const double OneThird      = 1.0/3.0;
    const double TwoThirds     = 2.0/3.0;
    const double OneQuarter    = 1.0/4.0;
    const double ThreeQuarters = 3.0/4.0;

    double courno = 1e-50, courno_proc = 1e-50;
    ctoprim(U, Q, courno);
    parallel_reduce(courno, courno_proc, MPI_MAX);
    dt = cfl / courno;
    if (parallel_IOProcessor()) {
        std::cout << "dt,courno " << dt << " " << courno << "\n";
    }

}

} // anonymous namespace

int main(int argc, char **argv) {
    int req = MPI_THREAD_MULTIPLE, prov;
    MPI_Init_thread(&argc, &argv, req, &prov);
    assert(prov == req);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    const int w = std::stoi(argv[1]), h = std::stoi(argv[2]), d = std::stoi(argv[3]);
    Var x("x"), y("y"), z("z"), c("c");

    DistributedImage<double> U(w, h, d, nc), Q(w, h, d, 6);
    U.set_domain(x, y, z, c);
    U.allocate();
    Q.set_domain(x, y, z, c);
    Q.allocate();

    init_data(U);

    double time = 0, dt = 0;
    for (int istep = 0; istep < nsteps; istep++) {
        if (parallel_IOProcessor()) {
            std::cout << "Advancing time step " << istep << ", time = " << time << "\n";
        }
        advance(U, Q, dt);
        time = time + dt;
    }

    // Func clamped;
    // clamped(x, y, z) = input(clamp(x, 0, input.global_width() - 1),
    //                          clamp(y, 0, input.global_height() - 1),
    //                          clamp(z, 0, input.global_channels() - 1));
    // Func heat3d;
    // Expr c0 = 0.5f, c1 = -0.25f;
    // heat3d(x, y, z) = c0 * clamped(x, y, z) + c1 * (clamped(x-1, y, z) + clamped(x+1, y, z) +
    //                                               clamped(x, y-1, z) + clamped(x, y+1, z) +
    //                                               clamped(x, y, z-1) + clamped(x, y, z+1));
    // heat3d
    //     .tile(x, y, xi, yi, 8, 8).vectorize(xi).unroll(yi)
    //     .parallel(z)
    //     .distribute(z);

    // output.set_domain(x, y, z);
    // output.placement().distribute(z);
    // output.allocate();

    // input.set_domain(x, y, z);
    // input.placement().distribute(z);
    // input.allocate(heat3d, output);

    // for (int z = 0; z < input.channels(); z++) {
    //     for (int y = 0; y < input.height(); y++) {
    //         for (int x = 0; x < input.width(); x++) {
    //             input(x, y, z) = input.global(0, x) + input.global(1, y) + input.global(2, z);
    //         }
    //     }
    // }

    // // Realize once to compile and allocate
    // heat3d.realize(input.get_buffer());
    // // Run the program and test output for correctness
    // const int niters = 100;
    // MPITiming timing(MPI_COMM_WORLD);
    // timing.barrier();
    // for (int i = 0; i < niters; i++) {
    //     timing.start();
    //     heat3d.realize(output.get_buffer());
    //     MPITiming::timing_t t = timing.stop();
    //     timing.record(t);
    // }
    // timing.reduce(MPITiming::Median);

    // timing.gather(MPITiming::Max);
    // timing.report();
    // MPI_Finalize();
    return 0;
}
