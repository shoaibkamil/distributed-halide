#include "Halide.h"
#include "mpi_timing.h"
#include <iomanip>
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

const double GAMMA = 1.4;

// Halide globals
int global_w = 0, global_h = 0, global_d = 0;
Var x("x"), y("y"), z("z"), c("c");
Func ctoprim("ctoprim"), courno_func("courno");

void init_data(DistributedImage<double> &data) {
    // XXX: make this a Halide stage as well so that we can
    // parallelize it the same way as in the Fortran code
    // (init_data.f90)
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

double max(double a, double b) {
    return a > b ? a : b;
}

double abs(double a) {
    return a < 0 ? -a : a;
}

double Huge() {
    return std::numeric_limits<double>::max();
}

Func build_ctoprim(Func U) {
    const double CV    = 8.3333333333e6;
    const double CVinv = 1.0 / CV;

    Expr rhoinv = Expr(1.0)/U(x,y,z,irho);

    Expr density, xvel, yvel, zvel, pressure, temperature;
    density = U(x,y,z,irho);
    xvel = U(x,y,z,imx)*rhoinv;
    yvel = U(x,y,z,imy)*rhoinv;
    zvel = U(x,y,z,imz)*rhoinv;

    Expr eint = U(x,y,z,iene)*rhoinv - Expr(0.5)*(pow(xvel,2) + pow(yvel,2) + pow(zvel,2));
    pressure = Expr(GAMMA-1.0) * eint * U(x,y,z,irho);
    temperature = eint * Expr(CVinv);

    Func Q;
    //Q(x,y,z) = Tuple(density, xvel, yvel, zvel, pressure, temperature);
    Q(x,y,z,c) = select(c == 0, density,
                        c == 1, xvel,
                        c == 2, yvel,
                        c == 3, zvel,
                        c == 4, pressure,
                        temperature);
    // Eliminate some performance loss of the select by bounding and unrolling 'c':
    Q.bound(c, 0, 6).unroll(c);
    return Q;
}

Func build_courno(Func Q) {
    // XXX: this can be combined with ctoprim, as a final element in
    // the tuple. Don't do that optimization yet, to get a faithful
    // performance comparison.

    // XXX: This needs to be a local reduction and then have an
    // external global reduction of the local values, as in the
    // Fortran code.
    RDom r(0, global_w, 0, global_h, 0, global_d);

    Expr dxinv = Expr(1.0 / dx);
    Expr c     = sqrt(Expr(GAMMA)*Q(r.x,r.y,r.z,4)/Q(r.x,r.y,r.z,0));

    Expr courx = ( c+abs(Q(r.x,r.y,r.z,1)) ) * dxinv;
    Expr coury = ( c+abs(Q(r.x,r.y,r.z,2)) ) * dxinv;
    Expr courz = ( c+abs(Q(r.x,r.y,r.z,3)) ) * dxinv;

    Expr courmx = max( Expr(-Huge()), courx );
    Expr courmy = max( Expr(-Huge()), coury );
    Expr courmz = max( Expr(-Huge()), courz );

    Expr maxcourpt = max(courmx, max(courmy, courmz));

    // Not called 'courno' because the actual value of courno is the
    // max of this value and the old value of courno.
    Func helper("helper");
    helper() = maximum(r, maxcourpt);
    return helper;
}

void build_pipeline(Func UAccessor, Func QAccessor) {
    ctoprim = build_ctoprim(UAccessor);
    courno_func  = build_courno(QAccessor);
}

void ctoprim_fort(DistributedImage<double> &U, DistributedImage<double> &Q, double &courno) {
    // const double CV    = 8.3333333333e6;
    // double CVinv = 1.0 / CV;

    // // XXX: Need to handle boundary conditions here. Is it easier to
    // // just implement in Halide now? These aren't terribly difficult
    // // loops here. Or should we verify these two C loops to make sure
    // // we have indexing/ordering correct first?
    // for (int z = 0; z < Q.extent(2); z++) {
    //     for (int y = 0; y < Q.extent(1); y++) {
    //         for (int x = 0; x < Q.extent(0); x++) {
    //             const double rhoinv = 1.0/U(x,y,z,0);
    //             Q(x,y,z,0) = U(x,y,z,0);
    //             Q(x,y,z,1) = U(x,y,z,1)*rhoinv;
    //             Q(x,y,z,2) = U(x,y,z,2)*rhoinv;
    //             Q(x,y,z,3) = U(x,y,z,3)*rhoinv;

    //             double eint = U(x,y,z,4)*rhoinv - 0.5*(pow(Q(x,y,z,1),2) + pow(Q(x,y,z,2),2) + pow(Q(x,y,z,3),2));

    //             Q(x,y,z,4) = (GAMMA-1.0)*eint*U(x,y,z,0);
    //             Q(x,y,z,5) = eint * CVinv;
    //         }
    //     }
    // }

    // double courmx, courmy, courmz;
    // courmx = -Huge(); courmy = -Huge(); courmz = -Huge();

    // // XXX: Need to handle boundary conditions here
    // for (int z = 0; z < Q.extent(2); z++) {
    //     for (int y = 0; y < Q.extent(1); y++) {
    //         for (int x = 0; x < Q.extent(0); x++) {
    //             const double dxinv = 1.0 / dx;
    //             const double c     = sqrt(GAMMA*Q(x,y,z,4)/Q(x,y,z,0));
    //             double courx, coury, courz;
    //             courx = ( c+abs(Q(x,y,z,1)) ) * dxinv;
    //             coury = ( c+abs(Q(x,y,z,2)) ) * dxinv;
    //             courz = ( c+abs(Q(x,y,z,3)) ) * dxinv;

    //             courmx = max( courmx, courx );
    //             courmy = max( courmy, coury );
    //             courmz = max( courmz, courz );

    //         }
    //     }
    // }
    // courno = max(courmx, max(courmy, max(courmz, courno)));
}

void advance(DistributedImage<double> &U, DistributedImage<double> &Q, double &dt) {
    const double OneThird      = 1.0/3.0;
    const double TwoThirds     = 2.0/3.0;
    const double OneQuarter    = 1.0/4.0;
    const double ThreeQuarters = 3.0/4.0;

    // double courno = 1e-50, courno_proc = 1e-50;
    // ctoprim(U, Q, courno);
    // parallel_reduce(courno, courno_proc, MPI_MAX);

    double courno = 1e-50;
    ctoprim.realize(Q);
    courno = max(evaluate<double>(courno_func), courno);
    // XXX: parallel_reduce courno
    dt = cfl / courno;

    if (parallel_IOProcessor()) {
        std::cout << std::scientific << "dt,courno " << std::setprecision(std::numeric_limits<double>::digits10) << dt << " " << courno << "\n";
    }
}

} // anonymous namespace

int main(int argc, char **argv) {
    int req = MPI_THREAD_MULTIPLE, prov;
    MPI_Init_thread(&argc, &argv, req, &prov);
    assert(prov == req);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    global_w = std::stoi(argv[1]);
    global_h = std::stoi(argv[2]);
    global_d = std::stoi(argv[3]);

    // XXX: should probably make the component the innermost
    // dimension, then w,h,d.
    DistributedImage<double> U(global_w, global_h, global_d, nc), Q(global_w, global_h, global_d, 6);

    // Impose periodic boundary conditions on U and Q
    std::vector<std::pair<Expr, Expr>>
        global_bounds_U = {std::make_pair(0, global_w), std::make_pair(0, global_h),
                           std::make_pair(0, global_d), std::make_pair(0, nc)},
        global_bounds_Q = {std::make_pair(0, global_w), std::make_pair(0, global_h),
                           std::make_pair(0, global_d), std::make_pair(0, 6)};
    Func Ut, Qt;
    Ut(x,y,z,c) = U(x,y,z,c);
    Qt(x,y,z,c) = Q(x,y,z,c);
    Func UAccessor("UAccessor"), QAccessor("QAccessor");
    UAccessor = BoundaryConditions::repeat_image(Ut, global_bounds_U);
    QAccessor = BoundaryConditions::repeat_image(Qt, global_bounds_Q);

    build_pipeline(UAccessor, QAccessor);

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
