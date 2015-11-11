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

const int qu    = 1;
const int qv    = 2;
const int qw    = 3;
const int qpres = 4;

const double dx = (prob_hi - prob_lo)/n_cell;

const double GAMMA = 1.4;

// Halide globals
int global_w = 0, global_h = 0, global_d = 0;
Var x("x"), y("y"), z("z"), c("c");
Param<double> timestep;
Func ctoprim("ctoprim"), courno_func("courno");
Func diffterm("diffterm"), hypterm("hypterm");
Func Uonethird("Uonethird"), Utwothirds("Utwothirds"), Uone("Uone");
ImageParam Up(Float(64), 4),
    Unewp(Float(64), 4),
    Qp(Float(64), 4),
    Dp(Float(64), 4),
    Fp(Float(64), 4);

template <typename T>
static void print_imgflat4d(DistributedImage<T> &img) {
    assert(img.dimensions() == 4);
    std::cout << std::scientific << std::setprecision(std::numeric_limits<double>::digits10);
    for (int c = 0; c < img.extent(3); c++) {
        for (int z = 0; z < img.extent(2); z++) {
            for (int y = 0; y < img.extent(1); y++) {
                for (int x = 0; x < img.extent(0); x++) {
                    std::cout << img(x,y,z,c) << " ";
                }
            }
        }
    }
    std::cout << "\n";
}

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

Func build_diffterm(Func Q) {
    Expr OneThird   = Expr(1.0)/Expr(3.0);
    Expr TwoThirds  = Expr(2.0)/Expr(3.0);
    Expr FourThirds = Expr(4.0)/Expr(3.0);

    Expr ALP = Expr( 0.8);
    Expr BET = Expr(-0.2);
    Expr GAM = Expr( 4.0)/Expr(105.0);
    Expr DEL = Expr(-1.0)/Expr(280.0);
    Expr CENTER = Expr(-205.0)/Expr(72.0);
    Expr OFF1   = Expr(8.0)/Expr(5.0);
    Expr OFF2   = Expr(-0.2);
    Expr OFF3   = Expr(8.0)/Expr(315.0);
    Expr OFF4   = Expr(-1.0)/Expr(560.0);
    Expr dxinv = Expr(1.0 / dx);

    Func difflux("difflux");
    // Pure step handles irho case
    difflux(x,y,z,c) = Expr(0.0);

    Expr ux_calc =
        (ALP*(Q(x+1,y,z,qu)-Q(x-1,y,z,qu))
         + BET*(Q(x+2,y,z,qu)-Q(x-2,y,z,qu))
         + GAM*(Q(x+3,y,z,qu)-Q(x-3,y,z,qu))
         + DEL*(Q(x+4,y,z,qu)-Q(x-4,y,z,qu)))*dxinv;
    Expr vx_calc =
        (ALP*(Q(x+1,y,z,qv)-Q(x-1,y,z,qv))
         + BET*(Q(x+2,y,z,qv)-Q(x-2,y,z,qv))
         + GAM*(Q(x+3,y,z,qv)-Q(x-3,y,z,qv))
         + DEL*(Q(x+4,y,z,qv)-Q(x-4,y,z,qv)))*dxinv;
    Expr wx_calc =
        (ALP*(Q(x+1,y,z,qw)-Q(x-1,y,z,qw))
         + BET*(Q(x+2,y,z,qw)-Q(x-2,y,z,qw))
         + GAM*(Q(x+3,y,z,qw)-Q(x-3,y,z,qw))
         + DEL*(Q(x+4,y,z,qw)-Q(x-4,y,z,qw)))*dxinv;

    Func loop1, ux, vx, wx;
    // Do it this way so that ux, vx and wx are calculated in the same
    // loop nest.
    loop1(x, y, z) = {ux_calc, vx_calc, wx_calc};
    ux(x, y, z) = loop1(x,y,z)[0];
    vx(x, y, z) = loop1(x,y,z)[1];
    wx(x, y, z) = loop1(x,y,z)[2];

    Expr uy_calc =
        (ALP*(Q(x,y+1,z,qu)-Q(x,y-1,z,qu))
         + BET*(Q(x,y+2,z,qu)-Q(x,y-2,z,qu))
         + GAM*(Q(x,y+3,z,qu)-Q(x,y-3,z,qu))
         + DEL*(Q(x,y+4,z,qu)-Q(x,y-4,z,qu)))*dxinv;

    Expr vy_calc =
        (ALP*(Q(x,y+1,z,qv)-Q(x,y-1,z,qv))
         + BET*(Q(x,y+2,z,qv)-Q(x,y-2,z,qv))
         + GAM*(Q(x,y+3,z,qv)-Q(x,y-3,z,qv))
         + DEL*(Q(x,y+4,z,qv)-Q(x,y-4,z,qv)))*dxinv;

    Expr wy_calc =
        (ALP*(Q(x,y+1,z,qw)-Q(x,y-1,z,qw))
         + BET*(Q(x,y+2,z,qw)-Q(x,y-2,z,qw))
         + GAM*(Q(x,y+3,z,qw)-Q(x,y-3,z,qw))
         + DEL*(Q(x,y+4,z,qw)-Q(x,y-4,z,qw)))*dxinv;

    Func loop2, uy, vy, wy;
    loop2(x, y, z) = {uy_calc, vy_calc, wy_calc};
    uy(x, y, z) = loop2(x,y,z)[0];
    vy(x, y, z) = loop2(x,y,z)[1];
    wy(x, y, z) = loop2(x,y,z)[2];

    Expr uz_calc =
        (ALP*(Q(x,y,z+1,qu)-Q(x,y,z-1,qu))
         + BET*(Q(x,y,z+2,qu)-Q(x,y,z-2,qu))
         + GAM*(Q(x,y,z+3,qu)-Q(x,y,z-3,qu))
         + DEL*(Q(x,y,z+4,qu)-Q(x,y,z-4,qu)))*dxinv;

    Expr vz_calc =
        (ALP*(Q(x,y,z+1,qv)-Q(x,y,z-1,qv))
         + BET*(Q(x,y,z+2,qv)-Q(x,y,z-2,qv))
         + GAM*(Q(x,y,z+3,qv)-Q(x,y,z-3,qv))
         + DEL*(Q(x,y,z+4,qv)-Q(x,y,z-4,qv)))*dxinv;

    Expr wz_calc =
        (ALP*(Q(x,y,z+1,qw)-Q(x,y,z-1,qw))
         + BET*(Q(x,y,z+2,qw)-Q(x,y,z-2,qw))
         + GAM*(Q(x,y,z+3,qw)-Q(x,y,z-3,qw))
         + DEL*(Q(x,y,z+4,qw)-Q(x,y,z-4,qw)))*dxinv;

    Func loop3, uz, vz, wz;
    loop3(x, y, z) = {uz_calc, vz_calc, wz_calc};
    uz(x, y, z) = loop3(x,y,z)[0];
    vz(x, y, z) = loop3(x,y,z)[1];
    wz(x, y, z) = loop3(x,y,z)[2];

    Expr uxx = (CENTER*Q(x,y,z,qu)
                + OFF1*(Q(x+1,y,z,qu)+Q(x-1,y,z,qu))
                + OFF2*(Q(x+2,y,z,qu)+Q(x-2,y,z,qu))
                + OFF3*(Q(x+3,y,z,qu)+Q(x-3,y,z,qu))
                + OFF4*(Q(x+4,y,z,qu)+Q(x-4,y,z,qu)))*pow(dxinv,2);

    Expr uyy = (CENTER*Q(x,y,z,qu)
                + OFF1*(Q(x,y+1,z,qu)+Q(x,y-1,z,qu))
                + OFF2*(Q(x,y+2,z,qu)+Q(x,y-2,z,qu))
                + OFF3*(Q(x,y+3,z,qu)+Q(x,y-3,z,qu))
                + OFF4*(Q(x,y+4,z,qu)+Q(x,y-4,z,qu)))*pow(dxinv,2);

    Expr uzz = (CENTER*Q(x,y,z,qu)
                + OFF1*(Q(x,y,z+1,qu)+Q(x,y,z-1,qu))
                + OFF2*(Q(x,y,z+2,qu)+Q(x,y,z-2,qu))
                + OFF3*(Q(x,y,z+3,qu)+Q(x,y,z-3,qu))
                + OFF4*(Q(x,y,z+4,qu)+Q(x,y,z-4,qu)))*pow(dxinv,2);

    Expr vyx = (ALP*(vy(x+1,y,z)-vy(x-1,y,z))
                + BET*(vy(x+2,y,z)-vy(x-2,y,z))
                + GAM*(vy(x+3,y,z)-vy(x-3,y,z))
                + DEL*(vy(x+4,y,z)-vy(x-4,y,z)))*dxinv;

    Expr wzx = (ALP*(wz(x+1,y,z)-wz(x-1,y,z))
                + BET*(wz(x+2,y,z)-wz(x-2,y,z))
                + GAM*(wz(x+3,y,z)-wz(x-3,y,z))
                + DEL*(wz(x+4,y,z)-wz(x-4,y,z)))*dxinv;

    // Update 0: imx
    difflux(x,y,z,imx) = Expr(eta)*(FourThirds*uxx + uyy + uzz + OneThird*(vyx+wzx));

    Expr vxx = (CENTER*Q(x,y,z,qv)
                + OFF1*(Q(x+1,y,z,qv)+Q(x-1,y,z,qv))
                + OFF2*(Q(x+2,y,z,qv)+Q(x-2,y,z,qv))
                + OFF3*(Q(x+3,y,z,qv)+Q(x-3,y,z,qv))
                + OFF4*(Q(x+4,y,z,qv)+Q(x-4,y,z,qv)))*pow(dxinv,2);

    Expr vyy = (CENTER*Q(x,y,z,qv)
                + OFF1*(Q(x,y+1,z,qv)+Q(x,y-1,z,qv))
                + OFF2*(Q(x,y+2,z,qv)+Q(x,y-2,z,qv))
                + OFF3*(Q(x,y+3,z,qv)+Q(x,y-3,z,qv))
                + OFF4*(Q(x,y+4,z,qv)+Q(x,y-4,z,qv)))*pow(dxinv,2);

    Expr vzz = (CENTER*Q(x,y,z,qv)
                + OFF1*(Q(x,y,z+1,qv)+Q(x,y,z-1,qv))
                + OFF2*(Q(x,y,z+2,qv)+Q(x,y,z-2,qv))
                + OFF3*(Q(x,y,z+3,qv)+Q(x,y,z-3,qv))
                + OFF4*(Q(x,y,z+4,qv)+Q(x,y,z-4,qv)))*pow(dxinv,2);

    Expr uxy = (ALP*(ux(x,y+1,z)-ux(x,y-1,z))
                + BET*(ux(x,y+2,z)-ux(x,y-2,z))
                + GAM*(ux(x,y+3,z)-ux(x,y-3,z))
                + DEL*(ux(x,y+4,z)-ux(x,y-4,z)))*dxinv;

    Expr wzy = (ALP*(wz(x,y+1,z)-wz(x,y-1,z))
                + BET*(wz(x,y+2,z)-wz(x,y-2,z))
                + GAM*(wz(x,y+3,z)-wz(x,y-3,z))
                + DEL*(wz(x,y+4,z)-wz(x,y-4,z)))*dxinv;

    // Update 1: imy
    difflux(x,y,z,imy) = Expr(eta)*(vxx + FourThirds*vyy + vzz + OneThird*(uxy+wzy));

    Expr wxx = (CENTER*Q(x,y,z,qw)
                + OFF1*(Q(x+1,y,z,qw)+Q(x-1,y,z,qw))
                + OFF2*(Q(x+2,y,z,qw)+Q(x-2,y,z,qw))
                + OFF3*(Q(x+3,y,z,qw)+Q(x-3,y,z,qw))
                + OFF4*(Q(x+4,y,z,qw)+Q(x-4,y,z,qw)))*pow(dxinv,2);

    Expr wyy = (CENTER*Q(x,y,z,qw)
                + OFF1*(Q(x,y+1,z,qw)+Q(x,y-1,z,qw))
                + OFF2*(Q(x,y+2,z,qw)+Q(x,y-2,z,qw))
                + OFF3*(Q(x,y+3,z,qw)+Q(x,y-3,z,qw))
                + OFF4*(Q(x,y+4,z,qw)+Q(x,y-4,z,qw)))*pow(dxinv,2);

    Expr wzz = (CENTER*Q(x,y,z,qw)
                + OFF1*(Q(x,y,z+1,qw)+Q(x,y,z-1,qw))
                + OFF2*(Q(x,y,z+2,qw)+Q(x,y,z-2,qw))
                + OFF3*(Q(x,y,z+3,qw)+Q(x,y,z-3,qw))
                + OFF4*(Q(x,y,z+4,qw)+Q(x,y,z-4,qw)))*pow(dxinv,2);

    Expr uxz = (ALP*(ux(x,y,z+1)-ux(x,y,z-1))
                + BET*(ux(x,y,z+2)-ux(x,y,z-2))
                + GAM*(ux(x,y,z+3)-ux(x,y,z-3))
                + DEL*(ux(x,y,z+4)-ux(x,y,z-4)))*dxinv;

    Expr vyz = (ALP*(vy(x,y,z+1)-vy(x,y,z-1))
                + BET*(vy(x,y,z+2)-vy(x,y,z-2))
                + GAM*(vy(x,y,z+3)-vy(x,y,z-3))
                + DEL*(vy(x,y,z+4)-vy(x,y,z-4)))*dxinv;

    // Update 2: imz
    difflux(x,y,z,imz) = Expr(eta)*(wxx + wyy + FourThirds*wzz + OneThird*(uxz+vyz));

    Expr txx = (CENTER*Q(x,y,z,6)
                + OFF1*(Q(x+1,y,z,6)+Q(x-1,y,z,6))
                + OFF2*(Q(x+2,y,z,6)+Q(x-2,y,z,6))
                + OFF3*(Q(x+3,y,z,6)+Q(x-3,y,z,6))
                + OFF4*(Q(x+4,y,z,6)+Q(x-4,y,z,6)))*pow(dxinv,2);

    Expr tyy = (CENTER*Q(x,y,z,6)
                + OFF1*(Q(x,y+1,z,6)+Q(x,y-1,z,6))
                + OFF2*(Q(x,y+2,z,6)+Q(x,y-2,z,6))
                + OFF3*(Q(x,y+3,z,6)+Q(x,y-3,z,6))
                + OFF4*(Q(x,y+4,z,6)+Q(x,y-4,z,6)))*pow(dxinv,2);

    Expr tzz = (CENTER*Q(x,y,z,6)
                + OFF1*(Q(x,y,z+1,6)+Q(x,y,z-1,6))
                + OFF2*(Q(x,y,z+2,6)+Q(x,y,z-2,6))
                + OFF3*(Q(x,y,z+3,6)+Q(x,y,z-3,6))
                + OFF4*(Q(x,y,z+4,6)+Q(x,y,z-4,6)))*pow(dxinv,2);

    Expr divu  = TwoThirds*(ux(x,y,z)+vy(x,y,z)+wz(x,y,z));
    Expr tauxx = Expr(2.0)*ux(x,y,z) - divu;
    Expr tauyy = Expr(2.0)*vy(x,y,z) - divu;
    Expr tauzz = Expr(2.0)*wz(x,y,z) - divu;
    Expr tauxy = uy(x,y,z)+vx(x,y,z);
    Expr tauxz = uz(x,y,z)+wx(x,y,z);
    Expr tauyz = vz(x,y,z)+wy(x,y,z);

    Expr mechwork = tauxx*ux(x,y,z) +
        tauyy*vy(x,y,z) +
        tauzz*wz(x,y,z) + pow(tauxy,2)+pow(tauxz,2)+pow(tauyz,2);

    mechwork = Expr(eta)*mechwork
        + difflux(x,y,z,imx)*Q(x,y,z,qu)
        + difflux(x,y,z,imy)*Q(x,y,z,qv)
        + difflux(x,y,z,imz)*Q(x,y,z,qw);

    // Update 3: iene
    difflux(x,y,z,iene) = Expr(alam)*(txx+tyy+tzz) + mechwork;

    return difflux;
}

Func build_hypterm(Func U, Func Q) {
    Expr OneThird   = Expr(1.0)/Expr(3.0);
    Expr TwoThirds  = Expr(2.0)/Expr(3.0);
    Expr FourThirds = Expr(4.0)/Expr(3.0);

    Expr ALP = Expr( 0.8);
    Expr BET = Expr(-0.2);
    Expr GAM = Expr( 4.0)/Expr(105.0);
    Expr DEL = Expr(-1.0)/Expr(280.0);
    Expr dxinv = Expr(1.0 / dx);

    Expr unp1 = Q(x+1,y,z,qu);
    Expr unp2 = Q(x+2,y,z,qu);
    Expr unp3 = Q(x+3,y,z,qu);
    Expr unp4 = Q(x+4,y,z,qu);
    Expr unm1 = Q(x-1,y,z,qu);
    Expr unm2 = Q(x-2,y,z,qu);
    Expr unm3 = Q(x-3,y,z,qu);
    Expr unm4 = Q(x-4,y,z,qu);

    Expr flux_irho_calc =
        - (ALP*(U(x+1,y,z,imx)-U(x-1,y,z,imx))
           + BET*(U(x+2,y,z,imx)-U(x-2,y,z,imx))
           + GAM*(U(x+3,y,z,imx)-U(x-3,y,z,imx))
           + DEL*(U(x+4,y,z,imx)-U(x-4,y,z,imx)))*dxinv;

    Expr flux_imx_calc =
        - (ALP*(U(x+1,y,z,imx)*unp1-U(x-1,y,z,imx)*unm1
                + (Q(x+1,y,z,qpres)-Q(x-1,y,z,qpres)))
           + BET*(U(x+2,y,z,imx)*unp2-U(x-2,y,z,imx)*unm2
                  + (Q(x+2,y,z,qpres)-Q(x-2,y,z,qpres)))
           + GAM*(U(x+3,y,z,imx)*unp3-U(x-3,y,z,imx)*unm3
                  + (Q(x+3,y,z,qpres)-Q(x-3,y,z,qpres)))
           + DEL*(U(x+4,y,z,imx)*unp4-U(x-4,y,z,imx)*unm4
                  + (Q(x+4,y,z,qpres)-Q(x-4,y,z,qpres))))*dxinv;

    Expr flux_imy_calc =
        - (ALP*(U(x+1,y,z,imy)*unp1-U(x-1,y,z,imy)*unm1)
           + BET*(U(x+2,y,z,imy)*unp2-U(x-2,y,z,imy)*unm2)
           + GAM*(U(x+3,y,z,imy)*unp3-U(x-3,y,z,imy)*unm3)
           + DEL*(U(x+4,y,z,imy)*unp4-U(x-4,y,z,imy)*unm4))*dxinv;

    Expr flux_imz_calc =
        - (ALP*(U(x+1,y,z,imz)*unp1-U(x-1,y,z,imz)*unm1)
           + BET*(U(x+2,y,z,imz)*unp2-U(x-2,y,z,imz)*unm2)
           + GAM*(U(x+3,y,z,imz)*unp3-U(x-3,y,z,imz)*unm3)
           + DEL*(U(x+4,y,z,imz)*unp4-U(x-4,y,z,imz)*unm4))*dxinv;

    Expr flux_iene_calc =
        - (ALP*(U(x+1,y,z,iene)*unp1-U(x-1,y,z,iene)*unm1
                + (Q(x+1,y,z,qpres)*unp1-Q(x-1,y,z,qpres)*unm1))
           + BET*(U(x+2,y,z,iene)*unp2-U(x-2,y,z,iene)*unm2
                  + (Q(x+2,y,z,qpres)*unp2-Q(x-2,y,z,qpres)*unm2))
           + GAM*(U(x+3,y,z,iene)*unp3-U(x-3,y,z,iene)*unm3
                  + (Q(x+3,y,z,qpres)*unp3-Q(x-3,y,z,qpres)*unm3))
           + DEL*(U(x+4,y,z,iene)*unp4-U(x-4,y,z,iene)*unm4
                  + (Q(x+4,y,z,qpres)*unp4-Q(x-4,y,z,qpres)*unm4)))*dxinv;

    Func loop1("loop1");
    loop1(x, y, z, c) = select(c == 0, flux_irho_calc,
                               c == 1, flux_imx_calc,
                               c == 2, flux_imy_calc,
                               c == 3, flux_imz_calc,
                               flux_iene_calc);
    loop1.bound(c, 0, 5).unroll(c).compute_root();


    unp1 = Q(x,y+1,z,qv);
    unp2 = Q(x,y+2,z,qv);
    unp3 = Q(x,y+3,z,qv);
    unp4 = Q(x,y+4,z,qv);
    unm1 = Q(x,y-1,z,qv);
    unm2 = Q(x,y-2,z,qv);
    unm3 = Q(x,y-3,z,qv);
    unm4 = Q(x,y-4,z,qv);

    flux_irho_calc=loop1(x,y,z,irho) -
        (ALP*(U(x,y+1,z,imy)-U(x,y-1,z,imy))
         + BET*(U(x,y+2,z,imy)-U(x,y-2,z,imy))
         + GAM*(U(x,y+3,z,imy)-U(x,y-3,z,imy))
         + DEL*(U(x,y+4,z,imy)-U(x,y-4,z,imy)))*dxinv;

    flux_imx_calc=loop1(x,y,z,imx) -
        (ALP*(U(x,y+1,z,imx)*unp1-U(x,y-1,z,imx)*unm1)
         + BET*(U(x,y+2,z,imx)*unp2-U(x,y-2,z,imx)*unm2)
         + GAM*(U(x,y+3,z,imx)*unp3-U(x,y-3,z,imx)*unm3)
         + DEL*(U(x,y+4,z,imx)*unp4-U(x,y-4,z,imx)*unm4))*dxinv;

    flux_imy_calc=loop1(x,y,z,imy) -
        (ALP*(U(x,y+1,z,imy)*unp1-U(x,y-1,z,imy)*unm1
              + (Q(x,y+1,z,qpres)-Q(x,y-1,z,qpres)))
         + BET*(U(x,y+2,z,imy)*unp2-U(x,y-2,z,imy)*unm2
                + (Q(x,y+2,z,qpres)-Q(x,y-2,z,qpres)))
         + GAM*(U(x,y+3,z,imy)*unp3-U(x,y-3,z,imy)*unm3
                + (Q(x,y+3,z,qpres)-Q(x,y-3,z,qpres)))
         + DEL*(U(x,y+4,z,imy)*unp4-U(x,y-4,z,imy)*unm4
                + (Q(x,y+4,z,qpres)-Q(x,y-4,z,qpres))))*dxinv;

    flux_imz_calc=loop1(x,y,z,imz) -
        (ALP*(U(x,y+1,z,imz)*unp1-U(x,y-1,z,imz)*unm1)
         + BET*(U(x,y+2,z,imz)*unp2-U(x,y-2,z,imz)*unm2)
         + GAM*(U(x,y+3,z,imz)*unp3-U(x,y-3,z,imz)*unm3)
         + DEL*(U(x,y+4,z,imz)*unp4-U(x,y-4,z,imz)*unm4))*dxinv;

    flux_iene_calc=loop1(x,y,z,iene) -
        (ALP*(U(x,y+1,z,iene)*unp1-U(x,y-1,z,iene)*unm1
              + (Q(x,y+1,z,qpres)*unp1-Q(x,y-1,z,qpres)*unm1))
         + BET*(U(x,y+2,z,iene)*unp2-U(x,y-2,z,iene)*unm2
                + (Q(x,y+2,z,qpres)*unp2-Q(x,y-2,z,qpres)*unm2))
         + GAM*(U(x,y+3,z,iene)*unp3-U(x,y-3,z,iene)*unm3
                + (Q(x,y+3,z,qpres)*unp3-Q(x,y-3,z,qpres)*unm3))
         + DEL*(U(x,y+4,z,iene)*unp4-U(x,y-4,z,iene)*unm4
                + (Q(x,y+4,z,qpres)*unp4-Q(x,y-4,z,qpres)*unm4)))*dxinv;


    Func loop2("loop2");
    loop2(x, y, z, c) = select(c == 0, flux_irho_calc,
                               c == 1, flux_imx_calc,
                               c == 2, flux_imy_calc,
                               c == 3, flux_imz_calc,
                               flux_iene_calc);
    loop2.bound(c, 0, 5).unroll(c).compute_root();


    unp1 = Q(x,y,z+1,qw);
    unp2 = Q(x,y,z+2,qw);
    unp3 = Q(x,y,z+3,qw);
    unp4 = Q(x,y,z+4,qw);
    unm1 = Q(x,y,z-1,qw);
    unm2 = Q(x,y,z-2,qw);
    unm3 = Q(x,y,z-3,qw);
    unm4 = Q(x,y,z-4,qw);

    flux_irho_calc=loop2(x,y,z,irho) -
        (ALP*(U(x,y,z+1,imz)-U(x,y,z-1,imz))
         + BET*(U(x,y,z+2,imz)-U(x,y,z-2,imz))
         + GAM*(U(x,y,z+3,imz)-U(x,y,z-3,imz))
         + DEL*(U(x,y,z+4,imz)-U(x,y,z-4,imz)))*dxinv;

    flux_imx_calc=loop2(x,y,z,imx) -
        (ALP*(U(x,y,z+1,imx)*unp1-U(x,y,z-1,imx)*unm1)
         + BET*(U(x,y,z+2,imx)*unp2-U(x,y,z-2,imx)*unm2)
         + GAM*(U(x,y,z+3,imx)*unp3-U(x,y,z-3,imx)*unm3)
         + DEL*(U(x,y,z+4,imx)*unp4-U(x,y,z-4,imx)*unm4))*dxinv;

    flux_imy_calc=loop2(x,y,z,imy) -
        (ALP*(U(x,y,z+1,imy)*unp1-U(x,y,z-1,imy)*unm1)
         + BET*(U(x,y,z+2,imy)*unp2-U(x,y,z-2,imy)*unm2)
         + GAM*(U(x,y,z+3,imy)*unp3-U(x,y,z-3,imy)*unm3)
         + DEL*(U(x,y,z+4,imy)*unp4-U(x,y,z-4,imy)*unm4))*dxinv;

    flux_imz_calc=loop2(x,y,z,imz) -
        (ALP*(U(x,y,z+1,imz)*unp1-U(x,y,z-1,imz)*unm1
              + (Q(x,y,z+1,qpres)-Q(x,y,z-1,qpres)))
         + BET*(U(x,y,z+2,imz)*unp2-U(x,y,z-2,imz)*unm2
                + (Q(x,y,z+2,qpres)-Q(x,y,z-2,qpres)))
         + GAM*(U(x,y,z+3,imz)*unp3-U(x,y,z-3,imz)*unm3
                + (Q(x,y,z+3,qpres)-Q(x,y,z-3,qpres)))
         + DEL*(U(x,y,z+4,imz)*unp4-U(x,y,z-4,imz)*unm4
                + (Q(x,y,z+4,qpres)-Q(x,y,z-4,qpres))))*dxinv;

    flux_iene_calc=loop2(x,y,z,iene) -
        (ALP*(U(x,y,z+1,iene)*unp1-U(x,y,z-1,iene)*unm1
              + (Q(x,y,z+1,qpres)*unp1-Q(x,y,z-1,qpres)*unm1))
         + BET*(U(x,y,z+2,iene)*unp2-U(x,y,z-2,iene)*unm2
                + (Q(x,y,z+2,qpres)*unp2-Q(x,y,z-2,qpres)*unm2))
         + GAM*(U(x,y,z+3,iene)*unp3-U(x,y,z-3,iene)*unm3
                + (Q(x,y,z+3,qpres)*unp3-Q(x,y,z-3,qpres)*unm3))
         + DEL*(U(x,y,z+4,iene)*unp4-U(x,y,z-4,iene)*unm4
                + (Q(x,y,z+4,qpres)*unp4-Q(x,y,z-4,qpres)*unm4)))*dxinv;

    Func flux("flux");
    flux(x, y, z, c) = select(c == 0, flux_irho_calc,
                              c == 1, flux_imx_calc,
                              c == 2, flux_imy_calc,
                              c == 3, flux_imz_calc,
                              flux_iene_calc);
    flux.bound(c, 0, 5).unroll(c).compute_root();

    return flux;
}

Func build_Uonethird(Func U, Func D, Func F) {
    Func Uonethird("Uonethird");
    Uonethird(x, y, z, c) = U(x,y,z,c) + timestep * (D(x,y,z,c) + F(x,y,z,c));
    return Uonethird;
}

Func build_Utwothirds(Func U, Func Unew, Func D, Func F) {
    Expr OneQuarter    = Expr(1.0)/Expr(4.0);
    Expr ThreeQuarters = Expr(3.0)/Expr(4.0);

    Func Utwothirds("Utwothirds");
    Utwothirds(x, y, z, c) = ThreeQuarters * U(x,y,z,c) +
        OneQuarter * (Unew(x,y,z,c) + timestep * (D(x,y,z,c) + F(x,y,z,c)));
    return Utwothirds;
}

Func build_Uone(Func U, Func Unew, Func D, Func F) {
    Expr OneThird  = Expr(1.0)/Expr(3.0);
    Expr TwoThirds = Expr(2.0)/Expr(3.0);

    Func Uone("Uone");
    Uone(x,y,z,c) = OneThird * U(x,y,z,c) +
        TwoThirds * (Unew(x,y,z,c) + timestep * (D(x,y,z,c) + F(x,y,z,c)));
    return Uone;
}


void build_pipeline(Func UAccessor, Func UnewAccessor, Func QAccessor, Func DAccessor, Func FAccessor) {
    ctoprim = build_ctoprim(UAccessor);
    courno_func  = build_courno(QAccessor);
    diffterm = build_diffterm(QAccessor);
    hypterm = build_hypterm(UAccessor, QAccessor);
    Uonethird = build_Uonethird(UAccessor, DAccessor, FAccessor);
    Utwothirds = build_Utwothirds(UAccessor, UnewAccessor, DAccessor, FAccessor);
    Uone = build_Uone(UAccessor, UnewAccessor, DAccessor, FAccessor);

    ctoprim.compile_jit();
    courno_func.compile_jit();
    diffterm.compile_jit();
    hypterm.compile_jit();
    Uonethird.compile_jit();
    Utwothirds.compile_jit();
    Uone.compile_jit();
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

void advance(DistributedImage<double> &U, DistributedImage<double> &Unew,
             DistributedImage<double> &Q,
             DistributedImage<double> &D, DistributedImage<double> &F,
             double &dt) {
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
    timestep.set(dt);

    if (parallel_IOProcessor()) {
        std::cout << std::scientific << "dt,courno " << std::setprecision(std::numeric_limits<double>::digits10) << dt << " " << courno << "\n";
    }

    // Calculate D at time N. (D = diffterm)
    diffterm.realize(D);
    // Calculate F at time N (N = hypterm)
    hypterm.realize(F);

    // Calculate U at time N+1/3
    Uonethird.realize(Unew);
    // Swap so that ctoprim/hypterm read from Unew (U <- Unew)
    Up.set(Unew);
    ctoprim.realize(Q);
    // Calculate D at time N+1/3
    diffterm.realize(D);
    // Calculate F at time N+1/3
    hypterm.realize(F);

    // Swap back (U <- U)
    Up.set(U);
    // Calculate U at time N+2/3
    Utwothirds.realize(Unew);
    Up.set(Unew); // U <- Unew
    ctoprim.realize(Q);
    // Calculate D at time N+2/3
    diffterm.realize(D);
    // Calculate F at time N+2/3
    hypterm.realize(F);

    Up.set(U); // U <- U
    // Calculate U at time N+1
    Uone.realize(U);
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
    DistributedImage<double> U(global_w, global_h, global_d, nc), Unew(global_w, global_h, global_d, nc), Q(global_w, global_h, global_d, 6);
    DistributedImage<double> D(global_w, global_h, global_d, nc), F(global_w, global_h, global_d, nc);

    // Impose periodic boundary conditions on U and Q
    std::vector<std::pair<Expr, Expr>>
        global_bounds_U = {std::make_pair(0, global_w), std::make_pair(0, global_h),
                           std::make_pair(0, global_d), std::make_pair(0, nc)},
        global_bounds_Q = {std::make_pair(0, global_w), std::make_pair(0, global_h),
                           std::make_pair(0, global_d), std::make_pair(0, 6)};
    Func Ut, Unewt, Qt, Dt, Ft;
    Ut(x,y,z,c) = Up(x,y,z,c);
    Unewt(x,y,z,c) = Unewp(x,y,z,c);
    Qt(x,y,z,c) = Qp(x,y,z,c);
    Dt(x,y,z,c) = Dp(x,y,z,c);
    Ft(x,y,z,c) = Fp(x,y,z,c);
    Func UAccessor("UAccessor"), UnewAccessor("UnewAccessor"), QAccessor("QAccessor");
    Func DAccessor("DAccessor"), FAccessor("FAccessor");
    UAccessor = BoundaryConditions::repeat_image(Ut, global_bounds_U);
    UnewAccessor = BoundaryConditions::repeat_image(Unewt, global_bounds_U);
    QAccessor = BoundaryConditions::repeat_image(Qt, global_bounds_Q);
    DAccessor(x,y,z,c) = D(x,y,z,c);
    FAccessor(x,y,z,c) = F(x,y,z,c);
    build_pipeline(UAccessor, UnewAccessor, QAccessor, DAccessor, FAccessor);

    U.set_domain(x, y, z, c);
    U.allocate();
    Unew.set_domain(x, y, z, c);
    Unew.allocate();
    Q.set_domain(x, y, z, c);
    Q.allocate();
    D.set_domain(x, y, z, c);
    D.allocate();
    F.set_domain(x, y, z, c);
    F.allocate();

    Up.set(U);
    Unewp.set(Unew);
    Qp.set(Q);
    Dp.set(D);
    Fp.set(F);

    init_data(U);

    double time = 0, dt = 0;
    timestep.set(dt);
    for (int istep = 0; istep < nsteps; istep++) {
        if (parallel_IOProcessor()) {
            std::cout << "Advancing time step " << istep << ", time = " << time << "\n";
        }
        advance(U, Unew, Q, D, F, dt);
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
