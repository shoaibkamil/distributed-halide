#include "Halide.h"
#include <iomanip>
#include <iostream>
#include <fstream>

/*
To check: 

$ for i in `seq 0 9`; do ./check.py floattest/f.iter${i}.dat floattest/f.iter${i}-f90.dat; done
 */

using namespace Halide;

const double pi = 3.141592653589793238462643383279502884197;
int rank, numprocs;

void print_img(const std::string &filename, Image<double> &img) {
    std::ofstream of(filename);
    of << std::scientific << std::setprecision(std::numeric_limits<double>::digits10);
    for (int x = 0; x < img.extent(0); x++) {
        of << img(x) << " ";
    }
    of << "\n";
    of.close();
}

int main(int argc, char **argv) {
    int req = MPI_THREAD_MULTIPLE, prov;
    MPI_Init_thread(&argc, &argv, req, &prov);
    assert(prov == req);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    Func f;
    Var x;

    int iters = 10;
    const int w = 10;
    Image<double> data(w), result(w);

    for (int x = 0; x < data.width(); x++) {
        data(x) = pi/(1+x);
    }

    // std::cout << std::scientific << std::setprecision(std::numeric_limits<double>::digits10);
    // std::cout << "C++    pi/1 " << pi/1 << "\n";
    // std::cout << "Halide pi/1 " << data(0) << "\n";
    // return 0;
    
    print_img("floattest/initialcond.dat", data);
    
    f(x) = sin((data(x) + data(x) + data(x)) + (data(x) + data(x) + data(x)));

    for (int i = 0; i < iters; i++) {
        f.realize(result);
        std::string file = "floattest/f.iter" + std::to_string(i) + ".dat";
        print_img(file, result);
        for (int x = 0; x < data.width(); x++) {
            data(x) = result(x);
        }
    }
    
    
    return 0;
}
