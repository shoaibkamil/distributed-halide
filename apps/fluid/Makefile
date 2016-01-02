include ../support/Makefile.inc

WITH_MPI ?=
CXX := $(if $(WITH_MPI), mpicxx, $(CXX))
MPI_CXX_FLAGS=$(if $(WITH_MPI), -DWITH_MPI=1, )

FORTRAN ?= mpif90

#CXX = g++

MPIP_LINK_FLAGS := -L/data/scratch/tyler/install/lib -L/data/scratch/tyler/sw/mpiP-3.4.1/lib -lmpiP -lbfd -lm -lunwind
MPI_COMPILE_FLAGS := -I/usr/local/include
MPI_LINK_FLAGS := -L/usr/local/lib -lmpichf90 -Wl,-rpath -Wl,/usr/local/lib -lmpich -lpmi -lopa -lmpl

all: distributed_fluid

distributed_fluid: fluid_distributed.cpp
#$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) -fopenmp -O3 -march=corei7-avx -mtune=corei7-avx fluid_distributed.cpp $(LIB_HALIDE) -o distributed_fluid -ldl -lpthread -lz $(LDFLAGS) $(MPIP_LINK_FLAGS) $(MPI_COMPILE_FLAGS) $(MPI_LINK_FLAGS)
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) -fopenmp -O3 -march=corei7-avx -mtune=corei7-avx fluid_distributed.cpp $(LIB_HALIDE) -o distributed_fluid -ldl -lpthread -lz $(LDFLAGS)

floats: floating-test.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) -O3 -march=corei7-avx -mtune=corei7-avx floating-test.cpp $(LIB_HALIDE) -o floats -ldl -lpthread -lz $(LDFLAGS)

fortran_floats: floating-test.f90
	$(FORTRAN) -O3 -march=corei7-avx -mtune=corei7-avx floating-test.f90 -o fortran_floats

clean:
	rm -f distributed_fluid floats
