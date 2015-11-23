include ../support/Makefile.inc

WITH_MPI ?=
CXX := $(if $(WITH_MPI), mpicxx, $(CXX))
MPI_CXX_FLAGS=$(if $(WITH_MPI), -DWITH_MPI=1, )

FORTRAN ?= mpif90

all: distributed_fluid

distributed_fluid: fluid_distributed.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) -O3 -march=corei7-avx -mtune=corei7-avx fluid_distributed.cpp $(LIB_HALIDE) -o distributed_fluid -ldl -lpthread -lz $(LDFLAGS)

floats: floating-test.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) -O3 -march=corei7-avx -mtune=corei7-avx floating-test.cpp $(LIB_HALIDE) -o floats -ldl -lpthread -lz $(LDFLAGS)

fortran_floats: floating-test.f90
	$(FORTRAN) -O3 -march=corei7-avx -mtune=corei7-avx floating-test.f90 -o fortran_floats

clean:
	rm -f distributed_fluid floats
