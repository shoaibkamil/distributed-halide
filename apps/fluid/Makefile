include ../support/Makefile.inc

WITH_MPI = 1
CXX := $(if $(WITH_MPI), mpicxx, $(CXX))
MPI_CXX_FLAGS=$(if $(WITH_MPI), -DWITH_MPI=1, )

NFM_HEADER := ../../non-linear-FM/src/
NFM_LIB := ../../non-linear-FM/bin/
ISL_HEADER := ../../non-linear-FM/isl-0.15/release/include
ISL_LIB := ../../non-linear-FM/isl-0.15/release/lib

ifeq ($(UNAME), Darwin)
LD_PATH_SETUP = DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:$(NFM_LIB):$(ISL_LIB)
else
LD_PATH_SETUP = LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(NFM_LIB):$(ISL_LIB)
endif

FORTRAN ?= mpif90

all: distributed_fluid

distributed_fluid: fluid_distributed.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) -I$(NFM_HEADER) -fopenmp -O3 -march=corei7-avx -mtune=corei7-avx fluid_distributed.cpp $(LIB_HALIDE) -L$(NFM_LIB) -lnfm -L$(ISL_LIB) -lisl -lgmp -ltinfo -o distributed_fluid -ldl -lpthread -lz $(LDFLAGS)

floats: floating-test.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) -I$(NFM_HEADER) -O3 -march=corei7-avx -mtune=corei7-avx floating-test.cpp $(LIB_HALIDE) -L$(NFM_LIB) -lnfm -L$(ISL_LIB) -lisl -lgmp -ltinfo -o floats -ldl -lpthread -lz $(LDFLAGS)

fortran_floats: floating-test.f90
	$(FORTRAN) -O3 -march=corei7-avx -mtune=corei7-avx floating-test.f90 -o fortran_floats

clean:
	rm -f distributed_fluid floats
