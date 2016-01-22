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

# g++ on OS X might actually be system clang without openmp
CXX_VERSION=$(shell $(CXX) --version)
ifeq (,$(findstring clang,$(CXX_VERSION)))
OPENMP_FLAGS=-fopenmp
else
OPENMP_FLAGS=
endif

distributed_matrix_multiply: matrix_multiply_distributed.cpp
	#$(MAKE) -C $(LIB_HALIDE)
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) -I$(NFM_HEADER) -O3 matrix_multiply_distributed.cpp $(LIB_HALIDE) -L$(NFM_LIB) -lnfm -L$(ISL_LIB) -lisl -lgmp -o distributed_matrix_multiply -ldl -ltinfo -lpthread -lz $(LDFLAGS)

clean:
	rm -f test distributed_matrix_multiply

run: distributed_matrix_multiply
	$(LD_PATH_SETUP)  ./distributed_matrix_multiply 16 16