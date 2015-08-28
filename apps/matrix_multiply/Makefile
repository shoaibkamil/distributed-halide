include ../support/Makefile.inc

WITH_MPI ?=
CXX := $(if $(WITH_MPI), mpicxx, $(CXX))
MPI_CXX_FLAGS=$(if $(WITH_MPI), -DWITH_MPI=1, )

# g++ on OS X might actually be system clang without openmp
CXX_VERSION=$(shell $(CXX) --version)
ifeq (,$(findstring clang,$(CXX_VERSION)))
OPENMP_FLAGS=-fopenmp
else
OPENMP_FLAGS=
endif

distributed_matrix_multiply: ../../ matrix_multiply_distributed.cpp
	$(MAKE) -C ../../ $(LIB_HALIDE)
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) -O3 matrix_multiply_distributed.cpp ../../$(LIB_HALIDE) -o distributed_matrix_multiply -ldl -lpthread -lz $(LDFLAGS)

clean:
	rm -f test distributed_matrix_multiply
