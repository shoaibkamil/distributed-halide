include ../support/Makefile.inc

WITH_MPI ?=
CXX := $(if $(WITH_MPI), mpicxx, $(CXX))
MPI_CXX_FLAGS=$(if $(WITH_MPI), -DWITH_MPI=1, )
MPI_RUN=$(if $(WITH_MPI), srun --exclusive -N$(MPI_NODES), )
MPI_CXX_FLAGS=$(if $(WITH_MPI), -DWITH_MPI=1, )

MPIP_LINK_FLAGS := -L/data/scratch/tyler/install/lib -L/data/scratch/tyler/sw/mpiP-3.4.1/lib -lmpiP -lbfd -lm -lunwind
MPI_COMPILE_FLAGS := -I/usr/local/include
MPI_LINK_FLAGS := -L/usr/local/lib -lmpichf90 -Wl,-rpath -Wl,/usr/local/lib -lmpich -lpmi -lopa -lmpl

all: test

halide_blur: halide_blur.cpp
	$(CXX) $(CXXFLAGS) halide_blur.cpp $(LIB_HALIDE) -o halide_blur -ldl -lpthread -lz $(LDFLAGS)

halide_blur.o: halide_blur
	./halide_blur

# g++ on OS X might actually be system clang without openmp
CXX_VERSION=$(shell $(CXX) --version)
ifeq (,$(findstring clang,$(CXX_VERSION)))
OPENMP_FLAGS=-fopenmp
else
OPENMP_FLAGS=
endif

# -O2 is faster than -O3 for this app (O3 unrolls too much)
test: test.cpp halide_blur.o
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -msse2 -Wall -O2 test.cpp halide_blur.o -o test -lpthread -ldl $(PNGFLAGS)

distributed_blur: halide_blur_distributed.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) -O3 halide_blur_distributed.cpp $(LIB_HALIDE) -o distributed_blur -ldl -lpthread -lz $(LDFLAGS)

distributed_blur_mpip: halide_blur_distributed.cpp
	g++ $(MPI_CXX_FLAGS) $(CXXFLAGS) -O3 $^ $(LIB_HALIDE) -o $@ -ldl -lpthread -lz $(LDFLAGS) $(MPIP_LINK_FLAGS) $(MPI_COMPILE_FLAGS) $(MPI_LINK_FLAGS) -DUSE_MPIP=1

nondistributed_blur: ../../ halide_blur_distributed.cpp
	$(MAKE) -C ../../ $(LIB_HALIDE)
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) -O3 halide_blur_distributed.cpp -DNON_DISTRIBUTED ../../$(LIB_HALIDE) -o nondistributed_blur -ldl -lpthread -lz $(LDFLAGS)

test_distributed: ../../ distributed_blur
	$(MPI_RUN) ./distributed_blur

clean:
	rm -f test halide_blur.o halide_blur distributed_blur
