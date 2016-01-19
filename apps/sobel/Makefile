include ../support/Makefile.inc

WITH_MPI ?=
CXX := $(if $(WITH_MPI), mpicxx, $(CXX))
MPI_CXX_FLAGS=$(if $(WITH_MPI), -DWITH_MPI=1, )

MPIP_LINK_FLAGS := -L/data/scratch/tyler/install/lib -L/data/scratch/tyler/sw/mpiP-3.4.1/lib -lmpiP -lbfd -lm -lunwind
MPI_COMPILE_FLAGS := -I/usr/local/include
MPI_LINK_FLAGS := -L/usr/local/lib -lmpichf90 -Wl,-rpath -Wl,/usr/local/lib -lmpich -lpmi -lopa -lmpl

all: distributed_sobel

distributed_sobel: sobel_distributed.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) -O3 sobel_distributed.cpp $(LIB_HALIDE) -o distributed_sobel -ldl -lpthread -lz $(LDFLAGS) $(PNGFLAGS)

distributed_sobel_mpip: sobel_distributed.cpp
	g++ $(MPI_CXX_FLAGS) $(CXXFLAGS) -O3 sobel_distributed.cpp $(LIB_HALIDE) -o distributed_sobel_mpip -ldl -lpthread -lz $(LDFLAGS) $(PNGFLAGS) $(MPIP_LINK_FLAGS) $(MPI_COMPILE_FLAGS) $(MPI_LINK_FLAGS) -DUSE_MPIP=1

clean:
	rm -f distributed_sobel
