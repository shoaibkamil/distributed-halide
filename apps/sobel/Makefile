include ../support/Makefile.inc

WITH_MPI ?=
CXX := $(if $(WITH_MPI), mpicxx, $(CXX))
MPI_CXX_FLAGS=$(if $(WITH_MPI), -DWITH_MPI=1, )

all: distributed_sobel

distributed_sobel: sobel_distributed.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) -O3 sobel_distributed.cpp $(LIB_HALIDE) -o distributed_sobel -ldl -lpthread -lz $(LDFLAGS) $(PNGFLAGS)

clean:
	rm -f distributed_sobel
