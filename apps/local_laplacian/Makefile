include ../support/Makefile.inc

WITH_MPI ?=
CXX := $(if $(WITH_MPI), mpicxx, $(CXX))
MPI_CXX_FLAGS=$(if $(WITH_MPI), -DWITH_MPI=1, )

MPIP_LINK_FLAGS := -L/data/scratch/tyler/install/lib -L/data/scratch/tyler/sw/mpiP-3.4.1/lib -lmpiP -lbfd -lm -lunwind
MPI_COMPILE_FLAGS := -I/usr/local/include
MPI_LINK_FLAGS := -L/usr/local/lib -lmpichf90 -Wl,-rpath -Wl,/usr/local/lib -lmpich -lpmi -lopa -lmpl

all: process

local_laplacian_gen: local_laplacian_gen.cpp
	$(CXX) $(CXXFLAGS) local_laplacian_gen.cpp -g $(LIB_HALIDE) -o local_laplacian_gen -lpthread -ldl -lz \
	$(LDFLAGS) $(LLVM_SHARED_LIBS)

local_laplacian.o: local_laplacian_gen
	./local_laplacian_gen

process: process.cpp local_laplacian.o
	$(CXX) $(CXXFLAGS) -Wall -O3 process.cpp local_laplacian.o -o process -lpthread -ldl $(PNGFLAGS) $(OPENGL_LDFLAGS)

out.png: process
	./process ../images/rgb.png 8 1 1 10 out.png

# Build rules for generating a visualization of the pipeline using HalideTraceViz
process_viz: local_laplacian_viz.o
	$(CXX) $(CXXFLAGS) -Wall -O3 process.cpp local_laplacian_viz.o -o process_viz -lpthread -ldl $(PNGFLAGS) $(CUDA_LDFLAGS) $(OPENCL_LDFLAGS) $(OPENGL_LDFLAGS)

local_laplacian_viz.o: local_laplacian_gen
	HL_TRACE=3 ./local_laplacian_gen 6
	mv local_laplacian.o local_laplacian_viz.o

local_laplacian.mp4: process_viz
	bash viz.sh

distributed_local_laplacian: ../../ local_laplacian_distributed.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) -O3 local_laplacian_distributed.cpp -g $(LIB_HALIDE) -o distributed_local_laplacian -lpthread -ldl -lz $(LDFLAGS) $(LLVM_SHARED_LIBS)

distributed_local_laplacian_mpip: ../../ local_laplacian_distributed.cpp
	g++ $(MPI_CXX_FLAGS) $(CXXFLAGS) -O3 local_laplacian_distributed.cpp -g $(LIB_HALIDE) -o distributed_local_laplacian_mpip -lpthread -ldl -lz $(LDFLAGS) $(LLVM_SHARED_LIBS) $(MPIP_LINK_FLAGS) $(MPI_COMPILE_FLAGS) $(MPI_LINK_FLAGS) -DUSE_MPIP=1

clean:
	rm -f process local_laplacian.o process_viz local_laplacian_viz.o local_laplacian_gen local_laplacian.mp4 distributed_local_laplacian
