include ../support/Makefile.inc

WITH_MPI ?=
CXX := $(if $(WITH_MPI), mpicxx, $(CXX))
MPI_CXX_FLAGS=$(if $(WITH_MPI), -DWITH_MPI=1, )

MPIP_LINK_FLAGS := -L/data/scratch/tyler/install/lib -L/data/scratch/tyler/sw/mpiP-3.4.1/lib -lmpiP -lbfd -lm -lunwind
MPI_COMPILE_FLAGS := -I/usr/local/include
MPI_LINK_FLAGS := -L/usr/local/lib -lmpichf90 -Wl,-rpath -Wl,/usr/local/lib -lmpich -lpmi -lopa -lmpl

CXXFLAGS += -g -Wall

.PHONY: clean

interpolate: interpolate.cpp
	$(CXX) $(CXXFLAGS) interpolate.cpp $(LIB_HALIDE) -o interpolate -lpthread -ldl -lz \
	$(PNGFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS)

distributed_interpolate: interpolate_distributed.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) interpolate_distributed.cpp $(LIB_HALIDE) -o distributed_interpolate -lpthread -ldl -lz \
	$(PNGFLAGS) $(CUDA_LDFLAGS) $(OPENCL_LDFLAGS) $(OPENGL_LDFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS)

distributed_interpolate_mpip: interpolate_distributed.cpp
	g++ $(MPI_CXX_FLAGS) $(CXXFLAGS) interpolate_distributed.cpp $(LIB_HALIDE) -o distributed_interpolate_mpip -lpthread -ldl -lz $(MPIP_LINK_FLAGS) $(MPI_COMPILE_FLAGS) $(MPI_LINK_FLAGS) $(PNGFLAGS) $(CUDA_LDFLAGS) $(OPENCL_LDFLAGS) $(OPENGL_LDFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS) -DUSE_MPIP=1

out.png: interpolate
	./interpolate ../images/rgba.png out.png

clean:
	rm -f interpolate interpolate.h out.png distributed_interpolate
