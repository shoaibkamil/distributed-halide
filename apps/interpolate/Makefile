include ../support/Makefile.inc

WITH_MPI ?=
CXX := $(if $(WITH_MPI), CC -dynamic, $(CXX))
MPI_CXX_FLAGS=$(if $(WITH_MPI), -DWITH_MPI=1, )

CXXFLAGS += -g -Wall

.PHONY: clean

interpolate: interpolate.cpp
	$(CXX) $(CXXFLAGS) interpolate.cpp $(LIB_HALIDE) -o interpolate -lpthread -ldl -lz \
	$(PNGFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS)

distributed_interpolate: interpolate_distributed.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) interpolate_distributed.cpp $(LIB_HALIDE) -o distributed_interpolate -lpthread -ldl -lz \
	$(PNGFLAGS) $(CUDA_LDFLAGS) $(OPENCL_LDFLAGS) $(OPENGL_LDFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS)

out.png: interpolate
	./interpolate ../images/rgba.png out.png

clean:
	rm -f interpolate interpolate.h out.png distributed_interpolate
