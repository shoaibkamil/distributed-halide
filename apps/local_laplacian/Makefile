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

all: process

local_laplacian_gen: local_laplacian_gen.cpp
	$(CXX) $(CXXFLAGS) -I$(NFM_HEADER) local_laplacian_gen.cpp -g $(LIB_HALIDE) -L$(NFM_LIB) -lnfm -L$(ISL_LIB) -lisl -lgmp -o local_laplacian_gen -lpthread -ldl -ltinfo -lz \
	$(LDFLAGS) $(LLVM_SHARED_LIBS)

local_laplacian.o: local_laplacian_gen
	$(LD_PATH_SETUP) ./local_laplacian_gen

process: process.cpp local_laplacian.o
	$(CXX) $(CXXFLAGS) -Wall -O3 process.cpp local_laplacian.o -o process -lpthread -ldl -ltinfo $(PNGFLAGS) $(OPENGL_LDFLAGS)

out.png: process
	./process ../images/rgb.png 8 1 1 10 out.png

# Build rules for generating a visualization of the pipeline using HalideTraceViz
process_viz: local_laplacian_viz.o
	$(CXX) $(CXXFLAGS) -Wall -O3 process.cpp local_laplacian_viz.o -o process_viz -lpthread -ldl -ltinfo $(PNGFLAGS) $(CUDA_LDFLAGS) $(OPENCL_LDFLAGS) $(OPENGL_LDFLAGS)

local_laplacian_viz.o: local_laplacian_gen
	$(LD_PATH_SETUP) HL_TRACE=3 ./local_laplacian_gen 6
	mv local_laplacian.o local_laplacian_viz.o

local_laplacian.mp4: process_viz
	bash viz.sh

distributed_local_laplacian: ../../ local_laplacian_distributed.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) -I$(NFM_HEADER)  -O3 local_laplacian_distributed.cpp -g $(LIB_HALIDE) -L$(NFM_LIB) -lnfm -L$(ISL_LIB) -lisl -lgmp -o distributed_local_laplacian -lpthread -ldl -ltinfo -lz \
	$(LDFLAGS) $(LLVM_SHARED_LIBS)

clean:
	rm -f process local_laplacian.o process_viz local_laplacian_viz.o local_laplacian_gen local_laplacian.mp4 distributed_local_laplacian
