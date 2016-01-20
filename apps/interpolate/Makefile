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

CXXFLAGS += -g -Wall

.PHONY: clean

interpolate: interpolate.cpp
	$(CXX) $(CXXFLAGS) -I$(NFM_HEADER) interpolate.cpp $(LIB_HALIDE) -L$(NFM_LIB) -lnfm -L$(ISL_LIB) -lisl -lgmp -o interpolate -lpthread -ldl -ltinfo -lz \
	$(PNGFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS)

distributed_interpolate: interpolate_distributed.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) -I$(NFM_HEADER) interpolate_distributed.cpp $(LIB_HALIDE) -L$(NFM_LIB) -lnfm -L$(ISL_LIB) -lisl -lgmp -o distributed_interpolate -lpthread -ldl -ltinfo -lz \
	$(PNGFLAGS) $(CUDA_LDFLAGS) $(OPENCL_LDFLAGS) $(OPENGL_LDFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS)

out.png: interpolate
	$(LD_PATH_SETUP) ./interpolate ../images/rgba.png out.png

clean:
	rm -f interpolate interpolate.h out.png distributed_interpolate

run: interpolate
	$(LD_PATH_SETUP) ./interpolate ../images/rgba.png out.png

distrib: distributed_interpolate
	$(LD_PATH_SETUP) ./distributed_interpolate 8 8