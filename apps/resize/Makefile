include ../support/Makefile.inc

WITH_MPI ?=
CXX := $(if $(WITH_MPI), mpicxx, $(CXX))
MPI_CXX_FLAGS=$(if $(WITH_MPI), -DWITH_MPI=1, )

MPIP_LINK_FLAGS := -L/data/scratch/tyler/install/lib -L/data/scratch/tyler/sw/mpiP-3.4.1/lib -lmpiP -lbfd -lm -lunwind
MPI_COMPILE_FLAGS := -I/usr/local/include
MPI_LINK_FLAGS := -L/usr/local/lib -lmpichf90 -Wl,-rpath -Wl,/usr/local/lib -lmpich -lpmi -lopa -lmpl

CXXFLAGS += -Wall -march=corei7-avx -mtune=corei7-avx

.PHONY: clean

resize: ../../ resize.cpp
	$(CXX) $(CXXFLAGS) resize.cpp $(LIB_HALIDE) -o resize -lpthread -ldl -lz $(PNGFLAGS) $(LDFLAGS)

distributed_resize: resize_distributed.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) resize_distributed.cpp $(LIB_HALIDE) -o distributed_resize -lpthread -ldl -lz $(PNGFLAGS) $(LDFLAGS)

distributed_resize_mpip: resize_distributed.cpp
	g++ $(MPI_CXX_FLAGS) $(CXXFLAGS) resize_distributed.cpp $(LIB_HALIDE) -o distributed_resize_mpip -lpthread -ldl -lz $(PNGFLAGS) $(LDFLAGS) $(MPIP_LINK_FLAGS) $(MPI_COMPILE_FLAGS) $(MPI_LINK_FLAGS) -DUSE_MPIP=1

distributed_downsample: downsample_distributed.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) downsample_distributed.cpp $(LIB_HALIDE) -o distributed_downsample -lpthread -ldl -lz $(PNGFLAGS) $(LDFLAGS)

distributed_upsample: upsample_distributed.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) upsample_distributed.cpp $(LIB_HALIDE) -o distributed_upsample -lpthread -ldl -lz $(PNGFLAGS) $(LDFLAGS)

out.png: resize
	./resize ../images/rgba.png out.png -f 2.0 -t cubic -s 3

clean:
	rm -f out.png resize
