include ../support/Makefile.inc

WITH_MPI ?=
CXX := $(if $(WITH_MPI), mpicxx, $(CXX))
MPI_CXX_FLAGS=$(if $(WITH_MPI), -DWITH_MPI=1, )

CXXFLAGS += -g -Wall

.PHONY: clean

resize: ../../ resize.cpp
	$(CXX) $(CXXFLAGS) resize.cpp $(LIB_HALIDE) -o resize -lpthread -ldl -lz $(PNGFLAGS) $(LDFLAGS)

distributed_resize: resize_distributed.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) resize_distributed.cpp $(LIB_HALIDE) -o distributed_resize -lpthread -ldl -lz $(PNGFLAGS) $(LDFLAGS)

distributed_downsample: downsample_distributed.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) downsample_distributed.cpp $(LIB_HALIDE) -o distributed_downsample -lpthread -ldl -lz $(PNGFLAGS) $(LDFLAGS)

distributed_upsample: upsample_distributed.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) upsample_distributed.cpp $(LIB_HALIDE) -o distributed_upsample -lpthread -ldl -lz $(PNGFLAGS) $(LDFLAGS)

out.png: resize
	./resize ../images/rgba.png out.png -f 2.0 -t cubic -s 3

clean:
	rm -f out.png resize
