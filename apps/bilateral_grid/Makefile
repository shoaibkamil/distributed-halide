include ../support/Makefile.inc

WITH_MPI ?=
CXX := $(if $(WITH_MPI), mpicxx, $(CXX))
MPI_CXX_FLAGS=$(if $(WITH_MPI), -DWITH_MPI=1, )

NFM_HEADER := ../../non-linear-FM/src/
NFM_LIB := ../../non-linear-FM/bin/
ISL_HEADER := ../../non-linear-FM/isl-0.15/release/include
ISL_LIB := ../../non-linear-FM/isl-0.15/release/lib

all: filter

bilateral_grid: bilateral_grid.cpp
	$(CXX) $(CXXFLAGS) -I$(NFM_HEADER) bilateral_grid.cpp -g $(LIB_HALIDE) -L$(NFM_LIB) -lnfm -L$(ISL_LIB) -lisl -lgmp -o bilateral_grid -lpthread -ldl -lz $(LDFLAGS) \
	$(LLVM_SHARED_LIBS)

bilateral_grid.o: bilateral_grid
	./bilateral_grid 8

filter: bilateral_grid.o filter.cpp
	$(CXX) $(CXXFLAGS) -O3 -ffast-math -Wall -Werror filter.cpp bilateral_grid.o -lpthread -ldl -o filter  $(PNGFLAGS)

bilateral_grid.mp4: bilateral_grid.cpp viz.sh
	bash viz.sh

out.png: filter
	./filter ../images/gray.png out.png 0.1 10

distributed_bilateral_grid: bilateral_grid_distributed.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) -I$(NFM_HEADER) -O3 bilateral_grid_distributed.cpp $(LIB_HALIDE) -L$(NFM_LIB) -lnfm -L$(ISL_LIB) -lisl -lgmp -o distributed_bilateral_grid -ldl -lpthread -lz $(LDFLAGS)

clean:
	rm -f bilateral_grid bilateral_grid.mp4 bilateral_grid.o bilateral_grid.h bilateral filter distributed_bilateral_grid
