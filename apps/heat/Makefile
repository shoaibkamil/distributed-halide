include ../support/Makefile.inc

WITH_MPI ?=
CXX := $(if $(WITH_MPI), mpicxx, $(CXX))
MPI_CXX_FLAGS=$(if $(WITH_MPI), -DWITH_MPI=1, )

all: distributed_heat

distributed_heat: heat_distributed.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) -O3 heat_distributed.cpp $(LIB_HALIDE) -o distributed_heat -ldl -lpthread -lz $(LDFLAGS)

clean:
	rm -f distributed_heat
