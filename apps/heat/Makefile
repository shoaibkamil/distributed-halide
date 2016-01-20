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

all: distributed_heat

distributed_heat: heat_distributed.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) -I$(NFM_HEADER) -O3 heat_distributed.cpp $(LIB_HALIDE) -L$(NFM_LIB) -lnfm -L$(ISL_LIB) -lisl -lgmp -o distributed_heat -ldl -ltinfo -lpthread -lz $(LDFLAGS)

clean:
	rm -f distributed_heat

run:
	$(LD_PATH_SETUP) ./distributed_heat 8 8 8
