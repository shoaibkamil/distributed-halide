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

ifeq ($(HL_TARGET),ptx)
  SCHEDULE=100
else
  SCHEDULE=0
endif

all: process

camera_pipe: camera_pipe.cpp
	$(CXX) $(CXXFLAGS) -I$(NFM_HEADER) camera_pipe.cpp -g $(LIB_HALIDE) -L$(NFM_LIB) -lnfm -L$(ISL_LIB) -lisl -lgmp -ltinfo -o camera_pipe -ldl -lpthread -lz $(LDFLAGS)

curved.o: camera_pipe
	$(LD_PATH_SETUP) ./camera_pipe 8 0 # 8-bit output,

fcam/Demosaic.o: fcam/Demosaic.cpp fcam/Demosaic.h
	$(CXX) $(CXXFLAGS) -c -Wall -fopenmp -O3 $< -o $@

fcam/Demosaic_ARM.o: fcam/Demosaic_ARM.cpp fcam/Demosaic_ARM.h
	$(CXX) $(CXXFLAGS) -c -Wall -fopenmp -O3 $< -o $@

process: process.cpp curved.o fcam/Demosaic.o fcam/Demosaic_ARM.o
	$(CXX) $(CXXFLAGS) -Wall -O3 $^ -o $@ -lpthread -ldl -fopenmp $(PNGFLAGS)

out.png: process
	./process ../images/bayer_raw.png 3700 2.0 50 5 out.png

../../bin/HalideTraceViz:
	$(MAKE) -C ../../ bin/HalideTraceViz

camera_pipe.avi: camera_pipe.cpp viz.sh $(HALIDE_TRACE_VIZ) ../../bin/HalideTraceViz
	bash viz.sh

distributed_camera_pipe: camera_pipe_distributed.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) -I$(NFM_HEADER) -O3 camera_pipe_distributed.cpp $(LIB_HALIDE) -L$(NFM_LIB) -lnfm -L$(ISL_LIB) -lisl -lgmp -ltinfo -o distributed_camera_pipe -ldl -lpthread -lz $(LDFLAGS)

clean:
	rm -f out.png process curved.o camera_pipe fcam/*.o distributed_camera_pipe

run: camera_pipe
	$(LD_PATH_SETUP) ./camera_pipe 8 0 # 8-bit output,

distrib: distributed_camera_pipe
	$(LD_PATH_SETUP) ./distributed_camera_pipe 8 0 # 8-bit output