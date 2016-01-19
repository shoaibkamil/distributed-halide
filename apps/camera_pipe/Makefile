include ../support/Makefile.inc

WITH_MPI ?=
CXX := $(if $(WITH_MPI), mpicxx, $(CXX))
MPI_CXX_FLAGS=$(if $(WITH_MPI), -DWITH_MPI=1, )

MPIP_LINK_FLAGS := -L/data/scratch/tyler/install/lib -L/data/scratch/tyler/sw/mpiP-3.4.1/lib -lmpiP -lbfd -lm -lunwind
MPI_COMPILE_FLAGS := -I/usr/local/include
MPI_LINK_FLAGS := -L/usr/local/lib -lmpichf90 -Wl,-rpath -Wl,/usr/local/lib -lmpich -lpmi -lopa -lmpl

ifeq ($(HL_TARGET),ptx)
  SCHEDULE=100
else
  SCHEDULE=0
endif

all: process

camera_pipe: ../../ camera_pipe.cpp
	$(MAKE) -C ../../ $(LIB_HALIDE)
	$(CXX) $(CXXFLAGS) camera_pipe.cpp -g ../../$(LIB_HALIDE) -o camera_pipe -ldl -lpthread -lz $(LDFLAGS)

curved.o: camera_pipe
	./camera_pipe 8 0 # 8-bit output,

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
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) -O3 camera_pipe_distributed.cpp $(LIB_HALIDE) -o distributed_camera_pipe -ldl -lpthread -lz $(LDFLAGS)

distributed_camera_pipe_mpip: camera_pipe_distributed.cpp
	g++ $(MPI_CXX_FLAGS) $(CXXFLAGS) -O3 camera_pipe_distributed.cpp $(LIB_HALIDE) -o distributed_camera_pipe_mpip -ldl -lpthread -lz $(LDFLAGS) $(MPIP_LINK_FLAGS) $(MPI_COMPILE_FLAGS) $(MPI_LINK_FLAGS) -DUSE_MPIP=1

clean:
	rm -f out.png process curved.o camera_pipe fcam/*.o distributed_camera_pipe
