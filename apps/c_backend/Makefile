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


pipeline: pipeline.cpp
	$(CXX) $(CXXFLAGS) -I$(NFM_HEADER) -Wall pipeline.cpp $(LDFLAGS) $(LIB_HALIDE) -L$(NFM_LIB) -lnfm -L$(ISL_LIB) -lisl -lgmp -o pipeline -lpthread -ldl -ltinfo -lz -g

pipeline_c.cpp: pipeline
	$(LD_PATH_SETUP) ./pipeline

pipeline_c.h: pipeline
	$(LD_PATH_SETUP) ./pipeline

pipeline_native.h: pipeline
	$(LD_PATH_SETUP) ./pipeline

pipeline_native.o: pipeline
	$(LD_PATH_SETUP) ./pipeline

run: run.cpp pipeline_native.h pipeline_c.cpp
	$(CXX) $(CXXFLAGS) -Wall run.cpp pipeline_c.cpp pipeline_native.o -lpthread -ldl -ltinfo -o run

test: run
	./run

clean:
	rm -f run pipeline_native.{h,o} pipeline_c.{cpp,h} pipeline
