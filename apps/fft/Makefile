include ../support/Makefile.inc

WITH_MPI = 1
CXX := $(if $(WITH_MPI), mpicxx, $(CXX))

CXXFLAGS += -g -Wall

NFM_HEADER := ../../non-linear-FM/src/
NFM_LIB := ../../non-linear-FM/bin/
ISL_HEADER := ../../non-linear-FM/isl-0.15/release/include
ISL_LIB := ../../non-linear-FM/isl-0.15/release/lib

ifeq ($(UNAME), Darwin)
LD_PATH_SETUP = DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:$(NFM_LIB):$(ISL_LIB)
else
LD_PATH_SETUP = LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(NFM_LIB):$(ISL_LIB)
endif


.PHONY: clean

ifeq ($(WITH_FFTW),1)
CXXFLAGS += -DWITH_FFTW
LDFLAGS += -lfftw3f
endif

bench_fft: main.cpp fft.cpp fft.h complex.h funct.h
	$(CXX) $(CXXFLAGS) -I$(NFM_HEADER) main.cpp fft.cpp $(LIB_HALIDE) -L$(NFM_LIB) -lnfm -L$(ISL_LIB) -lisl -lgmp -o bench_fft -lpthread -ldl -ltinfo -lz \
	$(LDFLAGS) $(LLVM_SHARED_LIBS)

bench_16x16: bench_fft
	$(LD_PATH_SETUP) ./bench_fft 16 16

bench_32x32: bench_fft
	$(LD_PATH_SETUP) ./bench_fft 32 32

bench_48x48: bench_fft
	$(LD_PATH_SETUP) ./bench_fft 48 48

bench_64x64: bench_fft
	$(LD_PATH_SETUP) ./bench_fft 64 64

clean:
	rm -f bench_fft c2c.html r2c.html c2r.html

run: bench_16x16
