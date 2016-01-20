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


BUILD_DIR = build_make

# If HL_TARGET isn't set, use host
HL_TARGET ?= host

all: $(BUILD_DIR)/wavelet

distributed_wavelet: wavelet_distributed.cpp
	$(CXX) $(CXXFLAGS) -I$(NFM_HEADER) wavelet_distributed.cpp -g $(LIB_HALIDE) -L$(NFM_LIB) -lnfm -L$(ISL_LIB) -lisl -lgmp -ltinfo -o distributed_wavelet -lpthread -ldl -lz $(LDFLAGS)

clean:
	@rm -rf $(BUILD_DIR)

# By default, %.generator is produced by building %_generator.cpp
$(BUILD_DIR)/%.generator: %_generator.cpp $(GENERATOR_DEPS)
	@echo Building Generator $(filter %_generator.cpp,$^)
	@mkdir -p $(BUILD_DIR)
	@$(CXX) $(CXXFLAGS) -I$(NFM_HEADER) -fno-rtti $(filter-out %.h,$^) -L$(NFM_LIB) -lnfm -L$(ISL_LIB) -lisl -lgmp -lz -ldl -ltinfo -lpthread -o $@

# By default, %.o/.h are produced by executing %.generator
$(BUILD_DIR)/%.o $(BUILD_DIR)/%.h: $(BUILD_DIR)/%.generator
	@echo Running Generator $<
	@mkdir -p $(BUILD_DIR)
	$(LD_PATH_SETUP) $< -g $(notdir $*) -o $(BUILD_DIR) target=$(HL_TARGET)

HL_MODULES = \
	$(BUILD_DIR)/daubechies_x.o \
	$(BUILD_DIR)/haar_x.o \
	$(BUILD_DIR)/inverse_daubechies_x.o \
	$(BUILD_DIR)/inverse_haar_x.o

$(BUILD_DIR)/wavelet.o: wavelet.cpp $(HL_MODULES)
	@$(CXX) $(CXXFLAGS) $(LIBPNG_CXX_FLAGS) -I$(BUILD_DIR) -I$(NFM_HEADER) -c $< -o $@

$(BUILD_DIR)/wavelet: $(BUILD_DIR)/wavelet.o
	@$(CXX) $(CXXFLAGS) -I$(NFM_HEADER) $^ $(HL_MODULES) $(PNGFLAGS) -L$(NFM_LIB) -lnfm -L$(ISL_LIB) -lisl -lgmp -ldl -ltinfo -lpthread -o $@

test: $(BUILD_DIR)/wavelet
	@echo Testing wavelet...
	@$< ../images/gray.png $(BUILD_DIR)

# Don't auto-delete the generators.
.SECONDARY:
