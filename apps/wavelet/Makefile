include ../support/Makefile.inc

WITH_MPI ?=
CXX := $(if $(WITH_MPI), mpicxx, $(CXX))
MPI_CXX_FLAGS=$(if $(WITH_MPI), -DWITH_MPI=1, )

MPIP_LINK_FLAGS := -L/data/scratch/tyler/install/lib -L/data/scratch/tyler/sw/mpiP-3.4.1/lib -lmpiP -lbfd -lm -lunwind
MPI_COMPILE_FLAGS := -I/usr/local/include
MPI_LINK_FLAGS := -L/usr/local/lib -lmpichf90 -Wl,-rpath -Wl,/usr/local/lib -lmpich -lpmi -lopa -lmpl

BUILD_DIR = build_make

# If HL_TARGET isn't set, use host
HL_TARGET ?= host

all: $(BUILD_DIR)/wavelet

distributed_wavelet: wavelet_distributed.cpp
	$(CXX) $(CXXFLAGS) wavelet_distributed.cpp -g $(LIB_HALIDE) -o distributed_wavelet -lpthread -ldl -lz $(LDFLAGS)

distributed_wavelet_mpip: wavelet_distributed.cpp
	g++ $(CXXFLAGS) wavelet_distributed.cpp -g $(LIB_HALIDE) -o distributed_wavelet_mpip -lpthread -ldl -lz $(LDFLAGS) $(MPIP_LINK_FLAGS) $(MPI_COMPILE_FLAGS) $(MPI_LINK_FLAGS) -DUSE_MPIP=1

clean:
	@rm -rf $(BUILD_DIR)

# By default, %.generator is produced by building %_generator.cpp
$(BUILD_DIR)/%.generator: %_generator.cpp $(GENERATOR_DEPS)
	@echo Building Generator $(filter %_generator.cpp,$^)
	@mkdir -p $(BUILD_DIR)
	@$(CXX) $(CXXFLAGS) -fno-rtti $(filter-out %.h,$^) -lz -ldl -lpthread -o $@

# By default, %.o/.h are produced by executing %.generator
$(BUILD_DIR)/%.o $(BUILD_DIR)/%.h: $(BUILD_DIR)/%.generator
	@echo Running Generator $<
	@mkdir -p $(BUILD_DIR)
	@$< -g $(notdir $*) -o $(BUILD_DIR) target=$(HL_TARGET)

HL_MODULES = \
	$(BUILD_DIR)/daubechies_x.o \
	$(BUILD_DIR)/haar_x.o \
	$(BUILD_DIR)/inverse_daubechies_x.o \
	$(BUILD_DIR)/inverse_haar_x.o

$(BUILD_DIR)/wavelet.o: wavelet.cpp $(HL_MODULES)
	@$(CXX) $(CXXFLAGS) $(LIBPNG_CXX_FLAGS) -I$(BUILD_DIR) -c $< -o $@

$(BUILD_DIR)/wavelet: $(BUILD_DIR)/wavelet.o
	@$(CXX) $(CXXFLAGS) $^ $(HL_MODULES) $(PNGFLAGS) -ldl -lpthread -o $@

test: $(BUILD_DIR)/wavelet
	@echo Testing wavelet...
	@$< ../images/gray.png $(BUILD_DIR)

# Don't auto-delete the generators.
.SECONDARY:
