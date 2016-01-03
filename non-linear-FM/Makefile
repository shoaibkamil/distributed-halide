CXX = g++ -std=c++11

THIS_MAKEFILE = $(realpath $(filter %Makefile, $(MAKEFILE_LIST)))
ROOT_DIR = $(strip $(shell dirname $(THIS_MAKEFILE)))
SRC_DIR  = $(ROOT_DIR)/src

BIN_DIR     = bin
BUILD_DIR   = $(BIN_DIR)/build
TMP_DIR     = $(BUILD_DIR)/tmp

ISL_HEADER := $(ROOT_DIR)/isl-0.15/release/include
ISL_LIB := $(ROOT_DIR)/isl-0.15/release/lib
CXXLAGS = -Wall -fPIC -DNDEBUG

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	ISL_HEADER := $(ROOT_DIR)/isl-0.15/debug/include
	ISL_LIB := $(ROOT_DIR)/isl-0.15/debug/lib
	CPPFLAGS = -DDEBUG
	CXXLAGS = -Wall -fPIC -g -O0
endif

LD_PATH_SETUP = LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(CURDIR)/$(BIN_DIR):$(ISL_LIB)

SOURCE_FILES = \
    nfm_constraint.cpp \
	nfm_context.cpp \
	nfm_context_domain.cpp \
	nfm_domain.cpp \
	nfm_isl_interface.cpp \
	nfm_polynom.cpp \
	nfm_solver.cpp \
	nfm_space.cpp

HEADER_FILES = \
	nfm_constraint.h \
	nfm_context.h \
	nfm_context_domain.h \
	nfm_debug.h \
	nfm_domain.h \
	nfm_isl_interface.h \
	nfm_polynom.h \
	nfm_solver.h \
	nfm_space.h

OBJECTS = $(SOURCE_FILES:%.cpp=$(BUILD_DIR)/%.o)
HEADERS = $(HEADER_FILES:%.h=$(SRC_DIR)/%.h)

.PHONY: all
all: $(BIN_DIR)/libnfm.a $(BIN_DIR)/libnfm.so

$(BIN_DIR)/libnfm.a: $(OBJECTS)
	@-mkdir -p $(BIN_DIR)
	@rm -f $(BIN_DIR)/libnfm.a
	ar q $(BIN_DIR)/libnfm.a $(OBJECTS)
	ranlib $(BIN_DIR)/libnfm.a

$(BIN_DIR)/libnfm.so: $(BIN_DIR)/libnfm.a
	$(CXX) -shared $(OBJECTS) -o $(BIN_DIR)/libnfm.so -L$(ISL_LIB) -lisl -lgmp

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(SRC_DIR)/%.h
	@-mkdir -p $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXLAGS) -I$(ISL_HEADER) -c $< -o $@ -MMD -MP -MF $(BUILD_DIR)/$*.d -MT $(BUILD_DIR)/$*.o -L$(ISL_LIB) -lisl -lgmp

$(BIN_DIR)/test: $(ROOT_DIR)/test/nfm_test.cpp $(BIN_DIR)/libnfm.so
	$(CXX) $(CPPFLAGS) $(CXXLAGS) $< -I$(SRC_DIR) -I$(ISL_HEADER) -L$(BIN_DIR) -L$(ISL_LIB) -lnfm -lisl -o $@

test: $(BIN_DIR)/test
	@-mkdir -p $(TMP_DIR)
	cd $(TMP_DIR); $(LD_PATH_SETUP) $(CURDIR)/$<
	@-echo

.PHONY: clean
clean:
	rm -rf $(BIN_DIR)/*
	rm -rf $(BUILD_DIR)/*
	rm -rf $(TMP_DIR)/*
