# Target executable name:
EXE = gigacheck

##########################################################
## USER SPECIFIC DIRECTORIES ##
##########################################################

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda

##########################################################
## CC COMPILER OPTIONS ##
##########################################################

CC=g++
CPP_STD=20
CC_FLAGS=-std=c++${CPP_STD}
CC_LIBS=

##########################################################
## NVCC COMPILER OPTIONS ##
##########################################################

# GPU architecture

# mx130
SM_ARCH=50
# generic ptx
COMPUTE_ARCH=50

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS=-arch=compute_${COMPUTE_ARCH} -code=compute_${COMPUTE_ARCH},sm_${SM_ARCH} # compile to ptx and mx130
NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= $(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= $(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

##########################################################
## BUILD CONFIGS ##
##########################################################

# Default configuration (overridden with: make CONFIG=release)
CONFIG ?= debug

# Compiler flags for each configuration
ifeq ($(CONFIG), debug)
    CC_FLAGS += -g -O0 -DDEBUG
    NVCC_FLAGS += -G -O0 -g
else ifeq ($(CONFIG), release)
    CC_FLAGS += -O3 -DNDEBUG
    NVCC_FLAGS += -O3 
else
    $(error Unknown configuration "$(CONFIG)")
endif

##########################################################
## PROJECT DIR STRUCTURE ##
##########################################################

# Source file directory:
SRC_DIR = src

# Include header file diretory:
INC_DIR = include

# Build directory:
BUILD_DIR = build

# Object file directory:
OBJ_DIR = ${BUILD_DIR}/obj

# Target dir
TARGET_DIR = ${BUILD_DIR}/${CONFIG}

##########################################################
## FILES ##
##########################################################

# Include files

CPP_INC_FILES := $(shell find $(INC_DIR) -name '*.h')

CU_INC_FILES := $(shell find $(INC_DIR) -name '*.cuh')

INC_FILES := $(CPP_INC_FILES) $(CU_INC_FILES)

# Source files:

CPP_SRC_FILES := $(shell find $(SRC_DIR) -name '*.cpp')

CU_SRC_FILES := $(shell find $(SRC_DIR) -name '*.cu')

# Object files

CPP_OBJS := $(patsubst $(SRC_DIR)/%.cpp,${OBJ_DIR}/%.o,$(CPP_SRC_FILES)) 

CU_OBJS := $(patsubst $(SRC_DIR)/%.cu,${OBJ_DIR}/%.cuo,$(CU_SRC_FILES))

OBJS := $(CPP_OBJS) $(CU_OBJS) 

# Resources

$(info Includes)
$(info +    CPP  Headers: $(CPP_INC_FILES))
$(info +    CUDA Headers: $(CU_INC_FILES))
$(info Sources)
$(info +    CPP : $(CPP_SRC_FILES))
$(info +    CUDA: $(CU_SRC_FILES))
$(info OBJS)
$(info +    CPP : $(CPP_OBJS))
$(info +    CUDA: $(CU_OBJS))
$(info )


##########################################################
## PIPELINE ##
##########################################################

# Default target to build the executable
all: $(TARGET_DIR)/$(EXE)

# Link c++ and CUDA compiled object files to target executable:
${TARGET_DIR}/$(EXE) : $(OBJS) | $(TARGET_DIR)
	@echo "Linking $@"
	$(CC) $(CC_FLAGS) $(OBJS) -o $@ -L$(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)
	@echo "\n"

# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp $(INC_FILES) | $(OBJ_DIR)
	@mkdir -p $(@D)
	@echo "Compiling source $@"
	$(CC) $(CC_FLAGS) -I${INC_DIR} -I$(CUDA_INC_DIR) -c $< -o $@
	@echo "\n"

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.cuo : $(SRC_DIR)/%.cu $(CU_INC_FILES) | $(OBJ_DIR) 
	@mkdir -p $(@D)
	@echo "Compiling $@"
	$(NVCC) $(NVCC_FLAGS) -I$(INC_DIR) -I/usr/include -I$(CUDA_INC_DIR) -c $< -o $@ $(NVCC_LIBS)
	@echo "\n"

$(OBJ_DIR):
	mkdir -p $@
	@echo "\n"

$(TARGET_DIR):
	mkdir -p $@
	@echo "\n"

$(IMAGE_DIR):
	@mkdir -p $@
	@echo "\n"

##########################################################
## COMMANDS ##
##########################################################

clean:
	rm -rf $(BUILD_DIR)