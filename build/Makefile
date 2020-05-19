TARGET  	= CUDNN_TEST
CC 			= g++

CODE_DIR 	= ../
LIB_DIR		= ../lib

SRC_DIR 	= $(CODE_DIR)/src
INC_DIR 	= $(CODE_DIR)/include

CUDA_DIR 	= /usr/local/cuda-10.0
NVCC		= $(CUDA_DIR)/bin/nvcc
CUDA_INC_DIR = $(CUDA_DIR)/include
CUDA_LIB_DIR = $(CUDA_DIR)/lib64

CUDNN_DIR   = /cuda
CUDNN_INC_DIR = ${CUDNN_DIR}/include
CUDNN_LIB_DIR = ${CUDNN_DIR}/lib64

vpath  		%.cpp $(SRC_DIR)
vpath  		%.cu  $(SRC_DIR)
vpath		%.h	  $(INC_DIR)

IFLAGS      = -I$(INC_DIR)\
	          -I$(CUDA_INC_DIR)\
			  -I$(CUDNN_INC_DIR)

OPT_FLAGS 	= -O3
ifeq ($(DEBUG), 1)
OPT_FLAGS 	= -g 
endif
CFLAGS  	= $(IFLAGS) -fPIC -Wall $(OPT_FLAGS) -std=c++11
NVCCFLAGS 	= $(IFLAGS) -Xcompiler -fPIC $(OPT_FLAGS) -std=c++11

ifeq ($(NVPROFILE), 1)
CFLAGS += -DNVPROFILE
endif

LDFLAGS     =  -L$(CUDA_LIB_DIR) -L$(CUDNN_LIB_DIR) -lcudart -lcudnn 

SRCS 		= $(wildcard $(SRC_DIR)/*.cpp)
CU_SRCS     = $(wildcard $(SRC_DIR)/*.cu)
INCS		= $(wildcard $(INC_DIR)/*.h)

OBJS		= $(notdir $(patsubst %.cpp, %.o, $(SRCS)))
OBJS		+= $(notdir $(patsubst %.cu, %.o, $(CU_SRCS)))

$(TARGET):	$(OBJS)
	$(CC) -o $(TARGET) $^ $(LDFLAGS)

%.o: %.cu $(INCS)
	$(NVCC) -c $(NVCCFLAGS) $< 

%.o: %.cpp $(INCS)
	$(CC) -c $(CFLAGS) $< 

clean:
	-rm $(TARGET)
	-rm $(OBJS)
