GENCODE_SM70 ?= -gencode=arch=compute_70,code=\"sm_70,compute_70\"

CPPSRC = ../../graph_parser/util.cpp ../../graph_parser/parse.cpp
CUSRC =
BASEEXE = PRvEN_INLINE
# CUDA source files (compiled with cudacc)
EXECUTABLE = ../../bin/$(BASEEXE)
#EXECUTABLE = a.out
CUSRC += pagerank.cu
OPTS = -O3
CUOPTS = $(GENCODE_SM70) 
INC = -I$(NVIDIA_COMPUTE_SDK_LOCATION)/../4.2/C/common/inc -I$(CUDA_INSTALL_PATH)/include
LIBS = -L$(NVIDIA_COMPUTE_SDK_LOCATION)/../4.2/C/lib -lcudart

OBJS = $(notdir $(CPPSRC:.cpp=.o))

all: $(OBJS)
	nvcc $(NVOPTS) $(OPTS) $(CUOPTS) $(CUSRC) *.o $(INC) -o $(EXECUTABLE) $(LIBS)

$(OBJS):
	g++ -c $(CPPSRC) $(OPTS)

clean:
	rm -f *.o *.cu.cpp $(EXECUTABLE)

cleancucpp:
	rm -f *.cu.cpp
