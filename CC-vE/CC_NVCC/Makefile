GENCODE_SM70 ?= -gencode=arch=compute_70,code=\"sm_70,compute_70\"

CPPSRC = ../../graph_parser/util.cpp ../../graph_parser/parse.cpp
CUSRC =
BASEEXE = CCvE_NVCC
# CUDA source files (compiled with cudacc)
EXECUTABLE = ../../bin/$(BASEEXE)
CUSRC += cc.cu
OPTS = -O3
CUOPTS = $(GENCODE_SM70)
INC = -I$(NVIDIA_COMPUTE_SDK_LOCATION)/../4.2/C/common/inc -I$(CUDA_INSTALL_PATH)/include
LIBS = -L$(NVIDIA_COMPUTE_SDK_LOCATION)/../4.2/C/lib -lcudart

OBJS = $(notdir $(CPPSRC:.cpp=.o))

all: 
	nvcc $(NVOPTS) $(OPTS) $(CUOPTS) $(CUSRC)  $(INC) -o $(EXECUTABLE) $(LIBS)

debug: 
	nvcc --keep $(NVOPTS) $(OPTS) $(CUOPTS) $(CUSRC)  $(INC) -o $(EXECUTABLE) $(LIBS)

$(OBJS):
	g++ -c $(CPPSRC) $(OPTS)

clean:
	rm -f *.o *.cu.cpp $(EXECUTABLE)

cleanup:
	rm -f *.cu.cpp
	rm -f *.ii
	rm -f *.fatbin
	rm -f *.cui
	rm -f *.s
	rm -f *.c
	rm -f *.gpu
	rm -f *.cubin
	rm -f *.o
	rm -f *.bc
	rm -f *.cpp
	rm -f *.module_id
	rm -f *.ptx
	rm -f *.out
