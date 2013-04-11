GCC_OPT := --openmp
NVCC_OPT := --compiler-bindir=/usr/bin -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30
LIBS := -lgomp
INCLUDE_DIRS := -I/opt/build-essentials/include

all: release debug

release: maprecode

maprecode: src/CUDA_MapRecode.cu src/CUDAHelper.hpp
	nvcc ${NVCC_OPT} ${LIBS} ${INCLUDE_DIRS} -Xcompiler ${GCC_OPT} -O3 src/CUDA_MapRecode.cu -o maprecode
	
debug: src/CUDA_MapRecode.cu src/CUDAHelper.hpp
	nvcc ${NVCC_OPT} ${LIBS} ${INCLUDE_DIRS} -Xcompiler ${GCC_OPT} -O0 -g src/CUDA_MapRecode.cu -o maprecode_debug
clean:
	rm *.o maprecode maprecode_debug
