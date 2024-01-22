NCCL_HOME=/home/vpcuser/nccl/build
NCCL_INC=-I$(NCCL_HOME)/include
NCCL_LIB=-L$(NCCL_HOME)/lib -lnccl

GPU_ARCH = -arch=sm_80

osnoise.x : osnoise.o lutexp.o
	nvcc -ccbin=mpicxx $(GPU_ARCH) -O2 osnoise.o lutexp.o -o osnoise.x $(NCCL_LIB)  -lm -lnvidia-ml

osnoise.o : osnoise.cu
	nvcc -ccbin=mpicxx $(GPU_ARCH) -c $(NCCL_INC) -dc -O2 osnoise.cu

lutexp.o : lutexp.cu
	nvcc -c -dc -O2 $(GPU_ARCH) lutexp.cu

clean :
	rm -f *.o osnoise.x

