osnoise.x : osnoise.o lutexp.o
	nvcc -ccbin=mpicxx -DGPU -O2 osnoise.o lutexp.o -o osnoise.x -lm -lnvidia-ml

osnoise.o : osnoise.cu
	nvcc -ccbin=mpicxx -c -dc -DGPU -O2 osnoise.cu

lutexp.o : lutexp.cu
	nvcc -c -dc -DGPU -O2 lutexp.cu

clean :
	rm -f *.o osnoise.x


