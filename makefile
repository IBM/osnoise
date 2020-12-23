#==============================================================
# Select a suitable architecture flag, or provide your own
#==============================================================
#arch = -mcpu=power9
#arch = -march=broadwell
#arch = -march=skylake-avx512
#arch = -march=cascadelake
#arch = -march=znver2

MPCC = mpicc

all : osnoise lutexp.o launcher extract_comp extract_step


osnoise : osnoise.c lutexp.o
	$(MPCC) -g -O2 $(arch) osnoise.c -o osnoise lutexp.o -lm

lutexp.o : lutexp.c
	gcc -c -g -Ofast $(arch) lutexp.c

launcher : launcher.c
	gcc -g launcher.c -o launcher

extract_comp : extract_comp.c
	gcc -g -O2 extract_comp.c -o extract_comp

extract_step : extract_step.c
	gcc -g -O2 extract_step.c -o extract_step

clean :
	rm -f osnoise lutexp.o launcher extract_comp extract_step 
