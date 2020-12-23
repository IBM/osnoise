# osnoise
Utility to characterize parallel application scaling issues caused by effects including OS noise.

Quick start :

  (1) edit "makefile" and set the MPI C compiler and options

  (2) make

  (3) mpirun --bind-to core -np 1024  ./osnoise -c 3.0 -t 300 -x 100 -n 31

  (4) inspect output, including the reported parallel efficiency

Many factors can impact scaling of MPI parallel applications, including the
effective latency and/or bandwidth of the messaging layers, possible interference
from daemon activity, congestion in the network, and actual variations in computation
or communication speeds due to any combination of hardware and software effects.
We have found that a simple synthetic benchmark can be very useful when it comes to
identifying issues that impact scaling for a given system.  There is a long history
of similar work, including benchmarks such as P-SNAP (Performance and Architecture
Laboratory System Noise Activity Program, from Los Alamos National Laboratory),
fixed work-quantum (FWQ) and fixed time-quantum (FTW) benchmarks (used for example
by Lawrence Livermore National Laboratory as Sequoia benchmarks), and Netguage from
ETH, for operating system noise measurement.  The current project was developed
independently.  It uses a calibrated sequence of steps, where each step consists of
a compute phase followed by MPI communication.  The default communication pattern
is nearest-neighbor boundary exchange on a 2D Cartesian process grid.  One can
optionally choose a globally synchronizing MPI call, MPI_Allreduce or MPI_Barrier,
after every computation step.  Message sizes can be specified at run time via command
line arguments.  The use of MPI at each step ensures that this benchmark is very
similar to many real-world simulations.  The MPI connections make the benchmark
very sensitive to daemon activity or to any source of inhomogeneity.  If any MPI
rank is slowed down at any stage of a given step, other MPI ranks will be impacted.
The resulting delays will be local for a nearest-neighbor communication pattern,
or global for a blocking collective communication pattern.

To build the executable, edit the makefile and set CC (normally mpicc) and optionally
set compiler options.  Then "make" should build the executable.  It is recommended to
use GNU compilers with an architecture flag that will generate hardware square-root
instructions if they are available  (for example, -march=broadwell, -march=znver2,
or -mcpu=power9, etc.).

Typical use of this benchmark would be to scan over a range of parameters, and map
out the parallel efficiency and the distribution of timing variations.  The scan
can be at a fixed scale or number of nodes, where the computation interval covers
a range such as {1.0 msec, 3.0 msec, 10.0 msec, 30.0 msec, 60.0 msec, 100.0 msec},
or the scan can be over the number of nodes for a fixed computation interval. This
second type of scan is a traditional "weak scaling" measurement.  Roughly speaking,
a nominal computation interval of ~1.0 msec corresponds to a rather fine-grained
parallel application : there will be either local or global communication at ~1.0
msec intervals.  Not all parallel applications exchange messages that frequently,
and it is often useful to map out the scaling behavior as a function of the 
computation interval.


It is often desirable to place one MPI rank on each core, and use options like these :

mpirun --bind-to core -np 1024  ./osnoise -c 3.0 -t 300 -x 100 -n 31

Options :

flag argument <br />
 -c  float : specifies the compute interval in msec (3.0 msec above) <br />
 -t  int   : specifies the target measurement time in seconds (300 sec above) <br />
 -x  int   : specifies the message size for exchange (100 bytes above) <br />
 -n  int   : specifies the number of histogram bins (31 bins above) <br />
 -m  char  : specifies the communication method (-m [exchange, allreduce, barrier]) <br />
 -k  char  : specifies the compute kernel (-k [sqrt, lut]) <br />
 -d        : requests a dump of all step times <br />
 -b        : requests an added barrier every 100 steps <br />
 -h        : prints a short help message <br />

With a 3.0 msec time interval for computation, and 100 byte message for exchange,
the elapsed time should be totally dominated by computation, and the measurement
should complete in very close to the specified target time.  Daemon activity and
other sources of OS noise will result in a longer than expected overall elapsed time.
Generally speaking, sensitivity to daemon activity is highest when all cores are
used for computation, and when the time interval between communication events is in
the ~10 msec range or lower.  

The default compute kernel computes the square-root of a pseudo-random number.  One
can optionally request a compute kernel that uses a look-up table to compute an
approximation to the exp() function (command-line option -k lut).  These two compute
kernels may behave differently.  The look-up table kernel is more sensitive to use
of caches, and may exhibit variable performance due to address-space layout 
randomization.  In a large-scale parallel job, there may be cache-line aliasing
effects caused by randomized address assignments, and that can result in variations
in compute performance even though the amount of work is identical across ranks.
On most systems, the Linux kernel uses address-space randomization by default, but
one can launch the job with a helper that disables this feature, see below.  The
default compute kernel is less sensitive to cache and address assignments.

The code prints a lot of information, including a histogram of all step times, where
one step represents completion of one pair of {compute, communicate} stages on any
MPI rank.  Ideally, this histogram should resemble a delta-function, with a peak
at very close to the time interval specified for the "compute" function.  The code
prints the ratio of the measured elapsed time to the expected time for an ideal system;
and that ratio provides a measure of the effective parallel efficiency.  The key
output items are the overall parallel efficiency, and the histogram of step times.

A summary is printed at the end with the host and cpu assignments for each MPI rank,
along with some cumulative data including the average compute time in msec, the
percent relative deviation, the accumulated time in MPI, and the number of involuntary
context switches.  This information has often been useful for identifying problems
with a specific host or cpu.  For example, if a certain cpu is not performing at the
typical level, all of the other MPI ranks will wait on the slow one; thus the slow
rank will have the minimum cumulative time in MPI.  If the computation rate is truly
uniform over all MPI ranks, there should be a small spread in cumulative MPI times.

xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

As mentioned above, some compute kernels, including the lookup table kernel, are
sensitive to caches, and may show variable performance when Linux uses address space
layout randomization (ASLR).  One can launch the job with a helper process that uses
the Linux personality() routine to disable ASLR and then exec the real program, which
will inherit the personality setting.  Note that it is not possible to directly set
the ALSR behavior in the parent process, because that has already been set by Linux.
An example of a launcher that disables ASLR is shown below:

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/personality.h>

int main(int argc, char * argv[])
{
  int rc;

  rc = personality(ADDR_NO_RANDOMIZE);
  if (rc == -1) fprintf(stderr, "personality failed ... not disabling ASLR\n");

  rc = execvp(argv[1], &argv[1]);
  if (rc == -1) {
    fprintf(stderr, "execvp failed for %s ... exiting\n", argv[1]);
    exit(0);
  }
  return 0;
}

Use of such a launcher would be : mpirun ... -np 128 ./launcher  ./your.exe  [program args] .

xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

It can be useful to look at the sequence of measured compute intervals for a given rank
or a set of ranks.  You can dump a binary file that contains all samples (-d option),
and then extract the compute times or the step times for a specified MPI rank.  The time
spent in communication routines = (step - compute).  Utilities to extract data from the
binary file are provided : extract_comp.c  and  extract_step.c .
