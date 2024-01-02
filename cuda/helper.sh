#!/bin/bash
#--------------------------------------------------------------------------------
# mpirun -np $nmpi helper.sh your.exe [args]
# optionally set BIND_SLOTS in your job script = #hwthreads per rank
# Note : for some OpenMP implementations (GNU OpenMP) use mpirun --bind-to none
#--------------------------------------------------------------------------------

cpus_per_node=80
declare -a list=(`seq 0 79`)

if [ -z "$OMPI_COMM_WORLD_RANK" ]; then

  # for MPICH
  let local_size=$MPI_LOCALNRANKS
  let local_rank=$MPI_LOCALRANKID

else

  # for OpenMPI
  let local_size=$OMPI_COMM_WORLD_LOCAL_SIZE
  let local_rank=$OMPI_COMM_WORLD_LOCAL_RANK

fi

# divide available slots evenly or specify slots by env variable
if [ -z "$BIND_SLOTS" ]; then
  let cpus_per_rank=$cpus_per_node/$local_size
else
  let cpus_per_rank=$BIND_SLOTS 
fi

if [ -z "$OMP_NUM_THREADS" ]; then
  let OMP_NUM_THREADS=1
fi

# BIND_STRIDE is used in OMP_PLACES ... it will be 1 if OMP_NUM_THREADS was not set
let BIND_STRIDE=$cpus_per_rank/$OMP_NUM_THREADS

let ndx=$local_rank*$cpus_per_rank
let start_cpu=${list[$ndx]}
let pdx=$ndx+$cpus_per_rank-1
let stop_cpu=${list[$pdx]}

#---------------------------------------------
# set OMP_PLACES or GOMP_CPU_AFFINITY
#---------------------------------------------
if [ "$USE_GOMP" == "yes" ]; then
  export GOMP_CPU_AFFINITY="$start_cpu-$stop_cpu:$BIND_STRIDE"
else
  export OMP_PLACES="{$start_cpu:$OMP_NUM_THREADS:$BIND_STRIDE}"
fi

#-------------------------------------------------
# set an affinity mask for each rank
#-------------------------------------------------
printf -v command "taskset -c %d-%d "  $start_cpu $stop_cpu

#-------------------------
# run the code
#-------------------------
exec $command "$@"
