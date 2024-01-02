#!/bin/bash
nodes=2
ppn=6
let nmpi=$nodes*$ppn
let cores=42*${nodes}
#--------------------------------------
cat >batch.job <<EOF
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -nnodes ${nodes}
#BSUB -P VEN201
#BSUB -J gpucheck
#BSUB -q batch
#BSUB -W 10
#---------------------------------------
 jsrun --rs_per_host 1 --cpu_per_rs 42 --gpu_per_rs 6 -d plane:${ppn} --bind proportional-packed:7  --np ${nmpi} ./osnoise.x -c 3.0 -t 300 -x 100 -k lut -b
EOF
#---------------------------------------
bsub  batch.job
