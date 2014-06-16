#! /bin/sh

export MIC_PPN=$1
export MIC_OMP_NUM_THREADS=$2
ibrun.new.symm -m $3

