#!/usr/bin/env bash
# Set walltime limit
#BSUB -W 72:00
#
# Set output file
#BSUB -o  gdd_ensemble.out
#
# Set error file
#BSUB -eo gdd_ensemble.stderr
#
# Specify node group
#BSUB -q gpuqueue -n 1 -gpu "num=1:mps=yes"
#
# nodes: number of nodes and GPU request
#BSUB -n 1 -R "rusage[mem=50]"
#BSUB -gpu "num=1:j_exclusive=yes:mode=shared"
#
# job name (default = name of script file)
#BSUB -J "gdd_ensemble"
source ~/.bashrc
module load cuda/10.1
conda activate vir-env
python gdd_ensemble.py

