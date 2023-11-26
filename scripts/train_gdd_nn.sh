#!/usr/bin/env bash
# Set walltime limit
#BSUB -W 108:00
#
# Set output file
#BSUB -o  scripts/fold.%I.out
#
# Set error file
#BSUB -eo scripts/fold.%I.stderr
#
# Specify node group
#BSUB -q gpuqueue
#
# nodes: number of nodes and GPU request
#BSUB -n 1 -R "rusage[mem=30]"
#BSUB -gpu "num=1:j_exclusive=yes:mode=shared"
#
# job name (default = name of script file)
#BSUB -J "train_gdd_nn[1-10]"
source ~/.bashrc
module load cuda/10.1
conda activate gdd_ens_env
python scripts/train_gdd_nn.py "$((${LSB_JOBINDEX}-1))"
