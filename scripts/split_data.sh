#!/usr/bin/env bash
# Set walltime limit
#BSUB -W 72:00
#
# Set output file
#BSUB -o  scripts/split_data.out
#
# Set error file
#BSUB -eo scripts/split_data.stderr
#
# Specify node group
#BSUB -q gpuqueue -n 1 -gpu "num=1:mps=yes"
#
# nodes: number of nodes and GPU request
#BSUB -n 1 -R "rusage[mem=50]"
#BSUB -gpu "num=1:j_exclusive=yes:mode=shared"
#
# job name (default = name of script file)
#BSUB -J "split_data"
source ~/.bashrc
module load cuda/10.1
conda activate gdd_ens_env
python scripts/split_data.py output/msk_solid_heme_ft.csv

