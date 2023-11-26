#!/usr/bin/env bash
# Set walltime limit
#BSUB -W 72:00
#
# Set output file
#BSUB -o  scripts/multi_prior.out
#
# Set error file
#BSUB -eo scripts/multi_prior.stderr
#
# Specify node group
#BSUB -q gpuqueue -n 1 -gpu "num=1:mps=yes"
#
# nodes: number of nodes and GPU request
#BSUB -n 1 -R "rusage[mem=50]"
#BSUB -gpu "num=1:j_exclusive=yes:mode=shared"
#
# job name (default = name of script file)
#BSUB -J "multi_prior"
source ~/.bashrc
module load cuda/10.1
conda activate gdd_ens_env
python scripts/adaptable_prior.py data/prior_table_multi.csv output/single_ft_res.csv output/multi_adj_single_ft_res.csv 

