#!/usr/bin/env bash
# Set walltime limit
#BSUB -W 72:00
#
# Set output file
#BSUB -o  single_prior.out
#
# Set error file
#BSUB -eo single_prior.stderr
#
# Specify node group
#BSUB -q gpuqueue -n 1 -gpu "num=1:mps=yes"
#
# nodes: number of nodes and GPU request
#BSUB -n 1 -R "rusage[mem=50]"
#BSUB -gpu "num=1:j_exclusive=yes:mode=shared"
#
# job name (default = name of script file)
#BSUB -J "single_prior"
source ~/.bashrc
module load cuda/10.1
conda activate vir-env
python adaptable_prior.py 'prior_table_single.csv' 'template_output_liver.csv' 'template_allprobs_liver.csv' single

