#!/usr/bin/env bash
# Set walltime limit
#BSUB -W 72:00
#
# Set output file
#BSUB -o  generate_feature_table.out
#
# Set error file
#BSUB -eo generate_feature_table.stderr
#
# Specify node group
#BSUB -q gpuqueue -n 1 -gpu "num=1:mps=yes"
#
# nodes: number of nodes and GPU request
#BSUB -n 1 -R "rusage[mem=70]"
#BSUB -gpu "num=1:j_exclusive=yes:mode=shared"
#
# job name (default = name of script file)
#BSUB -J "generate_feature_table"
source ~/.bashrc
module load cuda/10.1
conda activate vir-env
python generate_feature_table.py /data/morrisq/darmofam/gr37.fasta msk_solid_heme/

