#!/bin/bash
#SBATCH -J SDM						  # name of job
#SBATCH -A eecs	  # name of my sponsored account, e.g. class or research group, NOT ONID!
#SBATCH -t 3-00:00:00
#SBATCH -p dgx2								  # name of partition or queue
#SBATCH -o SDM.out				  # name of output file for this submission script
#SBATCH -e SDM.err				  # name of error file for this submission script
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=islammoh@oregonstate.edu
# load any software environment module required for app (e.g. matlab, gcc, cuda)
module load cuda/11.3 gcc/9.3

export PYTHON=/nfs/hpc/share/islammoh/miniconda3/envs/gnn/bin/python3.9 # give your environments path

# run my job (e.g. matlab, python)
$PYTHON main.py
