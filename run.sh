#!/bin/bash
#SBATCH -J SDM						  # name of job
#SBATCH -A eecs	  # name of my sponsored account, e.g. class or research group, NOT ONID!
#SBATCH -t 2-00:00:00
#SBATCH -p dgx2								  # name of partition or queue
#SBATCH -o SDM.out				  # name of output file for this submission script
#SBATCH -e SDM.err				  # name of error file for this submission script
#SBATCH --gres=gpu:1

#SBATCH --mail-type=BEGIN,END,FAIL                # send email when job begins, ends or aborts

#SBATCH --mail-user=ahmedna@oregonstate.edu        # send email to this address

# load any software environment module required for app (e.g. matlab, gcc, cuda)
module load cuda/11.3 gcc/9.3

export PYTHON=/nfs/hpc/share/ahmedna/miniconda3/envs/pyspecies/bin/python3.9 # give your environments path

# run my job (e.g. matlab, python)
$PYTHON main.py
