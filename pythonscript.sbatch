#!/bin/bash
#SBATCH --job-name="python-script"   # Sensible name for the job
#SBATCH --output=/cluster/home/einarjso/output/OUT%j.slurm
#SBATCH --account=ie-idi   # Account for consumed resources
#SBATCH --nodes=1             # Allocate 1 nodes for the job
#SBATCH -c28                  # Number of cores (can vary)
#SBATCH --time=00-00:10:00    # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --partition=CPUQ

module purge
module load Anaconda3/2020.07 

unset __conda_setup
conda activate deeplearning 
cd /cluster/home/einarjso/datvis_project 
python test.py 