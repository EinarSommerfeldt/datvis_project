#!/bin/bash
#SBATCH --job-name="python-script"   # Sensible name for the job
#SBATCH --output=/cluster/home/haakohu/slurm.out
#SBATCH --account=ie-idi   # Account for consumed resources
#SBATCH --nodes=1             # Allocate 1 nodes for the job
#SBATCH -c28                  # Number of cores (can vary)
#SBATCH --time=00-00:10:00    # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --partition=CPUQ

module purge
module load Anaconda3/2020.07 
# >>> conda initialize >>> You might not need this, but in my case I need it for some reason....
__conda_setup="$('/cluster/apps/eb/software/Anaconda3/2020.07/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
   if [ -f "/cluster/apps/eb/software/Anaconda3/2020.07/etc/profile.d/conda.sh" ]; then
      . "/cluster/apps/eb/software/Anaconda3/2020.07/etc/profile.d/conda.sh"
   else
      export PATH="/cluster/apps/eb/software/Anaconda3/2020.07/bin:$PATH"
   fi
fi
unset __conda_setup
conda activate deeplearning 
cd /cluster/home/einarjso/datvis_project 
python test.py 