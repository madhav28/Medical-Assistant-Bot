#!/bin/bash --login

# Set required resources
#SBATCH --job-name=mcs
#SBATCH --ntasks=1
#SBATCH --gpus=v100s:1
#SBATCH --mem=32G
#SBATCH --time=0:10:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lellaom@msu.edu
#SBATCH --output=mcs.out
#SBATCH --error=mcs.err

# load virtual environment
source /mnt/scratch/lellaom/Medical-Assistant-Bot/venv/bin/activate

# run python code
cd "/mnt/scratch/lellaom/Medical-Assistant-Bot/evaluation_metrics"
srun python mean_cosine_similarity.py

# write job information to output file
scontrol show job $SLURM_JOB_ID
module load powertools
js -j $SLURM_JOB_ID