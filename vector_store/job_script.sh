#!/bin/bash --login

# Set required resources
#SBATCH --job-name=cvs
#SBATCH --ntasks=1
#SBATCH --gpus=v100:1
#SBATCH --mem=32G
#SBATCH --time=0:15:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lellaom@msu.edu
#SBATCH --output=cvs.out
#SBATCH --error=cvs.err

# load virtual environment
source /mnt/scratch/lellaom/Medical-Assistant-Bot/venv/bin/activate

# run python code
cd "/mnt/scratch/lellaom/Medical-Assistant-Bot/vector_store"
srun python create_vector_store.py

# write job information to output file
scontrol show job $SLURM_JOB_ID
module load powertools
js -j $SLURM_JOB_ID