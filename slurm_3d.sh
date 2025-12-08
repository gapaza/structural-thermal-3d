#!/bin/bash
#SBATCH --job-name=pretrain_study13
#SBATCH --output=logs/datagen_3d_%j.out
#SBATCH --error=logs/datagen_3d_%j.err
#SBATCH --time=15:00:00
#SBATCH --account=fuge-prj-jrl
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Load modules
module load openblas/0.3.23/gcc/11.3.0/x86_64
module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-x86_64

# Activate virtual environment
source /home/gapaza/scratch/repos/diffusion-top-rl/nvenv/bin/activate

# Change directory
cd /home/gapaza/scratch/repos/diffusion-top-rl

# Get input argument
DATAPOINTS=$1

# Get pass argument
PASS=$2

# Run the training script
python3 -m studies.study13.gradient.pretrain --datapoints $DATAPOINTS --pass_number $PASS