#!/bin/bash
#SBATCH --job-name=datagen_3d_v1
#SBATCH --output=logs/datagen_3d_%j.out
#SBATCH --error=logs/datagen_3d_%j.err
#SBATCH --time=15:00:00
#SBATCH --account=fuge-prj-jrl
#SBATCH --cpus-per-task=50
#SBATCH --mem=128G

# Load modules
module load openblas/0.3.23/gcc/11.3.0/x86_64
module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-x86_64

# Activate virtual environment
unset PYTHONPATH
source /home/gapaza/scratch/environments/fem3d/bin/activate

# Change directory
cd /home/gapaza/scratch/repos/structural-thermal-3d

## Get input argument
#VOLFRAC=$1
#WEIGHT=$2
#FILENAME=$3

# Run the script
python3 -m main3D --num-procs 50 --samples 100 --save-dir /home/gapaza/scratch/datasets/thermoelastic3dv000