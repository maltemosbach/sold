#!/bin/bash
##SBATCH --account YOUR_COMPUTE_GROUP
#SBATCH --cpus-per-task=8
#SBATCH --error=job_%j.err
#SBATCH --gres=gpu:1
#SBATCH --job-name=sold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=job_%j.out
#SBATCH --time=1-00:00:00  # 1 day.

apptainer run --nv ../sold.sif python "$@"
