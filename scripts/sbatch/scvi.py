#!/bin/bash
#SBATCH --job-name=scvi
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00  # Set the desired time limit
#SBATCH --mem=250GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

conda activate scvi
python src/scvi_method.py  # Run your Python script
