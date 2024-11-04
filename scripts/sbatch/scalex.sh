#!/bin/bash
#SBATCH --job-name=scgen
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00:00  
#SBATCH --mem=64GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

conda activate scalex
python src/scalex_method.py --adata output/data/adata_34_5donors.h5ad --adata_bc output/data/adata_34_5donors_bc_scalex.h5ad --batch_key donor_id
