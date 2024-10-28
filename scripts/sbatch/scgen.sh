#!/bin/bash
#SBATCH --job-name=scgen
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00  
#SBATCH --mem=250GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

singularity run ../../images/scgen python src/scgen_method.py --adata output/data/35_44.h5ad --adata_bc output/data/35_44_bc.h5ad --batch_key donor_id --label_key cell_type
