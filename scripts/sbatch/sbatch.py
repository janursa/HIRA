#!/bin/bash
#SBATCH --job-name=sbatch
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00  
#SBATCH --mem=300GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

python src/helper.py  # Run your Python script
# singularity run ../../images/scgen python src/scgen_method.py --adata output/data/35_44.h5ad --batch_key donor_id --label_key cell_type
