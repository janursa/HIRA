#!/bin/bash
#SBATCH --job-name=sbatch
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00  
#SBATCH --mem=600GB
#SBATCH --partition=cpu


python src/test.py  # Run your Python script
# singularity run ../../images/scgen python src/scgen_method.py --adata output/data/35_44.h5ad --batch_key donor_id --label_key cell_type
