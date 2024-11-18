#!/bin/bash
#SBATCH --job-name=bce_ciim
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --mem=120GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL      
#SBATCH --mail-user=jalil.nourisa@gmail.com   


singularity run ../../images/scib python ../batch_correction/bce_nourisa/script.py --adata input/dataset_1_corrected_batch1.h5ad --temp_dir output/temp_dir --batch_key donor_id --label_key cell_type --baseline_layer counts --corrected_layer combat_corrected --normalize