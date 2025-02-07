#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --mem=1000GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL      
#SBATCH --mail-user=jalil.nourisa@gmail.com   

output_file="/vol/projects/jnourisa/all_pbmc_ageing.h5ad"
python src/preprocess/script.py --output_file $output_file$
