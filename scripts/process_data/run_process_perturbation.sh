#!/bin/bash
#SBATCH --job-name=perturbation_preprocessing
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --mem=1000GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL      
#SBATCH --mail-user=jalil.nourisa@gmail.com   


set -e


python src/process_dataset/parse_bioscience/script.py
# python src/process_dataset/xaira/script.py
