#!/bin/bash
#SBATCH --job-name=batch_correction
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=4-00:00:00
#SBATCH --mem=1000GB
#SBATCH --partition=cpu
#SBATCH --qos long
#SBATCH --mail-type=END,FAIL      
#SBATCH --mail-user=jalil.nourisa@gmail.com   

python src/batch_correction/script.py
