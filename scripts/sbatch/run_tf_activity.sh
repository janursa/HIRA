#!/bin/bash
#SBATCH --job-name=tf_activity
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00:00
#SBATCH --mem=500GB
#SBATCH --partition=cpu
#SBATCH --qos long
#SBATCH --mail-type=END,FAIL      
#SBATCH --mail-user=jalil.nourisa@gmail.com  


python src/feature_association/script.py