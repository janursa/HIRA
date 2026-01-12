#!/bin/bash
#SBATCH --job-name=in-silico
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=2:00:00
#SBATCH --mem=100GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL      
#SBATCH --mail-user=jalil.nourisa@gmail.com   


set -e

# python src/workflows/in_silico_simulation/script.py 
python /home/jnourisa/projs/ongoing/hiara/src/insilico_perturbation/optimization/script.py