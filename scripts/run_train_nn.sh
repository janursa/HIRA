#!/bin/bash
#SBATCH --job-name=train_nn
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00:00
#SBATCH --mem=250GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL      
#SBATCH --mail-user=jalil.nourisa@gmail.com   


set -e

# Ensure conda is initialized
source ~/miniconda3/etc/profile.d/conda.sh

conda activate cpa

echo "Running training ..."
python /home/jnourisa/projs/ongoing/ciim/src/clock/NN/script_train.py --mode train
# echo "Running testing ..." 
# python src/clock/NN/script_train.py --mode test