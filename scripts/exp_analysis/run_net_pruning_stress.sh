#!/bin/bash
#SBATCH --job-name=net_pruning_stress
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00:00
#SBATCH --mem=200GB
#SBATCH --partition=cpu

# Usage: sbatch scripts/exp_analysis/run_net_pruning_stress.sh <skeleton|promotor|unfiltered>
#        python src/exp_analysis/net_pruning_stress.py --aggregate   (after all variants finish)
set -e
source scripts/_env.sh
python src/exp_analysis/net_pruning_stress.py --variant "$1"
