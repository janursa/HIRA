#!/bin/bash
#SBATCH --job-name=clock_stress
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --mem=300GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jalil.nourisa@gmail.com

# Usage: sbatch scripts/exp_analysis/run_clock_stress.sh <variant>
#        python src/exp_analysis/clock_stress.py --aggregate   (after all variants finish)
set -e
[ -f .env ] && set -a && source .env && set +a
python src/exp_analysis/clock_stress.py --variant "$1"
