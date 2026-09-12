#!/bin/bash
#SBATCH --job-name=naive_effector
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --mem=500GB
#SBATCH --partition=cpu

# Reruns the aging association step on tfa_major_b's cached TF-activity features with a
# naive_ratio covariate, then checks how many of tfa_major_b's significant TFs survive.
# Usage: sbatch scripts/exp_analysis/run_naive_effector.sh
set -e
source scripts/_env.sh

python src/feature_association/run_analysis.py --analysis-name tfa_major_b_ctNaiveToEffector \
    --analysis-mode multi-cohort --association-type continous
python src/exp_analysis/naive_effector.py
