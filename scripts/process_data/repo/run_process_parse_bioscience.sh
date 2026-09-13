#!/bin/bash
#SBATCH --job-name=perturbation_preprocessing
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --mem=1000GB
#SBATCH --partition=cpu


set -e

source scripts/_env.sh


python src/process_dataset/parse_bioscience/script.py
