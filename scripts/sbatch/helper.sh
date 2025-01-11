#!/bin/bash
#SBATCH --job-name=ciim_helper
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=400GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL      
#SBATCH --mail-user=jalil.nourisa@gmail.com   


# dataset="data1"
dataset="pbmc_ageing"
data_file="input/${dataset}_downsample.h5ad"

python src/helper_dataset.py \
    --save_file ${data_file} --dataset ${dataset} --downsample && \

python src/helper_diff_analys.py --data_file ${data_file}