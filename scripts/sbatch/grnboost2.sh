#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jalil.nourisa@gmail.com
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G 

echo $"singularity run ../../../images/scenic python \
    ../../task_grn_inference/src/methods/single_omics/grnboost2/script.py \
    --multiomics_rna ${1} --prediction ${2} \
    --tf_all ../../task_grn_inference/resources/prior/tf_all.csv \
    --resources_dir ../../task_grn_inference/src/utils/"

singularity run ../../../images/scenic python \
    ../../task_grn_inference/src/methods/single_omics/grnboost2/script.py \
    --multiomics_rna ${1} --prediction ${2} \
    --tf_all ../../task_grn_inference/resources/prior/tf_all.csv \
    --resources_dir ../../task_grn_inference/src/utils/