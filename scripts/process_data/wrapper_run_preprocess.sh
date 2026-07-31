#!/bin/bash
# Submits one sbatch job per dataset by calling run_main.sh with the dataset name.
# Usage: bash wrapper_run_main.sh
# Datasets: discovery cohorts (onek1k, abf300, aida, perez_sle), rest: , parsebioscience, zhang, soundlife, op, CXCL9

datasets="onek1k abf300 aida perez_sle"

for ds in $datasets; do
    sbatch scripts/process_data/run_preprocess.sh $ds
    echo "Submitted: $ds"
done


