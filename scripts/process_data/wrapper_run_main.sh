#!/bin/bash
# Submits one sbatch job per dataset by calling run_main.sh with the dataset name.
# Usage: bash wrapper_run_main.sh
# Datasets: discovery cohorts (onek1k, abf300, aida, perez_sle)

datasets="data1 data7_allTPs_jalil data13 SLE"

for ds in $datasets; do
    sbatch scripts/process_data/run_main.sh $ds
    echo "Submitted: $ds"
done
