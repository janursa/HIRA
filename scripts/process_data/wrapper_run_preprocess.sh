#!/bin/bash
# Submits one sbatch job per dataset by calling run_main.sh with the dataset name.
# Usage: bash wrapper_run_main.sh
# Datasets: discovery cohorts (onek1k, abf300, aida, perez_sle), rest: , parsebioscience, zhang, soundlife, op, CXCL9

datasets="onek1k abf300 aida perez_sle parsebioscience zhang soundlife op CXCL9"

# Per-dataset overrides for run_preprocess.sh's default --mem=500GB --time=20:00:00
# (sbatch CLI flags win over the script's #SBATCH defaults):
#   soundlife: 13.8M-cell final merge OOMs at 500GB -> needs more memory
#   parsebioscience: 20 chunks x ~1h each runs past the 20h default -> needs more time
declare -A extra_sbatch_args=(
    [soundlife]="--mem=1000GB"
    [parsebioscience]="--time=30:00:00"
)

for ds in $datasets; do
    sbatch ${extra_sbatch_args[$ds]} scripts/process_data/run_preprocess.sh $ds
    echo "Submitted: $ds"
done


