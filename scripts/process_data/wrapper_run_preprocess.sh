#!/bin/bash
# Submits one sbatch job per dataset by calling run_preprocess.sh with the dataset name.
# Usage: bash wrapper_run_preprocess.sh
# Datasets: discovery cohorts (onek1k, abf300, aida, perez_sle), rest: , parsebioscience, wang, soundlife, op, CXCL9

source scripts/_env.sh

datasets="onek1k abf300 aida perez_sle parsebioscience wang soundlife op CXCL9"

# Per-dataset overrides for run_preprocess.sh's default --mem=500GB --time=20:00:00
# (sbatch CLI flags win over the script's #SBATCH defaults):
#   soundlife: 13.8M-cell final merge OOMs at 500GB -> needs more memory
#   parsebioscience: 20 chunks x ~1h each runs past 20h; final merge OOMs at 500GB (peak 469GiB)
declare -A extra_sbatch_args=(
    [soundlife]="--mem=1000GB"
    [parsebioscience]="--time=30:00:00 --mem=1000GB"
)

for ds in $datasets; do
    sbatch $SBATCH_MAIL ${extra_sbatch_args[$ds]} scripts/process_data/run_preprocess.sh $ds
    echo "Submitted: $ds"
done


