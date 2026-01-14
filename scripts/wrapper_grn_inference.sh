#!/bin/bash

set -e

# Configuration
FORCE=true
CELL_TYPE_GRANULARITY='major'
MAIN_DIR='/vol/projects/jnourisa/'
MAX_WORKERS=5

datasets=" abf300 " #  onek1k abf300 aida soundlife

# Path to the worker script
WORKER_SCRIPT="src/grn_inference/run_grn_inference.sh"

for dataset in $datasets; do
    echo "Submitting job for dataset: $dataset"
    sbatch $WORKER_SCRIPT $dataset $FORCE $CELL_TYPE_GRANULARITY $MAIN_DIR $MAX_WORKERS
done
