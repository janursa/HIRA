#!/bin/bash

set -e

# Load repo-level config (HIRA_DIR, HIRA_BASE_DIR, ...) if present
[ -f .env ] && set -a && source .env && set +a

# Configuration
FORCE=true
CELL_TYPE_GRANULARITY='major'
MAX_WORKERS=5

datasets=" onek1k abf300 aida perez_sle " #  onek1k abf300 aida perez_sle

# Path to the worker script
WORKER_SCRIPT="scripts/grn_inference/run_grn_inference.sh"

for dataset in $datasets; do
    echo "Submitting job for dataset: $dataset"
    # 4th arg (GRNS_DIR) omitted: run_grn_inference.sh defaults it to config.py's GRNS_DIR,
    # matching where feature_analysis.sh/clock_analysis.sh read GRNs from.
    sbatch $WORKER_SCRIPT $dataset $FORCE $CELL_TYPE_GRANULARITY '' $MAX_WORKERS
done
