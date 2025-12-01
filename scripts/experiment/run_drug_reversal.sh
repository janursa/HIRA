#!/bin/bash
#
# Run drug reversal analysis
# Identifies rejuvenating drugs based on TF activity reversal
#

# Activate environment
source ~/.bash_profile
conda activate py10

# Navigate to project root
cd "$(dirname "$0")/.." || exit 1

echo "=================================="
echo "Drug Reversal Analysis"
echo "=================================="

# Default parameters
DATASET="${1:-op}"
CELL_TYPES="${2:-CD4T CD8T}"
FEATURE_TYPE="${3:-tf_activity}"

echo "Dataset: $DATASET"
echo "Cell types: $CELL_TYPES"
echo "Feature type: $FEATURE_TYPE"
echo ""

# Run the script
python -m ciim.src.feature_association.perturbation.run_drug_reversal \
    --dataset "$DATASET" \
    --cell-types $CELL_TYPES \
    --feature-type "$FEATURE_TYPE"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "SUCCESS: Reversal analysis complete"
    echo "=================================="
else
    echo ""
    echo "=================================="
    echo "ERROR: Script failed with code $exit_code"
    echo "=================================="
fi

exit $exit_code
