#!/bin/bash
#
# Master script to run complete drug reversal analysis pipeline
# 1. Compute drug stats for all TFs
# 2. Run reversal analysis
#

# Activate environment
source ~/.bash_profile
conda activate py10


echo "=========================================="
echo "DRUG REVERSAL ANALYSIS PIPELINE"
echo "=========================================="

# Default parameters
DATASET="${1:-op}"
CELL_TYPES="${2:-CD4T CD8T}"
FEATURE_TYPE="${3:-tf_activity}"

# Check for --no-weighting flag in remaining arguments
NO_WEIGHTING_FLAG=""
shift 3  # Remove first 3 positional arguments
for arg in "$@"; do
    if [ "$arg" = "--no-weighting" ]; then
        NO_WEIGHTING_FLAG="--no-weighting"
        break
    fi
done

echo ""
echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Cell types: $CELL_TYPES"
echo "  Feature type: $FEATURE_TYPE"
if [ -n "$NO_WEIGHTING_FLAG" ]; then
    echo "  Mode: Standard Fisher's exact test (no weighting)"
else
    echo "  Mode: Weighted (centrality + effect sizes)"
fi
echo ""
echo "=========================================="

# Step 1: Compute all drug stats
echo ""
echo "STEP 1: Computing drug statistics for all TFs..."
echo "=========================================="

# bash scripts/experiment/compute_all_drug_stats.sh "$DATASET" "$CELL_TYPES" "$FEATURE_TYPE"

step1_exit=$?

if [ $step1_exit -ne 0 ]; then
    echo ""
    echo "ERROR: Step 1 failed. Aborting pipeline."
    exit $step1_exit
fi

# Step 2: Run reversal analysis
echo ""
echo "STEP 2: Running reversal analysis..."
echo "=========================================="

bash scripts/experiment/run_drug_reversal.sh "$DATASET" "$CELL_TYPES" "$FEATURE_TYPE" $NO_WEIGHTING_FLAG

step2_exit=$?

if [ $step2_exit -ne 0 ]; then
    echo ""
    echo "ERROR: Step 2 failed."
    exit $step2_exit
fi

# Success
echo ""
echo "=========================================="
echo "PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Stats: base_folder/output/stats/stats_drugs_all_${DATASET}_${FEATURE_TYPE}.csv"
echo "  - Reversal: base_folder/output/perturbations/reversal_stats_${DATASET}.csv"
echo "  - Plots: output/plots/perturbations/"
echo ""
echo "=========================================="

exit 0
