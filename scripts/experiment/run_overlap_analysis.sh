#!/bin/bash
#
# Master script to run complete intervention overlap analysis pipeline
# This identifies rejuvenating/accelerating interventions by comparing with aging
#
# Steps:
# 1. Run condition analysis (if needed) to get perturbation/disease stats
# 2. Run overlap analysis to identify rejuvenating/accelerating effects
#

set -e

# Activate environment
source ~/.bash_profile
conda activate py10

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

echo "=========================================="
echo "INTERVENTION OVERLAP ANALYSIS PIPELINE"
echo "=========================================="

# Parse keyword arguments
DATASET=""
CELL_TYPES="CD4T CD8T"
FEATURE_TYPE="tf_activity"
SKIP_CONDITION_STATS="false"
NO_WEIGHTING_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --cell-types)
            CELL_TYPES="$2"
            shift 2
            ;;
        --feature-type)
            FEATURE_TYPE="$2"
            shift 2
            ;;
        --skip-condition-stats)
            SKIP_CONDITION_STATS="true"
            shift
            ;;
        --no-weighting)
            NO_WEIGHTING_FLAG="--no-weighting"
            shift
            ;;
        -h|--help)
            echo ""
            echo "Usage: run_overlap_analysis.sh --dataset <name> [options]"
            echo ""
            echo "Required arguments:"
            echo "  --dataset <name>        Dataset name (e.g., op, CXCL9, parsebioscience, SLE_European)"
            echo ""
            echo "Optional arguments:"
            echo "  --cell-types <types>    Cell types (default: 'CD4T CD8T')"
            echo "  --feature-type <type>   Feature type (default: tf_activity)"
            echo "  --skip-condition-stats  Skip condition analysis (use cached stats)"
            echo "  --no-weighting          Use standard Fisher's test (no centrality weighting)"
            echo "  -h, --help              Show this help"
            echo ""
            echo "This pipeline:"
            echo "  1. Runs condition analysis (disease/perturbation) if needed"
            echo "  2. Compares with aging signatures to find rejuvenating/accelerating effects"
            echo ""
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

if [ -z "$DATASET" ]; then
    echo "ERROR: --dataset is required"
    echo "Use --help for usage"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Cell types: $CELL_TYPES"
echo "  Feature type: $FEATURE_TYPE"
echo "  Skip condition stats: $SKIP_CONDITION_STATS"
if [ -n "$NO_WEIGHTING_FLAG" ]; then
    echo "  Mode: Standard Fisher's exact test (no weighting)"
else
    echo "  Mode: Weighted (centrality + effect sizes)"
fi
echo ""
echo "=========================================="

# Step 1: Run condition analysis (if not skipped)
if [ "$SKIP_CONDITION_STATS" = "false" ]; then
    echo ""
    echo "STEP 1: Running condition analysis..."
    echo "=========================================="
    
    bash "$SCRIPT_DIR/run_condition_analysis.sh" \
        --dataset "$DATASET" \
        --cell-types "$CELL_TYPES" \
        --feature-type "$FEATURE_TYPE"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Condition analysis failed"
        exit 1
    fi
else
    echo ""
    echo "STEP 1: Skipping condition analysis (using cached stats)"
fi

# Step 2: Run overlap/reversal analysis
echo ""
echo "STEP 2: Running overlap analysis..."
echo "=========================================="

python -m ciim.src.overlap_analysis.run_drug_reversal \
    --dataset "$DATASET" \
    --cell-types $CELL_TYPES \
    --feature-type "$FEATURE_TYPE" \
    $NO_WEIGHTING_FLAG

if [ $? -ne 0 ]; then
    echo "ERROR: Overlap analysis failed"
    exit 1
fi

# Success
echo ""
echo "=========================================="
echo "PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Condition stats: base_folder/output/stats/stats_${DATASET}_*.csv"
echo "  - Overlap results: base_folder/output/perturbations/reversal_stats_${DATASET}.csv"
echo "  - Rejuvenating: base_folder/output/perturbations/rejuvenating_drugs_${DATASET}.csv"
echo ""
echo "=========================================="

exit 0
