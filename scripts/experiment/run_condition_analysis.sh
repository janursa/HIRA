#!/bin/bash
#
# Unified script for running condition-based analyses (disease and perturbation).
# Replaces separate disease and perturbation scripts.
#

set -e  # Exit on error

# Activate environment
source ~/.bash_profile
conda activate py10

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# Default values
DATASET=""
CELL_TYPES="CD4T CD8T"
FEATURE_TYPE="tf_activity"
DATA_TYPE="bulk"
SKIP_FEATURES="false"

# Parse keyword arguments
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
        --data-type)
            DATA_TYPE="$2"
            shift 2
            ;;
        --skip-features)
            SKIP_FEATURES="true"
            shift
            ;;
        -h|--help)
            echo ""
            echo "Usage: run_condition_analysis.sh --dataset <name> [options]"
            echo ""
            echo "Required arguments:"
            echo "  --dataset <name>        Dataset name (e.g., SLE_European, op, CXCL9)"
            echo ""
            echo "Optional arguments:"
            echo "  --cell-types <types>    Cell types to analyze (default: 'CD4T CD8T')"
            echo "  --feature-type <type>   Feature type: tf_activity or gene_expression (default: tf_activity)"
            echo "  --data-type <type>      Data type: bulk or sc (default: bulk)"
            echo "  --skip-features         Skip feature calculation, use cached data"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Available datasets:"
            echo "  Disease:      SLE_European, Covid_50MHH"
            echo "  Perturbation: op, CXCL9, parsebioscience"
            echo ""
            echo "Examples:"
            echo "  # Disease analysis with defaults"
            echo "  bash scripts/experiment/run_condition_analysis.sh --dataset SLE_European"
            echo ""
            echo "  # Perturbation analysis with custom cell types"
            echo "  bash scripts/experiment/run_condition_analysis.sh --dataset op --cell-types \"CD4T CD8T NK\""
            echo ""
            echo "  # Gene expression analysis"
            echo "  bash scripts/experiment/run_condition_analysis.sh --dataset SLE_European --feature-type gene_expression"
            echo ""
            echo "  # Single-cell data type"
            echo "  bash scripts/experiment/run_condition_analysis.sh --dataset op --data-type sc"
            echo ""
            echo "  # Skip feature calculation (use cached)"
            echo "  bash scripts/experiment/run_condition_analysis.sh --dataset SLE_European --skip-features"
            echo ""
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Condition Analysis (Unified)"
echo "=========================================="

# Validate required arguments
if [ -z "$DATASET" ]; then
    echo ""
    echo "ERROR: --dataset is required"
    echo ""
    echo "Use --help for usage information"
    echo ""
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Cell types: $CELL_TYPES"
echo "  Feature type: $FEATURE_TYPE"
echo "  Data type: $DATA_TYPE"
echo "  Skip feature calculation: $SKIP_FEATURES"
echo ""
echo "=========================================="
echo ""

# Build command
CMD="python -m ciim.src.feature_association.run_analysis \
    --dataset \"$DATASET\" \
    --cell-types $CELL_TYPES \
    --feature-type \"$FEATURE_TYPE\" \
    --data-type \"$DATA_TYPE\""

if [ "$SKIP_FEATURES" = "true" ]; then
    CMD="$CMD --skip-features"
fi

# Run analysis
echo "Running analysis..."
echo ""
eval $CMD

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "=========================================="
    echo "SUCCESS: Analysis complete"
    echo "=========================================="
    echo ""
    echo "Results saved to:"
    echo "  base_folder/output/stats/stats_${DATASET}_${DATA_TYPE}_${FEATURE_TYPE}_*.csv"
    echo ""
else
    echo "=========================================="
    echo "ERROR: Analysis failed with code $exit_code"
    echo "=========================================="
fi

exit $exit_code
