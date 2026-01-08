#!/bin/bash
#
# Unified script for running condition-based analyses (disease and perturbation).
# Replaces separate disease and perturbation scripts.
#

# Activate environment (allow bashrc errors to be ignored)
source ~/.bash_profile || true
conda activate py10

set -e  # Exit on error from this point forward

# Build command - pass all arguments directly to Python script
CMD="python -m ciim.src.feature_association.run_analysis $@"

# Run analysis
echo "Running analysis..."
echo "Command: $CMD"
echo ""
eval $CMD

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "=========================================="
    echo "SUCCESS: Analysis complete"
    echo "=========================================="
else
    echo "=========================================="
    echo "ERROR: Analysis failed with code $exit_code"
    echo "=========================================="
fi

exit $exit_code
