#!/bin/bash


# Retrieve positional arguments
normalize=$1
corr_method=$2
denoise=$3  # optional third argument for denoise

# Display received arguments for confirmation
echo "Running correlation analysis with:"
echo "Normalization: $normalize"
echo "Correlation Method: $corr_method"
if [ "$denoise" == "denoise" ]; then
    echo "Denoise: Enabled"
else
    echo "Denoise: Disabled"
fi

# Run the Python script with parsed arguments
python src/corr_analysis.py \
    --normalize "$normalize" \
    --corr_method "$corr_method" \
    $( [ "$denoise" == "denoise" ] && echo "--denoise" )