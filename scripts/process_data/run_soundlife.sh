#!/bin/bash
#SBATCH --job-name=soundlife_preprocessing
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=20:00:00
#SBATCH --mem=500GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL      
#SBATCH --mail-user=jalil.nourisa@gmail.com   

set -e

# Print start time
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo ""


# Set test mode (set to true for testing with subset, false for full processing)
TEST_MODE=false

# Build command
CMD="python src/process_dataset/soundlife/script.py \
    --input_dir /vol/projects/jnourisa/datasets/soundlife/ \
    --output_bulk /vol/projects/jnourisa/datasets/bulk/soundlife_bulk.h5ad \
    --output_metacell /vol/projects/jnourisa/datasets/metacell/soundlife_metacell.h5ad \
    --temp_dir /home/jnourisa/projs/ongoing/ciim/tmp/soundlife_processing"

# Add test flag if in test mode
if [ "$TEST_MODE" = true ]; then
    CMD="$CMD --test"
    echo "*** Running in TEST MODE (subsetting to 2 donors and 2 visits per file) ***"
    echo ""
else
    echo "*** Running in FULL MODE (processing all data) ***"
    echo ""
fi

# Run the SoundLife preprocessing pipeline
eval $CMD

echo ""
echo "Job completed at: $(date)"
echo "SoundLife preprocessing completed successfully!"
