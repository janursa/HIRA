#!/bin/bash
#SBATCH --job-name=test_subknn
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00:00
#SBATCH --mem=500GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jalil.nourisa@gmail.com

# Test subsample-kNN annotation on onek1k and compare against CellTypist majority voting.
# Output saved to datasets/sc/onek1k_subknn.h5ad (does not overwrite production files).

export HIARA_ANNOTATE_METHOD=subsample_knn

MAIN_DIR='/vol/projects/jnourisa/hiara'
INPUT_FILE="/vol/projects/CIIM/Healthy_Single_Cell_Data/count_matrix/data1_CMtx.h5ad"
OUT_DIR="${MAIN_DIR}/datasets/sc_test_subknn"
mkdir -p "$OUT_DIR"

set -e

# Override discovery cohort check by passing dataset name 'onek1k_test' (unmapped, not in DISCOVERY_COHORTS)
# so the subsample_knn branch is taken instead of majority_voting.
python src/process_data/preprocess/script.py \
    --dataset data1 \
    --processed_files_dir "$OUT_DIR" \
    --input_file "$INPUT_FILE"
