#!/bin/bash
#SBATCH --job-name=process_data
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=40:00:00
#SBATCH --mem=500GB
#SBATCH --partition=cpu

# Usage: sbatch run_preprocess.sh <dataset>
# e.g.:  sbatch run_preprocess.sh data1
dataset=$1
if [ -z "$dataset" ]; then
    echo "ERROR: no dataset provided. Usage: sbatch run_preprocess.sh <dataset>"
    exit 1
fi

source scripts/_env.sh

declare -A dependencies

dependencies=(
    ["process_dataset"]="src/process_data/preprocess/script.py"
    ["bulkify_code"]="src/process_data/bulkify/script.py"
    ["metacell_code"]="src/process_data/metacell/script.py"
)

# Import dataset name mapping from config.py (raw file key -> friendly name)
declare -A dataset_mapping
declare -A raw_key_for
while IFS='=' read -r key value; do
    dataset_mapping["$key"]="$value"
    raw_key_for["$value"]="$key"
done < <(python -c "
import sys
sys.path.insert(0, 'src')
from config import DATASET_NAME_MAPPING
for k, v in DATASET_NAME_MAPPING.items():
    print(f'{k}={v}')
")

set -e

# Define run flags
RUN_TEST=false
RUN_PROCESS_DATASET=true
RUN_PSEUDOBULK=true
RUN_METACELL=true
MAIN_DIR=$(python -c "import sys; sys.path.insert(0, 'src'); from config import base_dir; print(base_dir)")

# Root of the raw data lake (see README > Data Acquisition). Override with HIRA_RAW_DIR,
# or set INPUT_FILE_OVERRIDE to point at a single custom raw file/dir for this run.
RAW_DATA_DIR="${HIRA_RAW_DIR:?set HIRA_RAW_DIR in .env}"

if [ -n "$INPUT_FILE_OVERRIDE" ]; then
        input_file="$INPUT_FILE_OVERRIDE"
elif [ "$dataset" = "soundlife" ]; then
        input_file="${RAW_DATA_DIR}/soundlife/"  # DIRECTORY with multiple h5ad files
elif [ "$dataset" = "parsebioscience" ]; then
        input_file="${RAW_DATA_DIR}/perturbation_data/Parse_10M_PBMC_cytokines.h5ad"
elif [ "$dataset" = "op" ]; then
        input_file="${HIRA_OP_RAW_FILE:?set HIRA_OP_RAW_FILE in .env}"
elif [ "$dataset" = "CXCL9" ]; then
        input_file="${RAW_DATA_DIR}/Healthy_Single_Cell_Data/count_matrix/CXCL9_TI.h5ad"
else
        raw_key="${raw_key_for[$dataset]:-$dataset}"
        input_file="${RAW_DATA_DIR}/Healthy_Single_Cell_Data/count_matrix/${raw_key}_CMtx.h5ad"
fi

PROCESSED_FILES_DIR="${MAIN_DIR}/datasets/sc/"
mkdir -p "${MAIN_DIR}/datasets/sc" "${MAIN_DIR}/datasets/bulk" "${MAIN_DIR}/datasets/bulk_minor" "${MAIN_DIR}/datasets/metacell"

if [ "$RUN_PROCESS_DATASET" = true ]; then
        args="--dataset $dataset --processed_files_dir $PROCESSED_FILES_DIR --input_file $input_file"
        if [ "$RUN_TEST" = true ]; then
                args="${args} --run-test"
        fi
        cmd="python ${dependencies["process_dataset"]} $args"
        echo "Running (bash): $cmd"
        $cmd
fi

# Get mapped name for processed files
if [ -n "${dataset_mapping[$dataset]}" ]; then
        mapped_name="${dataset_mapping[$dataset]}"
else
        mapped_name="$dataset"
fi
PROCESSED_DATASET_FILE="${MAIN_DIR}/datasets/sc/${mapped_name}.h5ad"
BULK_ALL="${MAIN_DIR}/datasets/bulk/${mapped_name}.h5ad"
BULK_MINOR_CELLTYPE="${MAIN_DIR}/datasets/bulk_minor/${mapped_name}.h5ad"
METACELL_OUT="${MAIN_DIR}/datasets/metacell/${mapped_name}.h5ad"

if [ "$RUN_PSEUDOBULK" = true ]; then
        DOWNSAMPLE=false

        args="--sc_dataset_file $PROCESSED_DATASET_FILE \
              --bulk_all $BULK_ALL \
              --bulk_minor_celltype $BULK_MINOR_CELLTYPE"
        [ "$DOWNSAMPLE" = true ] && args="${args} --downsample"
        cmd="python ${dependencies["bulkify_code"]} $args"
        echo "Running (bash): $cmd"
        $cmd
fi

if [ "$RUN_METACELL" = true ]; then
        args="--sc_dataset_file $PROCESSED_DATASET_FILE --metacell_out $METACELL_OUT"
        [ "$RUN_TEST" = true ] && args="${args} --run-test"
        cmd="python ${dependencies["metacell_code"]} $args"
        echo "Running (bash): $cmd"
        $cmd
fi
