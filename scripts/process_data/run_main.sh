#!/bin/bash
#SBATCH --job-name=process_data
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00:00
#SBATCH --mem=500GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL      
#SBATCH --mail-user=jalil.nourisa@gmail.com   

declare -A dependencies

dependencies=(
    ["process_dataset"]="src/process_dataset/preprocess/script.py"
    ["bulkify_code"]="src/process_dataset/bulkify/script.py"
)

# Import dataset name mapping from config.py
declare -A dataset_mapping
while IFS='=' read -r key value; do
    dataset_mapping["$key"]="$value"
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
MAIN_DIR='/vol/projects/jnourisa/hiara/'


# datasets to include -> preprocessing 
datasets="  op" # data12 data7_allTPs_jalil data1 SLE 

for dataset in $datasets; do
        
        if [ "$dataset" = "op" ]; then
                input_file="/vol/projects/jnourisa/genernbi/resources/datasets_raw/op_perturbation_sc_counts.h5ad"
        elif [ "$dataset" = "CXCL9" ]; then
                input_file="/vol/projects/CIIM/Healthy_Single_Cell_Data/count_matrix/CXCL9_TI.h5ad"
        else
                input_file="/vol/projects/CIIM/Healthy_Single_Cell_Data/count_matrix/${dataset}_CMtx.h5ad"
        fi
        
        PROCESSED_FILES_DIR="${MAIN_DIR}/datasets/sc/"
        
        # Define the command
        if [ "$RUN_PROCESS_DATASET" = true ]; then
                args="--dataset $dataset --processed_files_dir $PROCESSED_FILES_DIR --input_file $input_file"
                if [ "$RUN_TEST" = true ]; then
                        args="${args} --run-test"
                fi
                cmd="python ${dependencies["process_dataset"]} $args"
                echo "Running (bash): $cmd"
                $cmd
        fi
        
done

for dataset in $datasets; do
        # Get mapped name for processed files
        if [ -n "${dataset_mapping[$dataset]}" ]; then
                mapped_name="${dataset_mapping[$dataset]}"
        else
                mapped_name="$dataset"
        fi
        
        PROCESSED_DATASET_FILE="${MAIN_DIR}/datasets/sc/${mapped_name}.h5ad"
        BULK_ALL="${MAIN_DIR}/datasets/bulk/${mapped_name}.h5ad"
        BULK_MINOR_CELLTYPE="${MAIN_DIR}/datasets/bulk/${mapped_name}_minor.h5ad"
        # BULK_M="${MAIN_DIR}/datasets/bulk/${mapped_name}_M.h5ad"
        # BULK_F="${MAIN_DIR}/datasets/bulk/${mapped_name}_F.h5ad"
        
        if [ "$RUN_PSEUDOBULK" = true ]; then
                # set the flags
                DOWNSAMPLE=false

                args="--sc_dataset_file $PROCESSED_DATASET_FILE \
                      --bulk_all $BULK_ALL \
                      --bulk_minor_celltype $BULK_MINOR_CELLTYPE"
                [ "$DOWNSAMPLE" = true ] && args="${args} --downsample"
                cmd="python ${dependencies["bulkify_code"]} $args"
                echo "Running (bash): $cmd"
                $cmd
        fi

done
