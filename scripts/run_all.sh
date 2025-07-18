#!/bin/bash
#SBATCH --job-name=all_steps
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --mem=800GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL      
#SBATCH --mail-user=jalil.nourisa@gmail.com   

declare -A dependencies

dependencies=(
    ["process_dataset"]="src/process_dataset/preprocess/script.py"
    ["bulkify_code"]="src/process_dataset/bulkify/script.py"
    ["grn_inference"]="src/workflows/grn_inference/script.py"
    
)

set -e
# Define run flags
RUN_PROCESS_DATASET=true
RUN_PSEUDOBULK=true
RUN_GRN=true
CELL_TYPE_GRANULARITY='major'
MAIN_DIR='/vol/projects/jnourisa/'
RUN_ASSOCIATION=false

MAX_WORKERS=10
data_type='sc'

# datasets to include -> preprocessing 
datasets="data7_allTPs_jalil data1 data13 SLE CXCL9" #data12 data7_allTPs_jalil data1 data13 SLE   CXCL9

for dataset in $datasets; do
        RAW_FILES_DIR="/vol/projects/CIIM/Healthy_Single_Cell_Data/count_matrix/"
        PROCESSED_FILES_DIR="${MAIN_DIR}/datasets/"
        
        # Define the command
        if [ "$RUN_PROCESS_DATASET" = true ]; then
                args="--dataset_name $dataset --processed_files_dir $PROCESSED_FILES_DIR --raw_files_dir $RAW_FILES_DIR"
                cmd="python ${dependencies["process_dataset"]} $args"
                echo "Running (bash): $cmd"
                $cmd
        fi
        
done

datasets=" CXCL9  data1 data12 data7_allTPs_jalil   data13_Korean  data13_Japanese SLE_European" #CXCL9  data1 data12 data7_allTPs_jalil   data13_Korean  data13_Japanese SLE_European


for dataset in $datasets; do
        PROCESSED_DATASET_FILE="${MAIN_DIR}/datasets/${dataset}_sc.h5ad"
        BULK_ALL="${MAIN_DIR}/datasets/${dataset}_bulk.h5ad"
        BULK_MINOR_CELLTYPE="${MAIN_DIR}/datasets/${dataset}_bulk_minor.h5ad"
        BULK_M="${MAIN_DIR}/datasets/${dataset}_bulk_M.h5ad"
        BULK_F="${MAIN_DIR}/datasets/${dataset}_bulk_F.h5ad"
        
        if [ "$RUN_PSEUDOBULK" = true ]; then
                # set the flags
                DOWNSAMPLE=false

                args="--sc_dataset_file $PROCESSED_DATASET_FILE \
                      --bulk_all $BULK_ALL \
                      --bulk_minor_celltype $BULK_MINOR_CELLTYPE \
                      --bulk_M $BULK_M \
                      --bulk_F $BULK_F "
                [ "$DOWNSAMPLE" = true ] && args="${args} --downsample"
                cmd="python ${dependencies["bulkify_code"]} $args"
                echo "Running (bash): $cmd"
                $cmd
        fi
        if [ "$RUN_GRN" = true ]; then
                FORCE=true # If true, overwrite the existing files in grns directory
                SAVE_GRNS_DIR="${MAIN_DIR}/output/grns/${dataset}/"
                
                DATASET_FILE="/vol/projects/jnourisa/datasets/${dataset}_${data_type}.h5ad" # tailors raw based on the given flags such as make, downsample, etc.

                args="  
                        --dataset_file $DATASET_FILE \
                        --save_grns_dir $SAVE_GRNS_DIR \
                        --max_workers $MAX_WORKERS \
                        --data_type $data_type \
                        --cell_type_granularity $CELL_TYPE_GRANULARITY \
                        "

                [ "$FORCE" = true ] && args="${args} --force"

                cmd="python ${dependencies["grn_inference"]} $args"
                echo "Running (bash): $cmd"
                $cmd
        fi

done

if [ "$RUN_ASSOCIATION" = true ]; then
        bash scripts/run_association_analysis.sh
fi
