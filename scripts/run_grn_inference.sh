#!/bin/bash
#SBATCH --job-name=grn_inference
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00:00
#SBATCH --mem=250GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL      
#SBATCH --mail-user=jalil.nourisa@gmail.com   

declare -A dependencies

dependencies=(
    ["grn_inference"]="src/workflows/grn_inference/script.py"
)

set -e
# Define run flags
RUN_GRN=true
CELL_TYPE_GRANULARITY='major'
MAIN_DIR='/vol/projects/jnourisa/'

MAX_WORKERS=10
data_type='sc'

datasets=" CXCL9 " #CXCL9  data1 data12 data7_allTPs_jalil   data13_Korean  data13_Japanese SLE_European

for dataset in $datasets; do
        if [ "$RUN_GRN" = true ]; then
                FORCE=true # If true, overwrite the existing files in grns directory
                SAVE_GRNS_DIR="${MAIN_DIR}/output/grns/${dataset}/"
                
                DATASET_FILE="/vol/projects/jnourisa/datasets/${data_type}/${dataset}.h5ad" # tailors raw based on the given flags such as make, downsample, etc.

                args="  
                        --dataset_file $DATASET_FILE \
                        --save_grns_dir $SAVE_GRNS_DIR \
                        --num_workers $MAX_WORKERS \
                        --data_type $data_type \
                        --cell_type_granularity $CELL_TYPE_GRANULARITY \
                        "

                [ "$FORCE" = true ] && args="${args} --force"

                cmd="python ${dependencies["grn_inference"]} $args"
                echo "Running (bash): $cmd"
                $cmd
        fi

done
