#!/bin/bash
#SBATCH --job-name=grn_inference
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=4:00:00
#SBATCH --mem=1000GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL      
#SBATCH --mail-user=jalil.nourisa@gmail.com   

declare -A dependencies

dependencies=(
    ["grn_inference"]="src/grn_inference/script.py"
)

set -e
# Define run flags
RUN_GRN=true
CELL_TYPE_GRANULARITY='major'
MAIN_DIR='/vol/projects/jnourisa/'

MAX_WORKERS=10


datasets=" perez_sle onek1k abf300 aida" # perez_sle onek1k abf300 aida soundlife

for dataset in $datasets; do
        data_type='sc'
        if [ "$dataset" = "soundlife" ] ; then
                data_type='bulk'
        fi
        if [ "$RUN_GRN" = true ]; then
                FORCE=true # If true, overwrite the existing files in grns directory
                SAVE_GRNS_DIR="${MAIN_DIR}/output/grns/${dataset}/${data_type}"
                

                args="  
                        --dataset $dataset \
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
