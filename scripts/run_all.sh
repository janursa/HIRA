#!/bin/bash
#SBATCH --job-name=all_steps
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00:00
#SBATCH --mem=1000GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL      
#SBATCH --mail-user=jalil.nourisa@gmail.com   

declare -A dependencies

dependencies=(
    ["grn_inference"]="/home/jnourisa/projs/ongoing/ciim/src/workflows/grn_inference/script.py"
    ["process_dataset"]="/home/jnourisa/projs/ongoing/ciim/src/process_dataset/preprocess/script.py"
    ["alis_code"]="/home/jnourisa/projs/ongoing/ciim/src/process_dataset/ali/script.py"
    ["bulkify_code"]="/home/jnourisa/projs/ongoing/ciim/src/process_dataset/bulkify/script.py"
    
)

set -e
# Define run flags
RUN_ALIS_CODE=false
RUN_PROCESS_DATASET=false
RUN_PSEUDOBULK=true
RUN_GRN=false
RUN_ASSOCIATION=false

MAX_WORKERS=10
data_type='sc'


# datasets to include -> preprocessing 
datasets="CXCL9 " #data12_CMtx data7_allTPs_jalil_CMtx data1_CMtx data13_CMtx  SLE data13_Korean_CMtx  data13_Japanese_CMtx 

for dataset in $datasets; do
        # Define the command
        if [ "$RUN_ALIS_CODE" = true ]; then
                args="--dataset_name $dataset --save_dir /vol/projects/jnourisa/datasets/"
                cmd="python ${dependencies["alis_code"]} $args"
                echo "Running (bash): $cmd"
                $cmd
        fi
done

datasets="  CXCL9 " #CXCL9  data1 data12 data7_allTPs_jalil   data13_Korean  data13_Japanese SLE_Asian  SLE_European


for dataset in $datasets; do
        DATASETS=($dataset) #('data1' 'data2' 'data3' 'data4' 'data5' 'data7' 'data8' 'data9' 'data10' 'data11') data7_allTPs_jalil
        RAW_DATASET_FILE="/vol/projects/jnourisa/datasets/${dataset}_raw.h5ad"
        PROCESSED_DATASET_FILE="/vol/projects/jnourisa/datasets/${dataset}_sc.h5ad" # tailors raw based on the given flags such as make, downsample, etc.
        BULK_ALL="/vol/projects/jnourisa/datasets/${dataset}_bulk.h5ad"
        BULK_MINOR_CELLTYPE="/vol/projects/jnourisa/datasets/${dataset}_bulk_minor.h5ad"
        BULK_M="/vol/projects/jnourisa/datasets/${dataset}_bulk_M.h5ad"
        BULK_F="/vol/projects/jnourisa/datasets/${dataset}_bulk_F.h5ad"


        
        if [ "$RUN_PROCESS_DATASET" = true ]; then
                # set the flags
                DOWNSAMPLE=false

                args="--raw_dataset_file $RAW_DATASET_FILE \
                        --processed_dataset_file $PROCESSED_DATASET_FILE "
                [ "$DOWNSAMPLE" = true ] && args="${args} --downsample"
                cmd="python ${dependencies["process_dataset"]} $args"
                echo "Running (bash): $cmd"
                $cmd
        fi
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
                SAVE_GRNS_DIR="output/grns/${dataset}/"
                
                DATASET_FILE="/vol/projects/jnourisa/datasets/${dataset}_${data_type}.h5ad" # tailors raw based on the given flags such as make, downsample, etc.

                args="  
                        --dataset_file $DATASET_FILE \
                        --save_grns_dir $SAVE_GRNS_DIR \
                        --max_workers $MAX_WORKERS \
                        --data_type $data_type \
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
