#!/bin/bash
#SBATCH --job-name=all_steps
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=2:00:00
#SBATCH --mem=250GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL      
#SBATCH --mail-user=jalil.nourisa@gmail.com   

declare -A dependencies

dependencies=(
    ["grn_inference"]="/home/jnourisa/projs/ongoing/ciim/src/workflows/grn_inference/script.py"
    ["process_dataset"]="/home/jnourisa/projs/ongoing/ciim/src/process_dataset/script.py"
    ["preprocess"]="/home/jnourisa/projs/ongoing/ciim/src/preprocess/script.py"
)


set -e
# Define run flags
RUN_PREPROCESS=false
RUN_PROCESS_DATASET=true
RUN_GRN=true


# datasets to include
datasets="data7_allTPs_jalil data1" 
genders="M F"

for gender in $genders; do
        GENDER=$gender
        for dataset in $datasets; do
                DATASETS=($dataset) #('data1' 'data2' 'data3' 'data4' 'data5' 'data7' 'data8' 'data9' 'data10' 'data11') data7_allTPs_jalil

                RAW_DATASET_FILE="/vol/projects/jnourisa/datasets/${dataset}_raw.h5ad"

                # Define the command
                if [ "$RUN_PREPROCESS" = true ]; then
                        args="--datasets $datasets --raw_dataset_file $RAW_DATASET_FILE"
                        cmd="python ${dependencies["preprocess"]} $args"
                        echo "Running (bash): $cmd"
                        $cmd
                fi
                if [ "$RUN_PROCESS_DATASET" = true ]; then
                        # set the flags
                        DOWNSAMPLE=false
                        

                        if [ "$GENDER" = both ]; then
                                PROCESSED_DATASET_FILE="/vol/projects/jnourisa/datasets/${dataset}_sc.h5ad" # tailors raw based on the given flags such as make, downsample, etc.
                                DATASET_BULK_FILE="/vol/projects/jnourisa/datasets/${dataset}_bulk.h5ad" # tailors raw based on the given flags such as make, downsample, etc.
                        else
                                PROCESSED_DATASET_FILE="/vol/projects/jnourisa/datasets/${dataset}_sc_${GENDER}.h5ad" # tailors raw based on the given flags such as make, downsample, etc.
                                DATASET_BULK_FILE="/vol/projects/jnourisa/datasets/${dataset}_bulk_${GENDER}.h5ad" # tailors raw based on the given flags such as make, downsample, etc.
                        fi

                        args="--raw_dataset_file $RAW_DATASET_FILE \
                                --processed_dataset_file $PROCESSED_DATASET_FILE \
                                --bulk_dataset_file $DATASET_BULK_FILE \
                                --gender $GENDER "
                        [ "$DOWNSAMPLE" = true ] && args="${args} --downsample"
                        cmd="python ${dependencies["process_dataset"]} $args"
                        echo "Running (bash): $cmd"
                        $cmd
                fi
                if [ "$RUN_GRN" = true ]; then
                        FORCE=true # If true, overwrite the existing files in grns directory
                        SAVE_GRNS_DIR="output/grns/${dataset}/"
                        
                        args="  
                                --dataset_file $PROCESSED_DATASET_FILE \
                                --save_grns_dir $SAVE_GRNS_DIR 
                                "

                        [ "$FORCE" = true ] && args="${args} --force"

                        cmd="python ${dependencies["grn_inference"]} $args"
                        echo "Running (bash): $cmd"
                        $cmd
                fi

        done
done
