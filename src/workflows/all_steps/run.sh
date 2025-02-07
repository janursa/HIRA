#!/bin/bash
#SBATCH --job-name=all_steps
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=400GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL      
#SBATCH --mail-user=jalil.nourisa@gmail.com   

# Define run flags
RUN_PREPROCESS=false
RUN_PROCESS_DATASET=true
RUN_GRN=false

# datasets to include
DATASETS=('data1') #('data1' 'data2' 'data3' 'data4' 'data5' 'data7' 'data8' 'data9' 'data11') data7_allTPs

# Define others flags
DOWNSAMPLE=false
ONLY_MALE=false
FORCE=true # If true, overwrite the existing files in grns directory


RAW_DATASET_FILE="/vol/projects/jnourisa/datasets/data1_raw.h5ad"
FILTERED_DATASET_FILE="/vol/projects/jnourisa/datasets/data1_sc.h5ad" # tailors raw based on the given flags such as make, downsample, etc.
DATASET_BULK_FILE="/vol/projects/jnourisa/datasets/data1_bulk.h5ad" # tailors raw based on the given flags such as make, downsample, etc.

SAVE_GRNS_DIR="output/grns/all/"


# Initialize the command
cmd="python src/workflows/all_steps/script.py 
        --raw_dataset_file ${RAW_DATASET_FILE} 
        --processed_dataset_file ${FILTERED_DATASET_FILE} 
        --bulk_dataset_file ${DATASET_BULK_FILE}
        --save_grns_dir ${SAVE_GRNS_DIR}
        --datasets ${DATASETS[@]}"
  

# Append flags based on conditions
[ "$RUN_PREPROCESS" = true ] && cmd="${cmd} --run_preprocess"
[ "$RUN_PROCESS_DATASET" = true ] && cmd="${cmd} --run_process_dataset"
[ "$RUN_GRN" = true ] && cmd="${cmd} --run_grn"
[ "$DOWNSAMPLE" = true ] && cmd="${cmd} --downsample"
[ "$ONLY_MALE" = true ] && cmd="${cmd} --only_male"
[ "$FORCE" = true ] && cmd="${cmd} --force"


# Run the command
echo "Running (bash): $cmd"
$cmd