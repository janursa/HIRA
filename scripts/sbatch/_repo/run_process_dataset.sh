#!/bin/bash
#SBATCH --job-name=process_dataset
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --mem=400GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL      
#SBATCH --mail-user=jalil.nourisa@gmail.com   

only_male=false
downsample=false
input_dataset_file="/vol/projects/HIARA/Healthy_Single_Cell_Data/output/processed_data/processed_data.h5ad"
processed_dataset_file='/vol/projects/jnourisa/adata_only_male_downsampled.h5ad'


# Initialize the command
cmd="python src/process_dataset/script.py --input_dataset_file ${input_dataset_file} --processed_dataset_file ${processed_dataset_file}"

# Append flags based on conditions
[ "$only_male" = true ] && cmd="${cmd} --only_male"
[ "$downsample" = true ] && cmd="${cmd} --downsample"

# Run the command
echo "Running: $cmd"
$cmd