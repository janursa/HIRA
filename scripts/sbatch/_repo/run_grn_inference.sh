#!/bin/bash
#SBATCH --job-name=grn_inference
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --mem=400GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL      
#SBATCH --mail-user=jalil.nourisa@gmail.com   

only_male=true
downsample=false
force=false
dataset="pbmc_ageing"
save_adata=true

        save_name = f'/vol/projects/jnourisa/{dataset}_only_male_{only_male}_downsample_{downsample}.h5ad'

# Initialize the command
cmd="python scripts/run_grn_inference.py --dataset ${dataset}"

# Append flags based on conditions
[ "$only_male" = true ] && cmd="${cmd} --only_male"
[ "$downsample" = true ] && cmd="${cmd} --downsample"
[ "$force" = true ] && cmd="${cmd} --force"
[ "$save_adata" = true ] && cmd="${cmd} --save_adata"

# Run the command
echo "Running: $cmd"
$cmd

