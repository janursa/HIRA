#!/bin/bash
#SBATCH --job-name=ciim_helper
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --mem=400GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL      
#SBATCH --mail-user=jalil.nourisa@gmail.com   

only_male=false
downsample=false
force=true
dataset="all"

# Initialize the command
cmd="python scripts/run_grn_inference.py --dataset ${dataset}"

# Append flags based on conditions
[ "$only_male" = true ] && cmd="${cmd} --only_male"
[ "$downsample" = true ] && cmd="${cmd} --downsample"
[ "$force" = true ] && cmd="${cmd} --force"

# Run the command
echo "Running: $cmd"
$cmd

