#!/bin/bash
#SBATCH --job-name=grn_inference
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=2:00:00
#SBATCH --mem=1000GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL      
#SBATCH --mail-user=jalil.nourisa@gmail.com   

declare -A dependencies

dependencies=(
    ["grn_inference"]="src/grn_inference/script.py"
)

set -e

# Load repo-level config (HIRA_DIR, HIRA_BASE_DIR, ...) if present
[ -f .env ] && set -a && source .env && set +a

# Parse command line arguments
DATASET=$1
FORCE=${2:-true}
CELL_TYPE_GRANULARITY=${3:-'major'}
# Defaults to config.py's GRNS_DIR (repo output/grns, or $HIRA_DIR/output/grns), so this
# lines up with where retrieve_net()/retrieve_net_consensus() read GRNs from downstream.
GRNS_DIR=${4:-$(python -c "import sys; sys.path.insert(0, 'src'); from config import GRNS_DIR; print(GRNS_DIR)")}
MAX_WORKERS=${5:-5}

# Determine data type based on dataset
data_type='sc'
if [ "$DATASET" = "soundlife" ] ; then
    data_type='bulk'
fi

# Run GRN inference
SAVE_GRNS_DIR="${GRNS_DIR}/${DATASET}/${data_type}"

args="  
    --dataset $DATASET \
    --save_grns_dir $SAVE_GRNS_DIR \
    --num_workers $MAX_WORKERS \
    --data_type $data_type \
    --cell_type_granularity $CELL_TYPE_GRANULARITY \
    "

[ "$FORCE" = true ] && args="${args} --force"

cmd="python ${dependencies["grn_inference"]} $args"
echo "Running (bash): $cmd"
$cmd
