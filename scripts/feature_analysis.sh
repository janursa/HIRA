#!/bin/bash
#SBATCH --job-name=feature_analysis
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --mem=250GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL      
#SBATCH --mail-user=jalil.nourisa@gmail.com   

set -e
# echo "---------------------------------------------------------- Consensus networks -----------------------------------------------------------------"
# python src/feature_association/consensus_nets.py

feature_type="tfa_traj"
cell_types="CD8T"
data_type="sc"


# feature_type="tf_activity"
# cell_types="CD8T"
# data_type="bulk"

echo "---------------------------------------------------------- Aging -----------------------------------------------------------------"
# python src/feature_association/run_analysis.py --feature-type $feature_type --data-type $data_type --analysis-mode multi-cohort  --association-type continous --cell-types $cell_types 
python src/feature_association/post_aging_analysis.py --feature-type $feature_type --data-type $data_type 
# echo "---------------------------------------------------------- Soundlife -----------------------------------------------------------------"
# python src/feature_association/run_analysis.py --analysis-mode single-cohort --datasets soundlife --cell-types all --data-type $data_type --feature-type $feature_type --association-type continous 
# python src/feature_association/post_condition_analysis.py --dataset soundlife --analysis-type aging --data-type $data_type --feature-type $feature_type --cell-types all  --skip-pathway

# echo "---------------------------------------------------------- SLE -----------------------------------------------------------------"
# python src/feature_association/run_analysis.py  --datasets "perez_sle" --cell-types all --data-type $data_type --feature-type $feature_type --association-type grouped 
# python src/feature_association/post_condition_analysis.py --dataset "perez_sle" --analysis-type disease --data-type $data_type --feature-type $feature_type --cell-types all 

# echo "---------------------------------------------------------- parsebioscience -----------------------------------------------------------------"
# python src/feature_association/run_analysis.py  --dataset parsebioscience --cell-types all --data-type $data_type --feature-type $feature_type --association-type grouped 
# python src/feature_association/post_condition_analysis.py --dataset parsebioscience --analysis-type perturbation --data-type $data_type --feature-type $feature_type --cell-types all --agreement opposite

# echo "---------------------------------------------------------- op -----------------------------------------------------------------"
# python src/feature_association/run_analysis.py  --dataset op --cell-types all --data-type $data_type --feature-type $feature_type --association-type grouped 
# python src/feature_association/post_condition_analysis.py --dataset op --analysis-type perturbation --data-type $data_type --feature-type $feature_type --cell-types all --agreement opposite

# echo "---------------------------------------------------------- CXCL9 -----------------------------------------------------------------"
# python src/feature_association/run_analysis.py --dataset CXCL9 --cell-types CD8T --data-type $data_type --feature-type $feature_type --association-type grouped 
# python src/feature_association/post_condition_analysis.py --dataset CXCL9 --analysis-type perturbation --data-type $data_type --feature-type $feature_type --cell-types CD4T CD8T --agreement opposite --skip-overview --skip-pathway


# echo "---------------------------------------------------------- compare IL10 effects to Ruxolitinib -----------------------------------------------------------------"
# python src/feature_association/compare_il10_ruxolitinib.py --feature-type $feature_type --data-type $data_type --cell-types CD4T