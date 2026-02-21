#!/bin/bash
#SBATCH --job-name=feature_analysis
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --mem=500GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL      
#SBATCH --mail-user=jalil.nourisa@gmail.com   

set -e

# echo "---------------------------------------------------------- Consensus networks -----------------------------------------------------------------"
# python src/feature_association/consensus_nets.py

analysis_name="tfa_major_b" #"tfa_major_b sub_tf_markers" #"tfa_sub_b" "sub_tf_markers" ct_pol_dist ccc_sub_b ct_freq
# cell_types="Naive_B Memory_B Tcm_Naive_CD8 Tem_Trm_CD8 Tem_Temra_CD8 MAIT Tcm_Naive_CD4 Tem_Effector_CD4 NonClassic_MONO Classic_MONO CD16_NK" # "Tcm_Naive_CD4"
test_mode="" #--test-mode

# echo "---------------------------------------------------------- Aging -----------------------------------------------------------------"
# python src/feature_association/run_analysis.py --analysis-name $analysis_name --analysis-mode multi-cohort  --association-type continous  $test_mode 
# python src/feature_association/post_aging_analysis.py --analysis-name $analysis_name --skip-pathway

# echo "---------------------------------------------------------- Soundlife -----------------------------------------------------------------"
# python src/feature_association/run_analysis.py --analysis-mode single-cohort --datasets soundlife   --analysis-name $analysis_name --association-type continous 
# python src/feature_association/post_condition_analysis.py --dataset soundlife --analysis-type aging --analysis-name $analysis_name    --skip-pathway

# echo "---------------------------------------------------------- SLE -----------------------------------------------------------------"
# python src/feature_association/run_analysis.py  --datasets "perez_sle"   --analysis-name $analysis_name --association-type grouped 
# python src/feature_association/post_condition_analysis.py --dataset "perez_sle" --analysis-type disease --analysis-name $analysis_name   --skip-pathway

echo "---------------------------------------------------------- parsebioscience -----------------------------------------------------------------"
python src/feature_association/run_analysis.py  --dataset parsebioscience   --analysis-name $analysis_name --association-type grouped 
python src/feature_association/post_condition_analysis.py --dataset parsebioscience --analysis-type perturbation --analysis-name $analysis_name   --agreement opposite --skip-pathway

# echo "---------------------------------------------------------- op -----------------------------------------------------------------"
# python src/feature_association/run_analysis.py  --dataset op   --analysis-name $analysis_name --association-type grouped 
# python src/feature_association/post_condition_analysis.py --dataset op --analysis-type perturbation --analysis-name $analysis_name   --agreement opposite --skip-pathway
# echo "---------------------------------------------------------- CXCL9 -----------------------------------------------------------------"
# python src/feature_association/run_analysis.py --dataset CXCL9 --cell-types CD4T CD8T --analysis-name $analysis_name --association-type grouped 
# python src/feature_association/post_condition_analysis.py --dataset CXCL9 --analysis-type perturbation --analysis-name $analysis_name --cell-types CD4T CD8T --agreement opposite --skip-overview --skip-pathway


# echo "---------------------------------------------------------- compare IL10 effects to Ruxolitinib -----------------------------------------------------------------"
# python src/feature_association/compare_il10_ruxolitinib.py --analysis-name $analysis_name --cell-types CD4T