set -e
# echo "---------------------------------------------------------- Consensus networks -----------------------------------------------------------------"
# python src/feature_association/consensus_nets.py

echo "---------------------------------------------------------- Aging -----------------------------------------------------------------"
python src/feature_association/run_analysis.py --feature-type tf_activity --data-type bulk --analysis-mode multi-cohort  --association-type continous
python src/feature_association/post_aging_analysis.py --feature-type tf_activity

# echo "---------------------------------------------------------- Soundlife -----------------------------------------------------------------"
# python src/feature_association/run_analysis.py --analysis-mode single-cohort --datasets soundlife --cell-types all --data-type bulk --feature-type tf_activity --association-type continous 
# python src/feature_association/post_condition_analysis.py --dataset soundlife --analysis-type aging --data-type bulk --feature-type tf_activity --cell-types all  --skip-pathway

# echo "---------------------------------------------------------- SLE -----------------------------------------------------------------"
# python src/feature_association/run_analysis.py  --datasets "perez_sle" --cell-types all --data-type bulk --feature-type tf_activity --association-type grouped 
# python src/feature_association/post_condition_analysis.py --dataset "perez_sle" --analysis-type disease --data-type bulk --feature-type tf_activity --cell-types all 

# echo "---------------------------------------------------------- parsebioscience -----------------------------------------------------------------"
# python src/feature_association/run_analysis.py  --dataset parsebioscience --cell-types all --data-type bulk --feature-type tf_activity --association-type grouped 
# python src/feature_association/post_condition_analysis.py --dataset parsebioscience --analysis-type perturbation --data-type bulk --feature-type tf_activity --cell-types all --agreement opposite

# echo "---------------------------------------------------------- op -----------------------------------------------------------------"
# python src/feature_association/run_analysis.py  --dataset op --cell-types all --data-type bulk --feature-type tf_activity --association-type grouped 
# python src/feature_association/post_condition_analysis.py --dataset op --analysis-type perturbation --data-type bulk --feature-type tf_activity --cell-types all --agreement opposite

# echo "---------------------------------------------------------- CXCL9 -----------------------------------------------------------------"
# python src/feature_association/run_analysis.py --dataset CXCL9 --cell-types CD4T CD8T --data-type bulk --feature-type tf_activity --association-type grouped 
# python src/feature_association/post_condition_analysis.py --dataset CXCL9 --analysis-type perturbation --data-type bulk --feature-type tf_activity --cell-types CD4T CD8T --agreement opposite --skip-overview --skip-pathway


# echo "---------------------------------------------------------- compare IL10 effects to Ruxolitinib -----------------------------------------------------------------"
# python src/feature_association/compare_il10_ruxolitinib.py --feature-type tf_activity --data-type bulk --cell-types CD4T