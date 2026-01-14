# python src/feature_association/run_analysis.py --feature-type tf_activity --data-type bulk --analysis-mode multi-cohort  --association-type continous
# python src/experiments/post_aging_analysis.py --feature-type tf_activity --skip-pathway


python src/feature_association/run_analysis.py --dataset CXCL9 --cell-types CD4T --data-type sc --feature-type tf_activity --association-type grouped 
python src/experiments/post_condition_analysis.py --dataset CXCL9 --analysis-type perturbation --data-type sc --feature-type tf_activity --cell-types CD4T --agreement opposite --skip-overview --skip-pathway
