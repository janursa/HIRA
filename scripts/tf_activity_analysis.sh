# make pipeline stop in case of error
set -e

# python src/feature_association/run_analysis.py --feature-type tf_activity --data-type bulk --analysis-mode multi-cohort  --association-type continous
# python src/experiments/post_aging_analysis.py --feature-type tf_activity 

# python src/feature_association/run_analysis.py --analysis-mode single-cohort --datasets soundlife --cell-types CD8T CD4T MONO NK --data-type bulk --feature-type tf_activity --association-type continous 
# python src/experiments/post_condition_analysis.py --dataset soundlife --analysis-type aging --data-type bulk --feature-type tf_activity --cell-types CD8T CD4T MONO NK  --skip-overview --skip-pathway

# python src/feature_association/run_analysis.py  --datasets "perez_sle" --cell-types CD4T CD8T --data-type bulk --feature-type tf_activity --association-type grouped 
# python src/experiments/post_condition_analysis.py --dataset "perez_sle" --analysis-type disease --data-type bulk --feature-type tf_activity --cell-types CD8T CD4T --skip-pathway


# python src/feature_association/run_analysis.py  --dataset parsebioscience --cell-types CD4T CD8T --data-type bulk --feature-type tf_activity --association-type grouped
# python src/experiments/post_condition_analysis.py --dataset parsebioscience --analysis-type perturbation --data-type bulk --feature-type tf_activity --cell-types CD4T CD8T --agreement opposite --skip-overview  --skip-pathway


python src/feature_association/run_analysis.py  --dataset op --cell-types CD8T --data-type bulk --feature-type tf_activity --association-type grouped 
python src/experiments/post_condition_analysis.py --dataset op --analysis-type perturbation --data-type bulk --feature-type tf_activity --cell-types CD8T --agreement opposite

# python src/feature_association/run_analysis.py --dataset CXCL9 --cell-types CD8T --data-type bulk --feature-type tf_activity --association-type grouped 
# python src/experiments/post_condition_analysis.py --dataset CXCL9 --analysis-type perturbation --data-type bulk --feature-type tf_activity --cell-types CD8T --agreement opposite --skip-overview --skip-pathway
