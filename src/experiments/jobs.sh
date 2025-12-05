
set -euo pipefail

python src/experiments/post_condition_analysis.py \
    --dataset op \
    --analysis-type perturbation \
    --data-type bulk \
    --feature-type tf_activity  \
    --cell-types CD4T CD8T

bash scripts/experiment/run_condition_analysis.sh \
    --dataset CXCL9 \
    --cell-types "CD4T CD8T" \
    --data-type sc \
    --feature-type tf_activity

python src/experiments/post_condition_analysis.py \
    --dataset CXCL9 \
    --analysis-type perturbation \
    --data-type sc \
    --feature-type tf_activity \
    --cell-types CD4T CD8T


python src/experiments/post_condition_analysis.py \
    --dataset soundlife \
    --analysis-type perturbation \
    --data-type bulk \
    --feature-type tf_activity \
    --cell-types CD4T CD8T

python src/experiments/post_condition_analysis.py \
  --dataset soundlife \
  --config-label aging_cmv_neg \
  --analysis-type aging \
  --data-type bulk \
  --feature-type tf_activity \
  --cell-types CD4T CD8T

python src/experiments/post_condition_analysis.py \
  --dataset soundlife \
  --config-label cmv \
  --analysis-type aging \
  --data-type bulk \
  --feature-type tf_activity \
  --cell-types CD4T CD8T