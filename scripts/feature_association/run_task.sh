#!/bin/bash
#SBATCH --job-name=feature_analysis
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --mem=500GB
#SBATCH --partition=cpu

# One feature-association task. Submitted per task by wrapper_feature_analysis.sh.
# Usage: sbatch scripts/feature_association/run_task.sh <analysis_name> <task>
set -e

source scripts/_env.sh

analysis_name=$1
task=$2
test_mode="" #--test-mode

mkdir -p "$(python -c "import sys; sys.path.insert(0, 'src'); from config import PLOTS_DIR; print(PLOTS_DIR)")"

echo "---------------------------------------- $task ($analysis_name) ----------------------------------------"
case "$task" in
  aging)
    python src/feature_association/run_analysis.py --analysis-name $analysis_name --analysis-mode multi-cohort --association-type continous $test_mode
    python src/feature_association/post_aging_analysis.py --analysis-name $analysis_name
    ;;
  soundlife)
    python src/feature_association/run_analysis.py --analysis-mode single-cohort --datasets soundlife --analysis-name $analysis_name --association-type continous $test_mode
    python src/feature_association/post_condition_analysis.py --dataset soundlife --analysis-type aging --analysis-name $analysis_name
    ;;
  perez_sle)
    python src/feature_association/run_analysis.py --datasets perez_sle --analysis-name $analysis_name --association-type grouped
    python src/feature_association/post_condition_analysis.py --dataset perez_sle --analysis-type disease --analysis-name $analysis_name
    ;;
  parsebioscience)
    python src/feature_association/run_analysis.py --dataset parsebioscience --cell-types CD4T CD8T NK B MONO --analysis-name $analysis_name --association-type grouped
    python src/feature_association/post_condition_analysis.py --dataset parsebioscience --analysis-type perturbation --analysis-name $analysis_name --cell-types CD4T CD8T NK B MONO --agreement opposite
    ;;
  op)
    python src/feature_association/run_analysis.py --dataset op --cell-types CD4T CD8T NK B --analysis-name $analysis_name --association-type grouped
    python src/feature_association/post_condition_analysis.py --dataset op --analysis-type perturbation --analysis-name $analysis_name --cell-types CD4T CD8T NK B --agreement opposite
    ;;
  CXCL9)
    python src/feature_association/run_analysis.py --dataset CXCL9 --cell-types CD4T CD8T --analysis-name $analysis_name --association-type grouped
    python src/feature_association/post_condition_analysis.py --dataset CXCL9 --analysis-type perturbation --analysis-name $analysis_name --cell-types CD4T CD8T --agreement opposite --skip-overview
    ;;
  il10_ruxolitinib)
    # needs op + parsebioscience to have finished
    python src/feature_association/compare_il10_ruxolitinib.py --analysis-name $analysis_name --cell-types CD4T
    ;;
  *)
    echo "unknown task: $task" >&2; exit 1
    ;;
esac
