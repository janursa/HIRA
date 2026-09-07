#!/bin/bash
# All explanatory/supplementary/stress analyses in one place. Not part of the main
# pipeline chain (see README.md) -- each reads outputs from stages 1-6.
# activation_vs_expression additionally needs the gene-expression analysis named by REF_GE_ANALYSIS.
# Stress tests are sbatch-submitted, not run inline -- they take 20-30h each.
# Usage: bash scripts/exp_analysis.sh [analysis_name] [-- confounders_args...]
set -e

[ -f .env ] && set -a && source .env && set +a

analysis_name="${1:-tfa_major_b}"
shift || true

mkdir -p "$(python -c "import sys; sys.path.insert(0, 'src'); from config import PLOTS_DIR; print(PLOTS_DIR)")"

echo "---------------------------------------------------------- Cohort stats -----------------------------------------------------------------"
python src/process_data/dataset_stats.py

echo "---------------------------------------------------------- GRN overlap -----------------------------------------------------------------"
python src/grn_inference/plot_overlap.py

echo "---------------------------------------------------------- Discovery vs validation -----------------------------------------------------------------"
python src/feature_association/discovery_validation_tables.py --analysis-name "$analysis_name"

echo "---------------------------------------------------------- Activation vs expression -----------------------------------------------------------------"
python src/feature_association/activation_vs_expression.py --tfa-analysis "$analysis_name"

echo "---------------------------------------------------------- Cohort age-confounders -----------------------------------------------------------------"
python src/exp_analysis/confounders.py "$@"

echo "---------------------------------------------------------- Clock stress tests (sbatch) -----------------------------------------------------------------"
for v in baseline gradientboosting nn wholegenome; do
    sbatch scripts/exp_analysis/run_clock_stress.sh "$v"
    echo "Submitted: $v"
done
sbatch --mem=500GB --time=30:00:00 scripts/exp_analysis/run_clock_stress.sh metacell
echo "Submitted: metacell"
echo "After all variants finish: python src/exp_analysis/clock_stress.py --aggregate"
