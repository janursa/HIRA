#!/bin/bash
# Supplementary tables and figures. Needs stages 1-5 (see scripts/readme.md) to have run:
# activation_vs_expression additionally needs the ge_major_b feature analysis.
# Usage: bash scripts/supp_figs.sh [analysis_name]
set -e

[ -f .env ] && set -a && source .env && set +a

analysis_name="${1:-tfa_major_b}"

mkdir -p "$(python -c "import sys; sys.path.insert(0, 'src'); from config import PLOTS_DIR; print(PLOTS_DIR)")"

echo "---------------------------------------------------------- Cohort stats -----------------------------------------------------------------"
python src/process_data/dataset_stats.py

echo "---------------------------------------------------------- GRN overlap -----------------------------------------------------------------"
python src/grn_inference/plot_overlap.py

echo "---------------------------------------------------------- Discovery vs validation -----------------------------------------------------------------"
python src/feature_association/discovery_validation_tables.py --analysis-name "$analysis_name"

echo "---------------------------------------------------------- Activation vs expression -----------------------------------------------------------------"
python src/feature_association/activation_vs_expression.py --tfa-analysis "$analysis_name"
