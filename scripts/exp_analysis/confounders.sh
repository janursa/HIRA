set -e

[ -f .env ] && set -a && source .env && set +a

echo "--------------------------------------------------------------cohort age-confounders--------------------------------------------------------------"
python src/exp_analysis/confounders.py "$@"
