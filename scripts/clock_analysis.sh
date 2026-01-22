set -e
# python src/feature_association/consensus_nets.py

# echo "--------------------------------------------------------------train clocks--------------------------------------------------------------"
# python src/clock/run_train.py

# echo "--------------------------------------------------------------cv--------------------------------------------------------------"
# python src/clock/run_cv.py

# echo "--------------------------------------------------------------comparision: sle --------------------------------------------------------------"
# python src/clock/clock_analysis.py --dataset perez_sle --analysis-type disease --cell-types CD4T CD8T


# echo "--------------------------------------------------------------comparision: parsebioscience --------------------------------------------------------------"
# python src/clock/clock_analysis.py --dataset parsebioscience --analysis-type perturbation --cell-types CD4T CD8T

echo "--------------------------------------------------------------comparision: op --------------------------------------------------------------"
python src/clock/clock_analysis.py --dataset op --analysis-type perturbation --cell-types CD4T CD8T

# # echo "--------------------------------------------------------------comparision: CLCX9 --------------------------------------------------------------"
# python src/clock/clock_analysis.py --dataset CXCL9 --analysis-type perturbation --cell-types CD4T


# echo "--------------------------------------------------------------comparision: soundlife --------------------------------------------------------------"
# python src/clock/clock_analysis.py --dataset soundlife --config-label aging_cmv_neg --analysis-type aging --cell-types CD4T CD8T 