#!/bin/bash
# Submits one SLURM job per feature-association task; they run in parallel.
# Usage: bash scripts/feature_association/wrapper_feature_analysis.sh [analysis_name] [task ...]
set -e

[ -f .env ] && set -a && source .env && set +a

analysis_name="${1:-tfa_major_b}" # tfa_major_b ge_major_mc ge_major_b ct_tf_markers tfa_sub_b ct_freq ct_pol_dist tfa_peg ccc_major_b ccc_sub_b
shift || true
tasks=("$@")
if [ ${#tasks[@]} -eq 0 ]; then
    tasks=(aging soundlife perez_sle parsebioscience op CXCL9)
    # only the aging task has a condition-plot implementation for the ge_* feature types
    # (post_condition_analysis.py -> wrapper_plots_gene_expression_condition raises)
    case "$analysis_name" in ge_*) tasks=(aging) ;; esac
fi

WORKER="scripts/feature_association/run_task.sh"
REF_GE=$(python -c "import sys; sys.path.insert(0, 'src'); from config import REF_GE_ANALYSIS; print(REF_GE_ANALYSIS)")

# consensus GRNs are shared by every task -- build once, up front
python src/feature_association/consensus_nets.py

# condition tasks read the aging task's stats_multi_cohort.csv, so aging goes first
declare -A jid
aging_dep=""
for task in "${tasks[@]}"; do
    [ "$task" = aging ] || continue
    out=$(sbatch --parsable $WORKER "$analysis_name" aging)
    jid[aging]=$out
    aging_dep="--dependency=afterok:$out"
    echo "submitted aging -> $out"
done

for task in "${tasks[@]}"; do
    [ "$task" = aging ] && continue
    out=$(sbatch --parsable $aging_dep $WORKER "$analysis_name" "$task")
    jid[$task]=$out
    echo "submitted $task -> $out ${aging_dep:+(after ${jid[aging]})}"
done

# il10_ruxolitinib reads the op and parsebioscience results, so it waits for both
dep=""
for t in op parsebioscience; do
    [ -n "${jid[$t]}" ] && dep="${dep}:${jid[$t]}"
done
if [ -n "$dep" ]; then
    out=$(sbatch --parsable --dependency=afterok"$dep" $WORKER "$analysis_name" il10_ruxolitinib)
    echo "submitted il10_ruxolitinib -> $out (after$dep)"
fi

# supp_figs.sh (activation_vs_expression) needs REF_GE_ANALYSIS alongside the TFA run
if [ "$analysis_name" != "$REF_GE" ]; then
    out=$(sbatch --parsable $WORKER "$REF_GE" aging)
    echo "submitted $REF_GE aging -> $out"
fi
