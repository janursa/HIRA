#!/bin/bash
# External benchmark: the published scImmuAging clocks (Li et al., Nat Aging 2025) run on
# HIRA's clock test cohorts. Pseudocells are built with hira.sif, prediction runs in
# scimmuaging.sif (R). Build it once: bash singularity/build.sh singularity/scimmuaging.sif scimmuaging.def
#
# Usage: bash scripts/clock_benchmark/script.sh [cohort ...]   (default: CLOCK_TEST_COHORTS)

set -e

source scripts/_env.sh

SIF=${SCIMMUAGING_SIF:-$HIRA_DIR/singularity/scimmuaging.sif}
[ -f "$SIF" ] || { echo "$SIF not found -- bash singularity/build.sh singularity/scimmuaging.sif scimmuaging.def" >&2; exit 1; }

out="$HIRA_DIR/temp/clock_benchmark"    # per-cohort predictions: intermediates, collect.py merges them
work="$HIRA_BASE_DIR/clock_benchmark"   # pseudocell matrices are heavy and disposable
mkdir -p "$out" "$work"

cohorts=${@:-$(python -c "from hira.src.config import CLOCK_TEST_COHORTS; print(' '.join(CLOCK_TEST_COHORTS))")}

for ds in $cohorts; do
    for ct in CD4T CD8T MONO NK B; do
        echo "---------------- $ds / $ct ----------------"
        python src/clock_benchmark/export_input.py --dataset "$ds" --cell-type "$ct" --out "$work/${ds}_${ct}.tsv.gz"
        singularity exec -B "$HIRA_BINDS" "$SIF" Rscript src/clock_benchmark/predict.R \
            "$work/${ds}_${ct}.tsv.gz" "$ct" "$out/pred_${ds}_${ct}.tsv"
        mv "$work/${ds}_${ct}.tsv.gz.meta.csv" "$out/meta_${ds}_${ct}.csv"
        rm -f "$work/${ds}_${ct}.tsv.gz"
    done
done

python src/clock_benchmark/collect.py
