#!/bin/bash
# Generates/acquires files in PRIOR_DIR (see config.py). For files with a known
# generator this builds them; for externally-sourced files with no known generator
# it prints where to get them manually.
#
# Usage: bash scripts/prior/acquire.sh <file>
# Files: gene_names aging_hallmark_genes disease_gene_association skeleton_promotor
#        skeleton_atac collectri tf_all hallmark essential_genes gene_aging_mechanisms

set -e

source scripts/_env.sh

TASK_GRN_REPO="${TASK_GRN_BENCHMARK_DIR:?set TASK_GRN_BENCHMARK_DIR in .env}"
PRIOR_DIR=$(python3 -c "import sys; sys.path.insert(0, 'src'); from config import PRIOR_DIR; print(PRIOR_DIR)")

file="$1"

case "$file" in
  gene_names)
    python3 src/process_data/prior/build_gene_names.py
    ;;
  aging_hallmark_genes)
    python3 src/process_data/prior/build_aging_hallmark_genes.py
    ;;
  disease_gene_association)
    python3 src/process_data/prior/build_disease_gene_association.py
    ;;
  skeleton_promotor)
    python3 src/process_data/prior/build_skeleton_promotor.py
    ;;
  skeleton_atac)
    # ATAC+motif skeleton (OP PBMC multiome). Rebuilding needs scglue + a GPU node:
    #   singularity exec --nv <scglue image> \
    #     python3 src/process_data/prior/build_skeleton_atac.py --dataset op
    # The built file already exists upstream, so copy it and add the `edge` column.
    python3 -c "
import sys, pandas as pd
sys.path.insert(0, 'src'); from config import PRIOR_DIR
d = pd.read_csv('${TASK_GRN_REPO}/resources/grn_benchmark/prior/skeleton_op.csv', index_col=0)
d = d[['source', 'target']].drop_duplicates()
d['edge'] = d['source'] + '_' + d['target']
d.to_csv(f'{PRIOR_DIR}/skeleton_atac.csv', index=False)
"
    ;;
  collectri)
    python3 -c "from hira.src.utils.util import flesh_out_collectri; flesh_out_collectri()"
    ;;
  tf_all)
    # Sourced from the sibling task_grn_inference repo; no independent generator here.
    cp "${TASK_GRN_REPO}/resources/grn_benchmark/prior/tf_all.csv" "${PRIOR_DIR}/tf_all.csv"
    ;;
  hallmark)
    # Fetches MSigDB_Hallmark_2020 via gseapy and caches it to PRIOR_DIR.
    python3 -c "from hira.src.pathway_analysis.util import get_hallmark; get_hallmark()"
    ;;
  essential_genes)
    cat <<EOF
No known generator for CRISPRInferredCommonEssentials.csv.
Source manually from DepMap (depmap.org, "CRISPRInferredCommonEssentials" gene effect file)
and place it at ${PRIOR_DIR}/CRISPRInferredCommonEssentials.csv
EOF
    ;;
  gene_aging_mechanisms)
    cat <<EOF
No known generator or source for gene-aging-mechanisms.tsv -- place a manually curated
copy at ${PRIOR_DIR}/gene-aging-mechanisms.tsv
EOF
    ;;
  *)
    echo "Usage: bash scripts/prior/acquire.sh <gene_names|aging_hallmark_genes|disease_gene_association|skeleton_promotor|skeleton_atac|collectri|tf_all|hallmark|essential_genes|gene_aging_mechanisms>"
    exit 1
    ;;
esac
