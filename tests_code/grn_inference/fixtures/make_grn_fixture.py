"""Run once by hand to (re)generate data/grn/baseline_net_<CT>.csv.

Reuses ../../preprocessing/data/preprocessing/baseline_sc.h5ad as input (same small annotated
fixture, not duplicated) and mirrors wrapper_grn's steps (src/grn_inference/script.py) exactly,
just called directly on the fixture instead of going through retrieve_adata.
"""
import os
import sys

import anndata as ad
import scanpy as sc

REPRO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIRA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(REPRO_DIR)))
sys.path.insert(0, HIRA_DIR)

from hira.src.utils.util import basic_qc, filter_rb_mt_genes  # noqa: E402
from hira.src.grn_inference.inference import main as main_inference  # noqa: E402
from hira.src.config import get_config, MAJOR_CT_LABEL  # noqa: E402

INPUT_PATH = os.path.join(os.path.dirname(REPRO_DIR), 'preprocessing', 'data', 'preprocessing', 'baseline_sc.h5ad')
OUT_DIR = os.path.join(REPRO_DIR, 'data', 'grn')
DATASET = 'wang'
TOP_N_EDGES = 100_000
MIN_GENES_PER_CELL = 10
MAX_GENES_PER_CELL = 5000
MIN_CELLS_PER_GENE = 20  # lower than production's 500: fixture only has ~hundreds of cells per cell type
# Spearman correlation is O(n_genes^2); cap the gene set so the fixture (and every test rerun of
# it) finishes in seconds rather than minutes. Real inference logic/code path is unaffected —
# only how many genes get fed into it.
MAX_GENES = 300


def infer_for_cell_type(adata_full, cell_type, bulk_group):
    adata = adata_full[adata_full.obs[MAJOR_CT_LABEL] == cell_type].copy()
    print(f'{cell_type}: {adata.shape[0]} cells before sampling', flush=True)

    sampled_indices = []
    for _, group_df in adata.obs.groupby(bulk_group, sort=False):
        n = min(5000, len(group_df))
        sampled_indices.extend(group_df.sample(n=n, random_state=0).index.tolist())
    adata = adata[sampled_indices].copy()
    adata = filter_rb_mt_genes(adata)

    adata = basic_qc(adata, min_genes_per_cell=MIN_GENES_PER_CELL, max_genes_per_cell=MAX_GENES_PER_CELL,
                      min_cells_per_gene=MIN_CELLS_PER_GENE)
    if adata.n_obs < 10 or adata.n_vars < 10:
        print(f'{cell_type}: too few cells/genes after QC ({adata.shape}), skipping', flush=True)
        return None
    if adata.n_vars > MAX_GENES:
        total_counts = adata.X.sum(axis=0)
        total_counts = getattr(total_counts, 'A1', total_counts).ravel()
        top_genes = adata.var_names[total_counts.argsort()[::-1][:MAX_GENES]]
        adata = adata[:, sorted(top_genes)].copy()

    X_norm = sc.pp.normalize_total(adata, inplace=False)['X']
    X_norm = sc.pp.log1p(X_norm, copy=True)

    net = main_inference(X_norm, adata.var_names)
    net['cell_type'] = cell_type
    net['sample_size'] = adata.shape[0]
    net['gene_size'] = adata.shape[1]
    # mirrors wrapper_grn: truncate within each motif-support class, not globally
    top = lambda d: d.sort_values(by='weight', ascending=False, key=abs).head(TOP_N_EDGES).index
    net = net.loc[top(net).union(top(net[net['skeleton_based']])).union(top(net[net['promotor_based']]))]
    print(f'{cell_type}: inferred network shape {net.shape}', flush=True)
    return net


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    adata_full = ad.read_h5ad(INPUT_PATH)
    bulk_group = get_config(DATASET).bulk_group
    adata_full.obs['bulk_group'] = adata_full.obs[bulk_group].astype(str).agg('_'.join, axis=1)

    for cell_type in sorted(adata_full.obs[MAJOR_CT_LABEL].unique()):
        net = infer_for_cell_type(adata_full, cell_type, 'bulk_group')
        if net is None or net.empty:
            # ponytail: an empty net is a degenerate baseline (no significant edges at this
            # fixture size) -- skip it rather than commit a header-only CSV to diff against.
            print(f'{cell_type}: no edges survived, not writing a baseline', flush=True)
            continue
        net.to_csv(os.path.join(OUT_DIR, f'baseline_net_{cell_type}.csv'), index=False)


if __name__ == '__main__':
    main()
