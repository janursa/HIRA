"""Pseudobulk an annotated single-cell dataset per donor x cell type.

Stage 2 of scripts/process_data/run_preprocess.sh; see --help for arguments.
Writes: <HIRA_BASE_DIR>/datasets/{bulk,bulk_minor}/<dataset>.h5ad
"""
import scanpy as sc
import argparse
import anndata as ad
import numpy as np
import pandas as pd
from hira.src.config import get_config, SUB_CT_LABEL, MAJOR_CT_LABEL, EXCLUDE_RB_MT_GENES
from hira.src.utils.util import bulkify_func, filter_rb_mt_genes

def normalize(adata):
    # keep the summed raw counts: CPM alone is composition-biased (a few strongly induced
    # genes dilute everything else), so DE needs TMM/median-of-ratios or voom/DESeq2 upstream
    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)
    return adata

def qc_bulk(adata, run_test=False):
    if run_test:
        cell_t = 1
    else:
        cell_t = 10
    # filter out bulk samples with less than cell_t cells
    low_cells = adata.obs['cell_count'] < cell_t
    print(f'Dropping {low_cells.sum()} bulk samples with less than {cell_t} cells', flush=True)
    # bulkify_func's internal filtering demotes obs_names from RangeIndex to a plain int64
    # Index, which current anndata rejects for boolean indexing (assert index.dtype != int)
    adata.obs_names = adata.obs_names.astype(str)
    adata = adata[~low_cells].copy()
    return adata

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sc_dataset_file', 
        type=str,
        required=True,
        help="Processed dataset file"
        )
    parser.add_argument('--bulk_all',
        type=str,
        required=True,
        help="Bulk data for all cell types"
        )
    parser.add_argument('--bulk_minor_celltype',
        type=str,
        required=True,
        help="Bulk data for minor cell types"
        )
    parser.add_argument('--run-test', 
        action='store_true',
        help="Whether to run in test mode (subset of data)"
        )

    args = parser.parse_args()
    adata = ad.read_h5ad(args.sc_dataset_file)
    if EXCLUDE_RB_MT_GENES:
        adata = filter_rb_mt_genes(adata)
    dataset = adata.obs['dataset'].unique()[0]
    cfg = get_config(dataset)
    bulk_group = cfg.bulk_group
    covariate_major = bulk_group + [MAJOR_CT_LABEL]

    # - main bulk data (per major celltype)
    print('Bulkifying main cell types')
    adata_bulk_major_celltypes = bulkify_func(adata, covariates=covariate_major)

    # per-donor x major-CT counts of each minor cell type, for downstream cell-type-ratio
    # covariates (e.g. naive/effector shift). Uses the major-CT group key ('sum_by') that
    # bulkify_func just set on adata.obs, before the minor-CT call below overwrites it.
    minor_counts = (
        adata.obs.assign(sum_by=adata.obs['sum_by'].astype(str))
        .groupby(['sum_by', SUB_CT_LABEL], observed=True)
        .size()
        .unstack(fill_value=0)
        .add_suffix('_count')
    )
    adata_bulk_major_celltypes.obs = adata_bulk_major_celltypes.obs.join(minor_counts, on='sum_by')
    adata_bulk_major_celltypes.obs[minor_counts.columns] = adata_bulk_major_celltypes.obs[minor_counts.columns].fillna(0).astype(int)

    adata_bulk_major_celltypes = normalize(adata_bulk_major_celltypes)
    adata_bulk_major_celltypes = qc_bulk(adata_bulk_major_celltypes, run_test=args.run_test)
    print(f'Writing bulk data for major cell types {adata_bulk_major_celltypes.shape} to {args.bulk_all}', flush=True)
    adata_bulk_major_celltypes.write(args.bulk_all)

    # - bulk minor
    print('Bulkifying minor cell types')
    covariates_minor = bulk_group + [SUB_CT_LABEL]
    adata_bulk_minor_celltypes = bulkify_func(adata, covariates=covariates_minor)
    adata_bulk_minor_celltypes = normalize(adata_bulk_minor_celltypes)
    adata_bulk_minor_celltypes = qc_bulk(adata_bulk_minor_celltypes, run_test=args.run_test)
    print(f'Writing bulk data for minor cell types {adata_bulk_minor_celltypes.shape} to {args.bulk_minor_celltype}', flush=True)
    adata_bulk_minor_celltypes.write(args.bulk_minor_celltype)

    print('DONE')


    