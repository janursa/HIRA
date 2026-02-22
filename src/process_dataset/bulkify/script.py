import scanpy as sc
import argparse
import anndata as ad
import numpy as np
import pandas as pd
from hiara.src.config import get_config, SUB_CT_LABEL, MAJOR_CT_LABEL
from task_grn_inference.src.process_data.helper_data import bulkify_func

def normalize(adata):
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)
    return adata
def qc_bulk(adata):
    # filter out bulk samples with less than 10 cells
    low_cells = adata.obs['cell_count'] < 10
    print(f'Dropping {low_cells.sum()} bulk samples with less than 10 cells', flush=True)
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
    dataset = adata.obs['dataset'].unique()[0]
    cfg = get_config(dataset)
    bulk_group = cfg.bulk_group
    covariate_major = bulk_group + [MAJOR_CT_LABEL]

    # - main bulk data (per major celltype)
    print('Bulkifying main cell types')
    adata_bulk_major_celltypes = bulkify_func(adata, covariates=covariate_major)
    adata_bulk_major_celltypes = normalize(adata_bulk_major_celltypes)
    if not args.run_test:
        adata_bulk_major_celltypes = qc_bulk(adata_bulk_major_celltypes)
    print(adata_bulk_major_celltypes.shape)
    adata_bulk_major_celltypes.write(args.bulk_all)

    # - bulk minor
    print('Bulkifying minor cell types')
    covariates_minor = bulk_group + [SUB_CT_LABEL]
    adata_bulk_minor_celltypes = bulkify_func(adata, covariates=covariates_minor)
    adata_bulk_minor_celltypes = normalize(adata_bulk_minor_celltypes)
    if not args.run_test:
        adata_bulk_minor_celltypes = qc_bulk(adata_bulk_minor_celltypes)
    adata_bulk_minor_celltypes.write(args.bulk_minor_celltype)


    print('DONE')


    