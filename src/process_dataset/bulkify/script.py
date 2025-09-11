import scanpy as sc
import argparse
import anndata as ad
import numpy as np
import pandas as pd

## VIASH START
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
parser.add_argument('--bulk_M',
    type=str,
    required=True,
    help="Bulk data for male"
    )
parser.add_argument('--bulk_F',
    type=str,
    required=True,
    help="Bulk data for female"
    )

args = parser.parse_args()

## VIASH END
from task_grn_inference.src.process_data.helper_data import bulkify_func

def normalize(adata):
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)
    return adata

if __name__ == '__main__':
    adata = ad.read_h5ad(args.sc_dataset_file)

    # - main bulk data (per major celltype)
    print('Bulkifying main cell types')
    covariates=['cell_type', 'donor_id', 'age'] # this should have one-to-one mapping with resulting bulked data (for example, if a donor has multiple treatment or disease, they will be summed together)
    if 'treatment' in adata.obs.columns:
        covariates.append('treatment')
    
    dataset = adata.obs['dataset'].unique()[0]
    if dataset == 'op':
        covariates = ['cell_type', 'plate_name', 'condition', 'well', 'donor_id']
    elif dataset == 'CXCL9':
        covariates = ['cell_type', 'pool_id', 'condition', 'donor_id']
        
    
    adata_bulk_major_celltypes = bulkify_func(adata, covariates=covariates)
    adata_bulk_major_celltypes = normalize(adata_bulk_major_celltypes)
    low_cells = adata_bulk_major_celltypes.obs['cell_count'] < 10
    print(f'Dropping {low_cells.sum()} bulk samples with less than 10 cells')
    adata_bulk_major_celltypes = adata_bulk_major_celltypes[~low_cells].copy()
    print(adata_bulk_major_celltypes.shape)
    adata_bulk_major_celltypes.write(args.bulk_all)

    # - bulk minor
    print('Bulkifying minor cell types')
    covariates_minor = covariates.copy()
    covariates_minor.append('Sub_CT')
    adata_bulk_minor_celltypes = bulkify_func(adata, covariates=covariates_minor)
    adata_bulk_minor_celltypes = normalize(adata_bulk_minor_celltypes)
    adata_bulk_minor_celltypes.write(args.bulk_minor_celltype)


    print('DONE')


    