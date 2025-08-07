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
from ciim.src.utils.util import bulkify_main

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
    
    adata_bulk_major_celltypes = bulkify_main(adata, covariates=covariates)
    adata_bulk_major_celltypes = normalize(adata_bulk_major_celltypes)
    adata_bulk_major_celltypes.write(args.bulk_all)

    # - bulk minor
    print('Bulkifying minor cell types')
    covariates_minor = covariates.copy()
    covariates_minor.append('Sub_CT')
    adata_bulk_minor_celltypes = bulkify_main(adata, covariates=covariates_minor)
    adata_bulk_minor_celltypes = normalize(adata_bulk_minor_celltypes)
    adata_bulk_minor_celltypes.write(args.bulk_minor_celltype)


    print('DONE')


    