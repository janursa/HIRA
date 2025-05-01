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
from helper import bulkify_main

if __name__ == '__main__':
    adata = ad.read_h5ad(args.sc_dataset_file)
    # - main bulk data (per major celltype)
    print('Bulkifying main cell types')
    covariates=['cell_type', 'donor_id', 'age'] # this should have one-to-one mapping with resulting bulked data (for example, if a donor has multiple treatment or disease, they will be summed together)
    if 'treatment' in adata.obs.columns:
        covariates.append('treatment')
    adata_bulk_major_celltypes = bulkify_main(adata, covariates=covariates)
    adata_bulk_major_celltypes.write(args.bulk_all)
    print('Bulkifying minor cell types')
    adata_bulk_minor_celltypes = bulkify_main(adata, covariates=covariates)
    adata_bulk_minor_celltypes.write(args.bulk_minor_celltype)

    if 'M' in adata.obs['sex'].unique():
        adata_m = adata[adata.obs['sex']=='M']
        adata_f = adata[adata.obs['sex']=='F']
    elif 'Male' in adata.obs['sex'].unique():
        adata_m = adata[adata.obs['sex']=='Male']
        adata_f = adata[adata.obs['sex']=='Female']
    else:
        raise ValueError('Sex is not standard')

    assert adata_m.shape[0] > 0, 'zero adata for male'
    assert adata_f.shape[0] > 0, 'zero adata for female'

    print('Bulking male')
    adata_bulk_major_celltypes_m = bulkify_main(adata_m, covariates=covariates)
    adata_bulk_major_celltypes_m.write(args.bulk_M)
    print('Bulking female')
    adata_bulk_major_celltypes_f = bulkify_main(adata_f, covariates=covariates)
    adata_bulk_major_celltypes_f.write(args.bulk_F)
    print('DONE')


    