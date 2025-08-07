from ciim.src.common import base_dir
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import os

test_data = False
to_save = f'{base_dir}/datasets/perturbation/pbmc_cytokine_bulk.h5ad'

print('Loading data', flush=True)
adata = ad.read_h5ad('/vol/projects/jnourisa/datasets/perturbation/Parse_10M_PBMC_cytokines.h5ad', backed='r')


group_keys = ['cell_type', 'cytokine', 'donor', 'bc1_well']
for key in group_keys:
    adata.obs[key] = adata.obs[key].astype('str')

adata.obs['group'] = adata.obs[group_keys].astype(str).agg('_'.join, axis=1)


if test_data:
    print('Using test data', flush=True)
    # Select one cell per group
    cell_indices = adata.obs.groupby('group').apply(lambda x: x.index[0]).values

    # Create a new AnnData object in memory (small)
    adata_subset = adata[cell_indices, :].to_memory()

    # Optionally remove the temporary 'group' column
    
    adata = adata_subset
    cell_count_t = 1
    
else:
    adata = adata.to_memory()
    cell_count_t = 10
    
min_genes = 10
min_cell = adata.obs['bc1_well'].nunique()*10

adata = adata[(adata.obs['gene_count']>min_genes) & (adata.obs['gene_count']<5000), adata.var['n_cells']>min_cell]
# - pseudo bulk
print('Filtering data', flush=True)
from ciim.src.process_dataset.bulkify.helper import bulkify_main
print('Creating pseudo bulk data', flush=True)

adata.obs['group'] = adata.obs['group'].astype('str')

adata_bulk = bulkify_main(adata, covariates=['group'], cell_count_t=cell_count_t)
aaa - normalize
print('Saving pseudo bulk data', flush=True)

del adata_bulk.uns['log1p']
del adata_bulk.var
adata_bulk.obs['is_control'] = adata_bulk.obs['treatment'] == 'PBS'
adata_bulk.obs = adata_bulk.obs[['group', 'cell_type', 'cytokine', 'donor', 'is_control']]

adata_bulk.obs['perturbation_type'] = 'cytokine'
adata_bulk.layers['X_norm'] = adata_bulk.X.copy()
print('adata_bulk', adata_bulk, flush=True)

adata_bulk.obs['age'] = 20
adata_bulk.obs = adata_bulk.obs.rename({'donor': 'donor_id'}, axis=1)
adata_bulk.obs['cell_type_minor'] = adata_bulk.obs['cell_type']
cell_type_map = {
    'B Intermediate/Memory': 'B',
    'B Naive': 'B',
    'CD14 Mono': 'MONO',
    'CD16 Mono': 'MONO',
    'CD4 Memory': 'CD4T',
    'CD4 Naive': 'CD4T',
    'Treg': 'CD4T',
    'CD8 Memory': 'CD8T',
    'CD8 Naive': 'CD8T',
    'NK': 'NK',
    'NK CD56bright': 'NK',
    'NKT': 'NK',
    'MAIT': 'CD8T',  # can also be CD4/CD8 double negative, but often grouped under CD8T
    'ILC': 'NK',     # grouped under innate lymphoid/NK for many atlases
    'Plasmablast': 'B',
    'HSPC': None,    # does not map cleanly to any of the 5 requested groups
    'cDC': None,     # dendritic cell, not part of B, MONO, CD8T, CD4T, NK
    'pDC': None      # plasmacytoid DC, same
}
adata_bulk.obs['cell_type'] = adata_bulk.obs['cell_type_minor'].map(cell_type_map)
adata_bulk = adata_bulk[~adata_bulk.obs['cell_type'].isna(), :]
adata_bulk.obs['well'] = adata_bulk.obs['group'].str.split('_').str[-1]

adata_bulk.write_h5ad(to_save, compression='gzip')