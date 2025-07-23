from ciim.src.common import base_dir
import anndata as ad
import numpy as np
import pandas as pd
import os

test_run = False

save_dir = f'{base_dir}/datasets/perturbation'
os.makedirs(save_dir, exist_ok=True)
ref_cell_types = ['HCT116'] #'HEK293T', 'HCT116' #Human Embryonic Kidney 293T cells, Human Colorectal Carcinoma Cell Line 116

for ref_cell_type in ref_cell_types:
    print('Reading data for', ref_cell_type, flush=True)
    adata = ad.read_h5ad(f'/vol/projects/CIIM/PerturbationDataset/Perturb_seq_dataset_Xiara/{ref_cell_type}_filtered_dual_guide_cells.h5ad', backed='r')
    adata.obs['is_control'] =  adata.obs['gene_target'] == 'Non-Targeting'
    
    
    to_save = f'{save_dir}/{ref_cell_type}.h5ad'

    if test_run: # test
        print('Running in test mode', flush=True)
        test_targets = adata.obs['gene_target'].unique()[:10]
        # Initialize a boolean mask of all False
        mask = pd.Series(False, index=adata.obs_names)

        for gene in test_targets:
            gene_cells = adata.obs.index[adata.obs['gene_target'] == gene]
            n_sample = min(10, len(gene_cells))
            sampled_cells = np.random.choice(gene_cells, size=n_sample, replace=False)
            mask.loc[sampled_cells] = True

        adata = adata[mask].to_memory()
        cell_count_t = 1
    else:
        # - QC
        print('Sending to memory', flush=True)
        adata = adata.to_memory()
        print('Running QC', flush=True)
        adata = adata[(adata.obs['n_genes_by_counts']>10) & (adata.obs['n_genes_by_counts']<5000) & (adata.obs['pct_counts_mt']<10)] 
        n_batches = adata.obs['sample'].nunique()
        n_cells_by_counts = 10*n_batches
        adata = adata[:, adata.var['n_cells_by_counts']>n_cells_by_counts]

        cell_count_t = 20
    # - pseudo bulk
    from ciim.src.process_dataset.bulkify.helper import bulkify_main
    print('Creating pseudo bulk data', flush=True)
    obs = adata.obs.copy()
    adata.obs['group'] = np.where(obs['is_control'], obs['sample'], obs['gene_target'])

    adata.obs['group'] = adata.obs['group'].astype('str')
    adata_bulk = bulkify_main(adata, covariates=['group'], cell_count_t=cell_count_t)
    print('Saving pseudo bulk data', flush=True)
    adata_bulk.write_h5ad(to_save, compression='gzip')