# !pip install sctk anndata
# !aws s3 cp s3://openproblems-bio/public/neurips-2023-competition/sc_counts.h5ad  ./resources_raw/ --no-sign-request

import anndata as ad 
import pandas as pd
import numpy as np
import sys
from scipy.sparse import csr_matrix
import scanpy as sc


from hiara import TASK_GRN_BENCHMARK_DIR, DATA_DIR
 


## VIASH START
par = {
    # 'op_perturbation_raw': f'{TASK_GRN_BENCHMARK_DIR}/resources/datasets_raw/op_perturbation_sc_counts.h5ad',
    'op_perturbation_raw': f'/vol/projects/jnourisa/task_grn_benchmark/resources/datasets_raw/op_perturbation_sc_counts.h5ad',
    'op_perturbation_bulk': f'{DATA_DIR}/bulk/op.h5ad',
}
## VIASH END

meta = { 
    'helper_dir': './'
}   
sys.path.append(TASK_GRN_BENCHMARK_DIR)
sys.path.append(meta['helper_dir'])
from helper import preprocess_sc, filter_func, pseudobulk_sum_func
from task_grn_inference import normalize_func

def main_perturbation(par):
    cell_counts_t = 10
        
    sc_counts_f = preprocess_sc(par)
    bulk_adata = pseudobulk_sum_func(sc_counts_f)
    bulk_adata = filter_func(bulk_adata, cell_counts_t)

    bulk_adata.obs = bulk_adata.obs.rename(columns={'sm_name':'perturbation'})

    bulk_adata = normalize_func(bulk_adata)

    bulk_adata.X = csr_matrix(bulk_adata.X)

    bulk_adata.obs['is_control'] = bulk_adata.obs['perturbation'].isin(['Dimethyl Sulfoxide'])
    bulk_adata.obs['is_positive_control'] = bulk_adata.obs['perturbation'].isin(['Dabrafenib', 'Belinostat'])

    print('writing op_perturbation_bulk')
    print(bulk_adata)

    meta = pd.DataFrame({
        "donor_id": ['donor_0', 'donor_1', 'donor_2'],
        "age": [45, 52, 45],
        "sex": ["Female", "Male", "Male"]
    })

    # join metadata into obs
    bulk_adata.obs = bulk_adata.obs.merge(meta, left_on='donor_id', right_on='donor_id', how='left')
    print('Writing bulk adata with shape:', bulk_adata.shape, ' to ', par['op_perturbation_bulk'], flush=True)
    bulk_adata.write(par['op_perturbation_bulk'])

if __name__ == '__main__':
    print('Processing perturbation data ...')
    main_perturbation(par)

    