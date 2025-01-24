
import sys
import subprocess
import os
import anndata as ad
import scanpy as sc 
import os
import anndata as ad
import numpy as np 
import pandas as pd 
import seaborn as sns
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from scipy.stats import spearmanr
import sys
import matplotlib.pyplot as plt
import scanpy as sc 
# import decoupler as dc 
import json
import warnings
from tqdm import tqdm
from scipy import stats
import numpy as np
from scipy.stats import spearmanr, t
from concurrent.futures import ProcessPoolExecutor
from functools import partial

sys.path.insert(0, '../')
from task_grn_inference.src.utils.util import basic_qc
sys.path.insert(0, './')
from src.helper import efficient_melting



import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def process_grn(task, par):
    (obs, save_file_name) = task
    batch_group = obs['batch_group'].unique()[0]
    cell_type = obs['cell_type'].unique()[0]
    age_group = obs['age_group'].unique()[0]

    # Read dataset
    adata = ad.read_h5ad(par['dataset_file'], backed='r')
    mask_sample = adata.obs.index.isin(obs.index)
    adata_sample = adata[mask_sample, :].to_memory()
    # Subset and QC
    adata_sample = basic_qc(adata_sample, min_cells_per_gene=par['min_cells_per_gene'], min_genes_per_cell=par['min_genes_per_cell'])
    if adata_sample.shape[0] == 0 or adata_sample.shape[1] == 0:
        return None
    
    # Normalize #TODO: skip this if it's given
    X_norm = sc.pp.normalize_total(adata_sample, inplace=False)['X']
    adata_sample.layers['X_norm'] = sc.pp.log1p(X_norm, copy=True)
    
    # Infer GRN
    expression_sample = adata_sample.layers['X_norm']
    gene_names = adata_sample.var_names
    net = infer_grn(expression_sample, gene_names)
    net['weight'] = pd.to_numeric(net['weight'], errors='coerce')
    net = net[net['weight'].abs() > par['weight_t']]
    
    # Add metadata
    net['batch_group'] = batch_group
    net['cell_type'] = cell_type
    net['age_group'] = age_group
    net['sample_size'] = adata_sample.shape[0]
    net['gene_size'] = adata_sample.shape[1]
    
    # Save results
    net.to_csv(save_file_name)

def infer_grns_all(par, obs):
    batches = par['batches']
    cell_types = obs['cell_type'].astype(str).unique()
    age_groups = obs['age_group'].astype(str).unique()
    print('Cell types to infer GRNs for : ', cell_types)
    # - prepare tasks
    tasks = []
    for cell_type in cell_types:
        for age_group in age_groups:
            for batch_group in batches:
                # Determine masks
                if cell_type == 'all_celltypes':
                    cell_type_mask = np.full(obs.shape[0], True, dtype=bool)
                else:
                    cell_type_mask = (obs['cell_type'] == cell_type)
                if batch_group == 'all_batches':
                    batch_group_mask = np.full(obs.shape[0], True, dtype=bool)
                else:
                    batch_group_mask = (obs['batch_group'] == batch_group)
                if age_group == 'all_agegroups':
                    age_group_mask = np.full(obs.shape[0], True, dtype=bool)
                else:
                    age_group_mask = (obs['age_group'] == age_group)
                # Combine masks
                mask_sample = batch_group_mask & age_group_mask & cell_type_mask
                obs_sample = obs[mask_sample].copy()
      
                save_file_name = os.path.abspath(f"{par['save_dir']}/net_{cell_type}_{age_group}_{batch_group}.csv")
                if par['force']:
                    tasks.append((obs_sample, save_file_name))
                else:
                    if not os.path.exists(save_file_name):  # Skip if file exists
                        tasks.append((obs_sample, save_file_name))
    
    print('number of tasks: ', len(tasks))
    # Run tasks in parallel
    grns_store = []
    with ProcessPoolExecutor(max_workers=par['max_workers']) as executor:
        for result in executor.map(partial(process_grn, par=par), tasks):
            if result is not None:
                grns_store.append(result)

def enrich_tf_local(net, adata_bulk, tf_all):
    net = net.pivot(index='source', columns='target', values='weight').fillna(0)
    net = net[[g for g in adata_bulk.var_names if g in net.columns]]
    tfs_present = np.intersect1d(net.index, tf_all)
    net = net[net.index.isin(tfs_present)]
    print('ratio of porosity: ', (net==0).sum().sum()/net.size)
    # - subset the adata
    adata_bulk = adata_bulk[:, adata_bulk.var_names.isin(net.columns)]
    # - enrich tfs 
    mat = adata_bulk.X.todense().T
    print(net.shape, mat.shape)
    tf_acts = np.dot(net, mat)
    # - format
    tf_acts = pd.DataFrame(tf_acts, index=net.index, columns=adata_bulk.obs.index)
    tf_acts = tf_acts.reset_index().melt(id_vars='source', var_name='sample', value_name='activity')
    tf_acts = tf_acts.set_index('sample').merge(adata_bulk.obs[['cell_type', 'donor_id', 'cell_count', 'age']], left_index=True, right_index=True).reset_index(drop=True)
    return tf_acts
def sparse_std(X):
    from sklearn.preprocessing import StandardScaler
    scalar = StandardScaler(with_mean=False)
    scalar.fit(X)
    X_var = scalar.var_
    return X_var
def sparse_corrcoef(A, B=None):
    if B is not None:
        A = sparse.vstack((A, B), format='csr')
    A = A.astype(np.float64)
    n = A.shape[1]
    # Compute the covariance matrix
    rowsum = A.sum(1)
    centering = rowsum.dot(rowsum.T.conjugate()) / n
    C = (A.dot(A.T.conjugate()) - centering) / (n - 1)
    # The correlation coefficients are given by
    # C_{i,j} / sqrt(C_{i} * C_{j})
    d = np.diag(C)
    coeffs = C / np.sqrt(np.outer(d, d))
    return coeffs


def infer_grn(X, gene_names):
    from scipy.stats import spearmanr
    std_devs = sparse_std(X)
    mask_zero_std = std_devs == 0
    gene_names = gene_names[~mask_zero_std]
    X_filtered = X[:, ~mask_zero_std]
    if False:
        corr, _ = spearmanr(X_filtered, nan_policy='raise')
    else:
        print('start corr calculation')
        corr = sparse_corrcoef(X_filtered.T)
        print(corr.shape)
    try:
        net = efficient_melting(corr.A, gene_names)
    except:
        net = efficient_melting(corr, gene_names)
    return net 

def enrich_tfs(adata_sample, net, tf_all, par):
    import decoupler
    from scipy.stats import zscore

    # - pseudobulk cell type-donor
    sys.path.insert(0, '../')
    from task_grn_inference.src.process_data.perturbation.opsca.script import sum_by

    adata_sample.obs['sum_by'] = '_' + adata_sample.obs['cell_type'].astype(str) + '_' + adata_sample.obs['donor_id'].astype(str)
    adata_sample.obs['sum_by'] = adata_sample.obs['sum_by'].astype('category')
    adata_bulk = sum_by(adata_sample, 'sum_by', unique_mapping=False)
    cell_count_df = adata_sample.obs.groupby('sum_by').size().reset_index(name='cell_count')
    adata_bulk.obs = adata_bulk.obs.merge(cell_count_df, on='sum_by')
    adata_bulk = adata_bulk[adata_bulk.obs['cell_count']>par['n_cells_t']]

    # - normalize
    if par['normalize']:
        sc.pp.normalize_total(adata_bulk)
        sc.pp.log1p(adata_bulk)

    # -enrich TFs
    net = net[net['source'].isin(tf_all)]

    if False:
        mat = pd.DataFrame(
            data=adata_bulk.X.todense(),  
            columns=adata_bulk.var_names,  
            index=adata_bulk.obs.index  
        )

        tf_acts, tf_pvals = decoupler.run_ulm(mat, net, source='source', target='target', weight='weight', use_raw=False)
        # - formatize
        tf_acts = tf_acts.reset_index().melt(id_vars='index', var_name='source', value_name='activity')
        obs = adata_bulk.obs[['cell_type', 'donor_id', 'cell_count', 'age']]
        obs = obs.reset_index()
        
        tf_acts['index'] = tf_acts['index'].astype(str)
        obs['index'] = obs['index'].astype(str)
        tf_acts = tf_acts.merge(obs, on='index', how='left').drop('index', axis=1)
        assert tf_acts.shape[0]==tf_acts.shape[0]
    else:
        tf_acts = enrich_tf_local(net, adata_bulk, tf_all)

    # - zscore normalizaton
    if par['zscore_transform']:
        tf_acts = tf_acts.apply(zscore, axis=0)

    
    return tf_acts
