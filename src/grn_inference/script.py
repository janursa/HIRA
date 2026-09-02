"""Infer a per-cell-type GRN for one dataset.

Submitted per dataset by scripts/grn_inference/wrapper_grn_inference.sh; see --help
for arguments. Writes one network CSV per cell type into GRNS_DIR.
"""
import sys
import os 
from tqdm import tqdm
import argparse
import anndata as ad
import pandas as pd
from hira.src.utils.util import retrieve_adata, basic_qc
import scanpy as sc 
import numpy as np 
from scipy.stats import spearmanr
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import subprocess

from hira.src.config import MAJOR_CTS, PRIOR_DIR
from hira.src.grn_inference.inference import main as main_inference
from hira import get_config

def wrapper_grn(task, par):
    '''
        Take the task and run the GRN inference, save the results to the file. 
    '''
    (cell_type, save_file_name) = task

    # adata = ad.read_h5ad(par['dataset_file'], backed='r')
    adata = retrieve_adata(dataset=par['dataset'], data_type=par['data_type'], condition='healthy', cell_type=cell_type)
    config = get_config(dataset=par['dataset'])
    pseudobulk_group = config.bulk_group
    # for each pseudobulk_group, select only 5000 single cells
    print('Shape before sampling: ', adata.shape, flush=True)
    sampled_indices = []
    for group_name, group_df in adata.obs.groupby(pseudobulk_group, sort=False):
        sample_size = min(5000, len(group_df))
        sampled_indices.extend(group_df.sample(n=sample_size, random_state=0).index.tolist())
    adata = adata[sampled_indices].copy()
    print('Shape after sampling: ', adata.shape, flush=True)

    if False:
        obs = adata.obs.copy()
        gene_names = np.loadtxt(f'{PRIOR_DIR}/gene_names.txt', dtype=str)
        mask_genes = np.isin(adata.var_names, gene_names)
        adata = adata[:, mask_genes].to_memory()
    adata = basic_qc(adata, min_cells_per_gene=par['min_cells_per_gene'], min_genes_per_cell=par['min_genes_per_cell'], max_genes_per_cell=par['max_genes_per_cell'])

    # Infer GRN
    if par['data_type'] == 'sc':
        X_norm = sc.pp.normalize_total(adata, inplace=False)['X']
        X_norm = sc.pp.log1p(X_norm, copy=True)
    else:
        X_norm = adata.X

    net = main_inference(X_norm, adata.var_names)
    
    print("Adding metadata to the inferred network", flush=True)
    net['cell_type'] = cell_type
    net['sample_size'] = adata.shape[0]
    net['gene_size'] = adata.shape[1]

    # Store a generous superset; pruning/truncation is a load-time choice (see retrieve_net).
    net = net.sort_values(by='weight', ascending=False, key=abs).head(par['top_n_edges'])
    print('Shape of the inferred network: ', net.shape, flush=True)
    net.to_csv(save_file_name, index=False)
    

def infer_grns_all(par):
    '''
        Infer GRNs for all cell types, age groups and batch groups.
    '''
    print(par, flush=True)

    # - prepare tasks
    tasks = []
    for cell_type in par['cell_types']:
        save_file_name = os.path.abspath(f"{par['save_grns_dir']}/net_{cell_type}.csv")
        if par['force']:
            tasks.append((cell_type, save_file_name))
        else:
            if not os.path.exists(save_file_name):  
                tasks.append((cell_type, save_file_name))

    print('number of tasks: ', len(tasks), flush=True)
    # - run tasks 
    with ProcessPoolExecutor(max_workers=par['num_workers']) as executor:
        for result in executor.map(partial(wrapper_grn, par=par), tasks):
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', 
        type=str,
        required=True,
        )

    parser.add_argument(
        '--save_grns_dir',
        type=str,
        required=True,
        help="File of the dataset"
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help="Force to rewrite existing files (only grns)."
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
    )

    parser.add_argument(
        '--data_type',
        type=str,
        default='sc',
        help="Type of data: bulk or single-cell"
    )

    parser.add_argument(
        '--cell_type_granularity',
        type=str,
        default='major',
        help="Whether to infer GRNs for major or minor cell types."
    )

    args = parser.parse_args()
    par = {
        # - grn inference parameters
            'dataset': args.dataset,
            'cell_types': MAJOR_CTS, #TODO: fix me
            'min_genes_per_cell': 10, 
            'max_genes_per_cell': 5000 if args.data_type == 'sc' else 1e6, 
            'min_cells_per_gene': 500 if args.data_type == 'sc' else 100,
            'data_type': args.data_type,
            'num_workers': args.num_workers,
            'force': args.force,
            'top_n_edges': 500_000,
            'save_grns_dir': args.save_grns_dir,
            # 'temp_dir': 'results_folder/grns/temp/',
    } 
    print('running grn inference...', flush=True)
    os.makedirs(par['save_grns_dir'], exist_ok=True)
    infer_grns_all(par)
    print('GRN inference completed', flush=True)