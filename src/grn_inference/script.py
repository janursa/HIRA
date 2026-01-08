
import sys
import os 
from tqdm import tqdm
import argparse
import anndata as ad
import pandas as pd
import scanpy as sc 
import numpy as np 
from scipy.stats import spearmanr
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import subprocess

from ciim.src.config import CELL_TYPES, minor_cell_types
from ciim.src.grn_inference.inference import main as main_inference
from task_grn_inference.src.utils.util import basic_qc
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_file', 
    type=str,
    required=True,
    help="Processed dataset file after filtering"
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
        'dataset_file': args.dataset_file,
        'weight_t': 0.05,
        # 'cell_types': minor_cell_types,
        # 'cell_type_col': 'Sub_CT',
        'min_genes_per_cell': 10, 
        'max_genes_per_cell': 5000 if args.data_type == 'sc' else 1e6, 
        'min_cells_per_gene': 1000 if args.data_type == 'sc' else 100,
        'data_type': args.data_type,
        'num_workers': args.num_workers,
        'force': args.force,
        'save_grns_dir': args.save_grns_dir,
        # 'temp_dir': 'output/grns/temp/',
} 

if args.cell_type_granularity == 'major':
    par['cell_types'] = CELL_TYPES
    par['cell_type_col'] = 'Major_CT'
elif args.cell_type_granularity == 'minor':
    par['cell_types'] = minor_cell_types
    par['cell_type_col'] = 'Sub_CT'
else:
    raise ValueError(f"Unknown cell type granularity: {args.cell_type_granularity}. Use 'major' or 'minor'.")

def wrapper_grn(task, par):
    '''
        Take the task and run the GRN inference, save the results to the file. 
    '''
    (cell_type, save_file_name) = task

    adata = ad.read_h5ad(par['dataset_file'], backed='r')
    obs = adata.obs.copy()

    # Filter for the specific cell type, age group and batch group
    # Determine masks
    if cell_type == 'all_celltypes':
        cell_type_mask = np.full(obs.shape[0], True, dtype=bool)
    elif cell_type == 'T':
        cell_type_mask = (obs[par['cell_type_col']].isin(['CD4T', 'CD8T']))
    else:
        cell_type_mask = (obs[par['cell_type_col']] == cell_type)

    mask_sample = cell_type_mask
    if mask_sample.sum() == 0:
        print(f"Error: No cells left after filtering for {cell_type}_{age_group}_{batch_group}", flush=True)
        return  
    adata = adata[mask_sample, :].to_memory()
    adata = basic_qc(adata, min_cells_per_gene=par['min_cells_per_gene'], min_genes_per_cell=par['min_genes_per_cell'], max_genes_per_cell=par['max_genes_per_cell'])

    # Infer GRN
    if par['data_type'] == 'sc':
        X_norm = sc.pp.normalize_total(adata, inplace=False)['X']
        X_norm = sc.pp.log1p(X_norm, copy=True)
    else:
        X_norm = adata.X

    net = main_inference(X_norm, adata.var_names, par['weight_t'])
    
    print("Adding metadata to the inferred network", flush=True)
    net['cell_type'] = cell_type
    net['sample_size'] = adata.shape[0]
    net['gene_size'] = adata.shape[1]
    net.to_csv(save_file_name, index=False)
    

def infer_grns_all(par):
    '''
        Infer GRNs for all cell types, age groups and batch groups.
    '''
    print(par, flush=True)
    # - read dataset
    adata = ad.read_h5ad(par['dataset_file'], backed='r')
    obs = adata.obs.copy()

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
    # - run GRN inference
    print('running grn inference...', flush=True)
    os.makedirs(par['save_grns_dir'], exist_ok=True)
    infer_grns_all(par)
    print('GRN inference completed', flush=True)