
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

from ciim.src.common import cell_types, minor_cell_types

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
        'batches': ['all_batches'],
        # 'cell_types': minor_cell_types,
        # 'cell_type_col': 'Sub_CT',
        'age_groups': ['all_agegroups'], # ['all_agegroups',  '50-', '50+' '65_75', '55_64', '75+', '34-', '35_44', '45_54']
        'min_genes_per_cell': 10, 
        'max_genes_per_cell': 5000, 
        'min_cells_per_gene': 1000,
        'data_type': args.data_type,
        'num_workers': args.num_workers,
        'force': args.force,
        'save_grns_dir': args.save_grns_dir,
        'temp_dir': 'output/grns/temp/'
} 

if args.cell_type_granularity == 'major':
    par['cell_types'] = cell_types
    par['cell_type_col'] = 'Major_CT'
elif args.cell_type_granularity == 'minor':
    par['cell_types'] = minor_cell_types
    par['cell_type_col'] = 'Sub_CT'
else:
    raise ValueError(f"Unknown cell type granularity: {args.cell_type_granularity}. Use 'major' or 'minor'.")

dependencies = {
    'grn_method': '/home/jnourisa/projs/ongoing/ciim/src/inference_methods/simple_corr/script.py',
}

os.makedirs(par['temp_dir'], exist_ok=True)

## VIASH END


def wrapper_grn(task, par):
    '''
        Take the task and run the GRN inference, save the results to the file. 
    '''
    (batch_group, cell_type, age_group, save_file_name) = task


    # Read dataset
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
    if batch_group == 'all_batches':
        batch_group_mask = np.full(obs.shape[0], True, dtype=bool)
    else:
        batch_group_mask = (obs['batch_group'] == batch_group)
    if age_group == 'all_agegroups':
        age_group_mask = np.full(obs.shape[0], True, dtype=bool)
    else:
        age_group_mask = (obs['age_group'] == age_group)
    mask_sample = cell_type_mask & batch_group_mask & age_group_mask
    if mask_sample.sum() == 0:
        print(f"Error: No cells left after filtering for {cell_type}_{age_group}_{batch_group}")
        return  
    adata_sample = adata[mask_sample, :].to_memory()
    
    # Infer GRN
    adata_file = f"{par['temp_dir']}/{cell_type}_{age_group}_{batch_group}.h5ad"
    adata_sample.write(adata_file)
    args = f"--rna {adata_file} --prediction {save_file_name} \
            --min_cells_per_gene {par['min_cells_per_gene']} \
            --min_genes_per_cell {par['min_genes_per_cell']} \
            --data_type {par['data_type']} \
            --weight_t {par['weight_t']} "
    command_grn = f"python {dependencies['grn_method']} {args}"
    try:
        subprocess.run(command_grn, shell=True, check=True)
    except IndexError:
        print(f"Error: No cells or genes left after filtering for {cell_type}_{age_group}_{batch_group}") 


    # Add metadata
    if os.path.exists(save_file_name):
        print("Adding metadata to the inferred network")
        net = pd.read_csv(save_file_name)
        print('----- batch_group: ', batch_group)
        net['batch_group'] = batch_group
        net['cell_type'] = cell_type
        net['age_group'] = age_group
        net['sample_size'] = adata_sample.shape[0]
        net['gene_size'] = adata_sample.shape[1]
        net.to_csv(save_file_name, index=False)

    
    

def infer_grns_all(par):
    '''
        Infer GRNs for all cell types, age groups and batch groups.
    '''
    print(par)
    # - read dataset
    adata = ad.read_h5ad(par['dataset_file'], backed='r')
    obs = adata.obs.copy()
    
    # - prepare tasks
    batches = par['batches']
    cell_types = par['cell_types']
    age_groups = par['age_groups']

    # - prepare tasks
    tasks = []
    for cell_type in cell_types:
        for age_group in age_groups:
            for batch_group in batches:
                save_file_name = os.path.abspath(f"{par['save_grns_dir']}/net_{cell_type}_{age_group}_{batch_group}.csv")
                if par['force']:
                    tasks.append((batch_group, cell_type, age_group, save_file_name))
                else:
                    if not os.path.exists(save_file_name):  
                        tasks.append((batch_group, cell_type, age_group, save_file_name))

    print('number of tasks: ', len(tasks))
    # - run tasks 
    with ProcessPoolExecutor(num_workers=par['num_workers']) as executor:
        for result in executor.map(partial(wrapper_grn, par=par), tasks):
            pass

if __name__ == '__main__':
    # - run GRN inference
    print('running grn inference...')
    os.makedirs(par['save_grns_dir'], exist_ok=True)
    infer_grns_all(par)
    print('GRN inference completed')
