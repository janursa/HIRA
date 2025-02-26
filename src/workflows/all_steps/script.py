
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

## VIASH START
parser = argparse.ArgumentParser()
parser.add_argument(
    '--run_preprocess',
    action='store_true',
    help="Whether to run preprocess (merging the datasets)"
)

parser.add_argument(
    '--run_process_dataset',
    action='store_true',
    help="Whether to run process_dataset (filtering the dataset)"
)

parser.add_argument(
    '--run_grn',
    action='store_true',
    help="Whether to run grn inference"
)


parser.add_argument(
    '--raw_dataset_file',
    type=str,
    required=True,
    help="The raw dataset file after merging"
)
parser.add_argument('--processed_dataset_file', 
    type=str,
    required=True,
    help="Processed dataset file after filtering"
    )
parser.add_argument('--bulk_dataset_file', 
    type=str,
    required=True,
    help="Processed dataset file after filtering, bulked"
    )

parser.add_argument('--max_workers', 
    type=int,
    required=False,
    default=10,
    help="Processed dataset file after filtering, bulked"
    )
 
parser.add_argument(
    '--downsample',
    action='store_true',
    help="Whether to equalize cell counts and donor sizes. Default is False."
)
parser.add_argument(
    '--only_male',
    action='store_true',
    help="Whether to subset the data to only males. Default is False."
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

parser.add_argument('--datasets', nargs='+', help='List of datasets to include', required=True)

args = parser.parse_args()


par = {
    # - run flags
        'run_preprocess': args.run_preprocess,
        'run_process_dataset': args.run_process_dataset,
        'run_grn': args.run_grn,
    # - preprocess datasets
        'datasets': args.datasets,
        'raw_dataset_file': args.raw_dataset_file,
        'scale': False,
    # - process dataset
        'processed_dataset_file': args.processed_dataset_file,
        'bulk_dataset_file': args.bulk_dataset_file,
        'n_cell_t': 200, # inclusion minimum number of cells per donor 
        'only_male': args.only_male,
        'downsample': args.downsample,

    # - grn inference parameters
        'weight_t': 0.05,
        'batches': ['all_batches'],
        'cell_types': ['B', 'CD4T', 'CD8T', 'MONO', 'NK', 'T'],
        'age_groups': ['all_agegroups', '65_75', '55_64', '75+', '34-', '35_44', '45_54'], # ['all_agegroups']
        'min_genes_per_cell': 10, 
        'max_genes_per_cell': 5000, 
        'min_cells_per_gene': 2500,
        'max_workers': args.max_workers,
        'force': args.force,
        'save_grns_dir': args.save_grns_dir,
        'temp_dir': 'output/grns/temp/'
} 


dependencies = {
    'grn_method': '/home/jnourisa/projs/ongoing/ciim/src/inference_methods/simple_corr/script.py',
    'process_dataset': '/home/jnourisa/projs/ongoing/ciim/src/process_dataset/script.py',
    'preprocess': '/home/jnourisa/projs/ongoing/ciim/src/preprocess/script.py'
}

os.makedirs(par['temp_dir'], exist_ok=True)

## VIASH END


def wrapper_grn(task, par):
    '''
        Take the task and run the GRN inference, save the results to the file. 
    '''
    (batch_group, cell_type, age_group, save_file_name) = task


    # Read dataset
    adata = ad.read_h5ad(par['processed_dataset_file'], backed='r')
    obs = adata.obs.copy()

    # Filter for the specific cell type, age group and batch group
    # Determine masks
    if cell_type == 'all_celltypes':
        cell_type_mask = np.full(obs.shape[0], True, dtype=bool)
    elif cell_type == 'T':
        cell_type_mask = (obs['cell_type'].isin(['CD4T', 'CD8T']))
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
    adata = ad.read_h5ad(par['processed_dataset_file'], backed='r')
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
    with ProcessPoolExecutor(max_workers=par['max_workers']) as executor:
        for result in executor.map(partial(wrapper_grn, par=par), tasks):
            pass

def main(par):
    if par['run_preprocess']:
        print('running preprocess...')
        args = f"--raw_dataset_file {par['raw_dataset_file']} "
        for i, dataset in enumerate(par['datasets']):
            if i == 0:
                args += f" --datasets {dataset}"
            else:
                args += f" {dataset}"
        command = f"python {dependencies['preprocess']} {args}"

        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError(f"Error: command proprocess failed with exit code {result.returncode}")
        print('preprocess completed')
    if par['run_process_dataset']:
        args = f"--raw_dataset_file {par['raw_dataset_file']} \
                --processed_dataset_file {par['processed_dataset_file']} \
                --bulk_dataset_file {par['bulk_dataset_file']}  \
                --n_cell_t {par['n_cell_t']}"

        if par['downsample']:
            args += " --downsample"
        if par['only_male']:
            args += " --only_male"

        print('running process dataset...')
        command = f"python {dependencies['process_dataset']} {args}"

        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError(f"Error: command process dataset failed with exit code {result.returncode}")
        print('process dataset completed')

    if par['run_grn']:
        # - run GRN inference
        print('running grn inference...')
        os.makedirs(par['save_grns_dir'], exist_ok=True)
        infer_grns_all(par)
        print('GRN inference completed')

if __name__ == '__main__': 
    
    main(par)