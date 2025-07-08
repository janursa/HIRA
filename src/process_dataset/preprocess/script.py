import os
default_n_threads = 3 # Change this based on the number of threads you want to use (equal to the number of cores in your machine (--cpus-per-task in the SLURM script))
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"
###
import numpy as np
import scanpy as sc
import seaborn as sns
import pandas as pd
import anndata as ad
import gc
import time
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

## VIASH START
parser = argparse.ArgumentParser()

parser.add_argument('--processed_files_dir', 
    type=str,
    required=True,
    help="Processed files dir"
    )
    
parser.add_argument('--raw_files_dir', 
    type=str,
    required=True,
    help="Location of raw files"
    )
parser.add_argument('--n_cell_t', 
    type=int,
    required=False,
    default=200,
    help="Number of threshold for cell count per donor to include"
    )
parser.add_argument('--dataset_name', help='dataset to process', required=True)

par = vars(parser.parse_args())


## VIASH END
from ciim.src.process_dataset.preprocess.helper import merge_datasets, annotate_celltypes, trim_adata, qc_check, process_obs
def all_preprocessing_steps(adata):
    print('Running QC...', flush=True)
    adata = qc_check(adata)
    adata.obs = adata.obs.astype('str')

    print('Running cell type annotation...', flush=True)
    adata = annotate_celltypes(adata)
    print('Cell type annotation done.', flush=True)
    adata = trim_adata(adata)

    # - make some additional changes to obs only
    obs = adata.obs.copy()
    obs = process_obs(obs=obs, par=par)
    assert not obs.isna().any().any()
    obs = obs.reindex(adata.obs.index)

    adata.obs = obs.copy()

    return adata
def main(par):
    dataset_name = par['dataset_name']
    if dataset_name == 'CXCL9':
        file_name = f"{par['raw_files_dir']}{dataset_name}_TI.h5ad"
    else:
        file_name = f"{par['raw_files_dir']}{dataset_name}_CMtx.h5ad"
    adata_main = ad.read_h5ad(file_name, backed='r')
        
    if 'race' in adata_main.obs.columns:
        races = adata_main.obs['race'].unique()
        for race in races:
            mask = adata_main.obs['race']==race
            adata = adata_main[mask].to_memory()
            adata = all_preprocessing_steps(adata)
            adata.obs['dataset'] = f"{dataset_name}_{race}"
            adata.write_h5ad(f"{par['processed_files_dir']}/{dataset_name}_{race}_sc.h5ad", compression='gzip')
    else:
        adata = adata_main.to_memory()
        adata = all_preprocessing_steps(adata)

        adata.obs['dataset'] = f"{dataset_name}"
        adata.write_h5ad(f"{par['processed_files_dir']}/{dataset_name}_sc.h5ad", compression='gzip')
    # adata = merge_datasets(par)


    print(adata, flush=True)



if __name__ == "__main__":
    print(par)
    main(par)
    
    
