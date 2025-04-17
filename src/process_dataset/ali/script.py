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
parser.add_argument(
    '--save_dir',
    type=str,
    required=True,
    help="The output dir"
)
parser.add_argument('--dataset_name', help='dataset to process', required=True)

args = parser.parse_args()


par= {
    'dataset_name': args.dataset_name,
    'inp_CMtx': "/vol/projects/CIIM/Healthy_Single_Cell_Data/count_matrix/",
    'save_dir': args.save_dir,
    'scale': False
}

## VIASH END
from ciim.src.process_dataset.ali.helper import merge_datasets, annotate_celltypes, trim_adata, qc_check
def all_steps(adata):
    adata = qc_check(adata)
    adata.obs = adata.obs.astype('str')

    adata = annotate_celltypes(adata)
    adata = trim_adata(adata)
    return adata
def main(par):
    dataset_name = par['dataset_name']
    adata_main = ad.read_h5ad(f"{par['inp_CMtx']}{dataset_name}_CMtx.h5ad", backed='r')
        
    if 'race' in adata_main.obs.columns:
        races = adata_main.obs['race'].unique()
        print(races)
        for race in races:
            mask = adata_main.obs['race']==race
            adata = adata_main[mask].to_memory()
            adata = all_steps(adata)
            adata.obs['dataset'] = f"{dataset_name}_{race}_raw"
            adata.write_h5ad(f"{par['save_dir']}/{dataset_name}_{race}_raw.h5ad", compression='gzip')
    else:
        adata = adata_main.to_memory()
        adata = all_steps(adata)

        adata.obs['dataset'] = f"{dataset_name}_raw"
        adata.write_h5ad(f"{par['save_dir']}/{dataset_name}_raw.h5ad", compression='gzip')
    # adata = merge_datasets(par)


    print(adata, flush=True)



if __name__ == "__main__":
    print(par)
    main(par)
    
    
