
import sys
import os 
from tqdm import tqdm
import argparse
import anndata as ad
import numpy as np
import pandas as pd

## VIASH START
parser = argparse.ArgumentParser()

parser.add_argument(
    '--raw_dataset_file',
    type=str,
    required=True,
    help="The input dataset"
)
parser.add_argument('--processed_dataset_file', 
    type=str,
    required=True,
    help="Processed dataset file"
    )
    
parser.add_argument('--n_cell_t', 
    type=int,
    required=False,
    default=200,
    help="Number of threshold for cell count per donor to include"
    )

parser.add_argument(
    '--downsample',
    action='store_true',
    help="Whether to equalize cell counts and donor sizes. Default is False."
)
parser.add_argument(
    '--gender',
    type=str,
    default='both'
)

args = parser.parse_args()

par_input = vars(args)

par = {
    'equalize_donor_size': True,
}

for key in par_input.keys():
    if par_input[key] is not None:
        par[key] = par_input[key]

## VIASH END


from ciim.src.utils.util import basic_qc
from helper import process_obs

def main(par):
    print('Processing dataset...')
    adata = ad.read_h5ad(par['raw_dataset_file'], backed='r')
    obs = adata.obs.copy()
    obs = process_obs(obs=obs, par=par)
    assert not obs.isna().any().any()
    obs = obs.reindex(adata.obs.index)

    adata.obs = obs.copy()
    print('Processed adata size: ', adata.shape)
    
    print(f"Saving adata in progress. Loading adata...")
    adata = adata.to_memory()
    adata = basic_qc(adata, min_cells_per_gene=100, min_genes_per_cell=10)
    print(f"Saving adata to {par['processed_dataset_file']}")
    adata.write(par['processed_dataset_file'])



if __name__ == '__main__':
    print(par, flush=True)
    
    main(par)


        