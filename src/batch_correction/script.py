

import argparse

import sys
import os 


current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.join(parent_directory, '../'))

import anndata as ad 
import pandas as pd
import numpy as np
import scanpy as sc
import argparse

def main(par):

    print(par)

    adata = ad.read_h5ad(par['adata'])

    sc.pp.filter_genes(adata, min_cells=200)
    sc.pp.filter_cells(adata, min_genes=10)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # - correct 
    sc.pp.combat(adata, key=par['batch_key'], inplace=True)  


    print(f"finished batch correction")
    adata.write_h5ad(par['adata_bc'])

if __name__ == '__main__':

    par = {
        'batch_key': 'dataset',
        'label_key': 'cell_type',
        'adata': '/vol/projects/jnourisa/adata_all.h5ad',
        'adata_bc': '/vol/projects/jnourisa/adata_all_bc.h5ad'
    }

    print(par)

    main(par)
    