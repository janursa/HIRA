"""Aggregate an annotated single-cell dataset into metacells (default 15 cells each).

Usage: python src/process_data/metacell/script.py --sc_dataset_file ... --metacell_out ...
Writes: <HIRA_BASE_DIR>/datasets/metacell/<dataset>.h5ad
"""
import scanpy as sc
import argparse
import anndata as ad
from hira.src.config import get_config, MAJOR_CT_LABEL
from hira.src.utils.util import metacellify_func

def normalize(adata):
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)
    return adata

def qc_metacell(adata, cell_t):
    low_cells = adata.obs['cell_count'] < cell_t
    print(f'Dropping {low_cells.sum()} metacells with less than {cell_t} cells', flush=True)
    return adata[~low_cells].copy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sc_dataset_file',
        type=str,
        required=True,
        help="Processed dataset file"
        )
    parser.add_argument('--metacell_out',
        type=str,
        required=True,
        help="Metacell data for major cell types"
        )
    parser.add_argument('--target-size',
        type=int,
        default=15,
        help="Target number of cells per metacell"
        )
    parser.add_argument('--run-test',
        action='store_true',
        help="Whether to run in test mode (subset of data)"
        )

    args = parser.parse_args()
    adata = ad.read_h5ad(args.sc_dataset_file)
    dataset = adata.obs['dataset'].unique()[0]
    cfg = get_config(dataset)
    covariates = cfg.bulk_group + [MAJOR_CT_LABEL]

    print(f'Building metacells (target size={args.target_size} cells, covariates={covariates})', flush=True)
    adata_metacell = metacellify_func(adata, target_size=args.target_size, covariates=covariates)
    adata_metacell = normalize(adata_metacell)
    adata_metacell = qc_metacell(adata_metacell, cell_t=1 if args.run_test else 5)
    print(f'Writing metacell data {adata_metacell.shape} to {args.metacell_out}', flush=True)
    adata_metacell.write(args.metacell_out)

    print('DONE')
