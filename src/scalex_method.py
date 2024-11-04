import sys
import anndata as ad
import scalex

## VIASH START
par = {
    'adata': 'resources_test/task_batch_integration/cxg_immune_cell_atlas/dataset.h5ad',
    'adata_bc': 'output.h5ad',
}

## VIASH END
import argparse
parser = argparse.ArgumentParser(description="Batch correction")
parser.add_argument('--adata', type=str, help='Path to the anndata file')
parser.add_argument('--adata_bc', type=str, help='Path to the anndata file')
parser.add_argument('--batch_key', type=str, help='Batch name')

args = parser.parse_args()

if args.adata:
    par['adata'] = args.adata
if args.adata_bc:
    par['adata_bc'] = args.adata_bc
if args.batch_key:
    par['batch_key'] = args.batch_key

print(par)

adata = ad.read_h5ad(par['adata'])

print('Run SCALEX', flush=True)
adata_bc = scalex.SCALEX(
    adata,
    batch_key= par['batch_key'],
    ignore_umap=True,
    impute=adata.obs[par['batch_key']].cat.categories[0],
    processed=True,
    max_iteration=40,
    min_features=10,
    min_cells=10,
    n_top_features=0,
    outdir=None,
    gpu=1,
)

print("Write output to file", flush=True)
adata_bc.write_h5ad(par['adata_bc'])