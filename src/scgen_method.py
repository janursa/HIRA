import anndata as ad 
import pandas as pd
import numpy as np
import scanpy as sc
import scgen
import argparse


## VIASH START
par = {
    'batch_key': 'plate_name',
    'label_key': 'cell_type'
}
## VIASH END
parser = argparse.ArgumentParser(description="Batch correction")
parser.add_argument('--adata', type=str, help='Path to the anndata file')
parser.add_argument('--adata_bc', type=str, help='Path to the anndata file')
parser.add_argument('--batch_key', type=str, help='Batch name')
parser.add_argument('--label_key', type=str, help='label name')

args = parser.parse_args()

if args.adata:
    par['adata'] = args.adata
if args.adata_bc:
    par['adata_bc'] = args.adata_bc
if args.label_key:
    par['label_key'] = args.label_key
if args.batch_key:
    par['batch_key'] = args.batch_key

print(par)

adata = ad.read_h5ad(par['adata'])
print(adata)

if True: # normalize
    print('Normalize')
    sc.pp.normalize_total(adata) 
    sc.pp.log1p(adata) 

scgen.SCGEN.setup_anndata(adata, batch_key=par['batch_key'], labels_key=par['label_key'])
model = scgen.SCGEN(adata)
model.train(
    max_epochs=100,
    batch_size=64,
    early_stopping=True,
    early_stopping_patience=25
)

corrected_adata = model.batch_removal()

adata.obsm["X_scgen"] = model.get_latent_representation()
adata.layers[f'scgen_corrected'] = corrected_adata.X
print(f"finished batch correction")
adata.write_h5ad(par['adata_bc'])