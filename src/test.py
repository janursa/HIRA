
import seaborn as sns
import anndata as ad 
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import sys 
sys.path.append('./')
from src.helper import *

adata_all = ad.read_h5ad('input/adata.h5ad')

adata_all.layers['counts'] = adata_all.X.copy()

for i, age_donor in enumerate(adata_all.obs.age_donor.unique()):
    adata_sub = adata_all[adata_all.obs.age_donor==age_donor]
    adata_sub = basic_qc(adata_sub, min_genes_per_cell=10, min_cells_per_gene=100)

    # - normalize apr 
    adata_sub.X = adata_sub.layers['counts'].copy()
    sc.experimental.pp.normalize_pearson_residuals(adata_sub)
    adata_sub.layers['apr'] = adata_sub.X.copy()

    # - normalize sla
    adata_sub.X = adata_sub.layers['counts'].copy()
    sc.pp.normalize_total(adata_sub)
    sc.pp.log1p(adata_sub)
    adata_sub.layers['sla'] = adata_sub.X.copy()
    
    if i == 0:
        adata_sub_all = adata_sub
    else:
        adata_sub_all = ad.concat([adata_sub_all, adata_sub], axis=0)
adata_sub_all.write('input/adata_n.h5ad')