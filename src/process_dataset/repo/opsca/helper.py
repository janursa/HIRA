import anndata as ad 
import pandas as pd
import numpy as np
import sctk
from scipy import sparse
import scanpy as sc
from task_grn_inference import sum_by
from hiara import basic_qc


import sys


def preprocess_sc(par):
    sc_counts = ad.read_h5ad(par['op_perturbation_raw'])
    sc_counts.obs['donor_id'] = sc_counts.obs.donor_id.map({'Donor 1': 'donor_0', 'Donor 2': 'donor_1', 'Donor 3': 'donor_2'})

    meta = pd.DataFrame({
        "donor_id": ['donor_0', 'donor_1', 'donor_2'],
        "age": [45, 52, 45],
        "sex": ["Female", "Male", "Male"]
    })
    sc_counts.obs = sc_counts.obs.rename(columns={'sm_name':'condition'})
    sc_counts.obs = sc_counts.obs.merge(meta, left_on='donor_id', right_on='donor_id', how='left')

    sc_counts.obs['sex'] = sc_counts.obs['sex'].astype(str)
    sc_counts.obs['age'] = sc_counts.obs['age'].astype(str)
    sc_counts.obs['is_control'] = sc_counts.obs['condition'].isin(['Dimethyl Sulfoxide'])
    sc_counts.obs['is_positive_control'] = sc_counts.obs['condition'].isin(['Dabrafenib', 'Belinostat'])

    # clean up
    sc_counts.obs = sc_counts.obs[['well', 'row', 'col', 'plate_name', 'cell_type', 'donor_id', 'condition', 'age', 'sex', 'is_control', 'is_positive_control']]
    sc_counts.X = sc_counts.layers['counts']
    del sc_counts.layers 
    del sc_counts.obsm 
    sc_counts.var_names_make_unique()
    # merge cell types
    if True:
        MAJOR_CTS = ['NK cells', 'T cells CD4+', 'T cells CD8+', 'T regulatory cells', 'B cells', 'Myeloid cells']
        T_cell_types = ['T regulatory cells', 'T cells CD4+']
        cell_type_map = {cell_type: 'CD4+' if cell_type in T_cell_types else cell_type for cell_type in MAJOR_CTS}
        sc_counts.obs['cell_type'] = sc_counts.obs['cell_type'].map(cell_type_map)
        sc_counts.obs['cell_type'] = sc_counts.obs['cell_type'].apply(lambda name: {'B cells': 'B', 'Myeloid cells':'MONO', 'NK cells':'NK', 'T cells CD8+':'CD8T', 'CD4+': 'CD4T' }.get(name, name))

    sc_counts = basic_qc(sc_counts)

    sc_counts.var = sc_counts.var[[]]

    del sc_counts.obsm
    del sc_counts.uns
    return sc_counts


# def pseudobulk_mean_func(bulk_adata):
#     bulk_adata.layers['counts'] = bulk_adata.X.copy()
#     rows_adj = []
#     for i, row in enumerate(bulk_adata.X):
#         count = bulk_adata.obs.cell_count[i]
#         rows_adj.append(row/count)

#     bulk_adata.layers['n_counts'] = np.asarray(rows_adj)

#     return bulk_adata
def filter_func(bulk_adata, cell_counts_t):
    '''Filters pseudobulked data by removing outliers compound, 
    samples with low cell counts, and genes with low coverage
    '''
    ### filter
    # samples with less than 10 cells
    bulk_adata_filtered = bulk_adata.copy()
    # toxic ones
    outliers_toxic = ['Alvocidib', 'UNII-BXU45ZH6LI', 'CGP 60474', 'BMS-387032']
    bulk_adata_filtered = bulk_adata_filtered[~bulk_adata_filtered.obs.condition.isin(outliers_toxic),:]
    # remove those with less than 10 cells left 

    mask_low_cell_count = bulk_adata_filtered.obs.cell_count < cell_counts_t
    bulk_adata_filtered = bulk_adata_filtered[~mask_low_cell_count]
    
    # remove those that have less than 2 cells types left per donor
    to_go_compounds = []
    for donor_id in bulk_adata_filtered.obs.donor_id.unique():
        adata_donor = bulk_adata_filtered[bulk_adata_filtered.obs.donor_id.eq(donor_id)]
        cell_type_n = adata_donor.obs.groupby('condition').size()
        to_go_compounds.append(cell_type_n[cell_type_n<=2].index.astype(str))
    to_go_compounds = np.unique(np.concatenate(to_go_compounds))
    outliers_two_celltype = ['CEP-18770 (Delanzomib)', 'IN1451', 'MLN 2238', 'Oprozomib (ONX 0912)']
    # assert np.all(to_go_compounds==outliers_two_celltype)
    bulk_adata_filtered = bulk_adata_filtered[~bulk_adata_filtered.obs.condition.isin(to_go_compounds),:]

    # remove big class misbalance in all donors 
    outliers_misbalance_all = ['Proscillaridin A;Proscillaridin-A'] 
    bulk_adata_filtered = bulk_adata_filtered[~bulk_adata_filtered.obs.condition.isin(outliers_misbalance_all),:]
    # remove big class misbalance in 1 donor
    outliers_misbalance_donor_2 = ['Vorinostat']
    bulk_adata_filtered = bulk_adata_filtered[~ (bulk_adata_filtered.obs.condition.isin(outliers_misbalance_donor_2) & (bulk_adata_filtered.obs.donor_id=='donor_1')),:]
    outliers_misbalance_donor_3 = ['AT13387', 'Ganetespib (STA-9090)']
    bulk_adata_filtered = bulk_adata_filtered[~ (bulk_adata_filtered.obs.condition.isin(outliers_misbalance_donor_3) & (bulk_adata_filtered.obs.donor_id=='donor_2')),:]
    print(f"number of initial samples: {len(bulk_adata)}, number of samples after filtering: {len(bulk_adata_filtered)}")
    # low gene coverage
    mask_to_go_genes = ((bulk_adata_filtered.X == 0).sum(axis=0)/bulk_adata_filtered.shape[0])>0.7
    print('number of removed genes:', mask_to_go_genes.sum())
    bulk_adata_filtered = bulk_adata_filtered[:,~mask_to_go_genes] 
    bulk_adata_filtered.obs.drop(columns=['plate_well_cell_type'], inplace=True)
    # for the sake of seurat
    for key in ['cell_type','plate_name']:
        bulk_adata_filtered.obs[key] = bulk_adata_filtered.obs[key].astype(str)
    return bulk_adata_filtered

def pseudobulk_sum_func(sc_counts):
    # pseudobulk
    #group cell types per well
    sc_counts.obs['plate_well_cell_type'] = sc_counts.obs['plate_name'].astype('str') \
        + '_' + sc_counts.obs['well'].astype('str') \
        + '_' + sc_counts.obs['cell_type'].astype('str')
    sc_counts.obs['plate_well_cell_type'] = sc_counts.obs['plate_well_cell_type'].astype('category')
    bulk_adata = sum_by(sc_counts, 'plate_well_cell_type')
    bulk_adata.obs['cell_count'] = sc_counts.obs.groupby('plate_well_cell_type').size().values
    bulk_adata.X = np.array(bulk_adata.X.todense())

    print('ratio of missingness' , (bulk_adata.X==0).sum()/bulk_adata.X.size)
    bulk_adata.var = bulk_adata.var.reset_index()
    bulk_adata.var.set_index('index', inplace=True)

    bulk_adata.X = np.nan_to_num(bulk_adata.X, nan=0)
    return bulk_adata