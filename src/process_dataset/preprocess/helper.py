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
### Load the .h5ad files
def merge_datasets(par) -> ad.AnnData:
    print('Merging datasets...')
    gc.collect()

    for i, dataset_name in tqdm(enumerate(par['datasets']), desc='Merging datasets', total=len(par['datasets'])):
        if i == 0:
            adata = ad.read_h5ad(f"{par['inp_CMtx']}{dataset_name}_CMtx.h5ad")
            adata.obs['dataset'] = dataset_name
        else:
            adata = adata.concatenate(ad.read_h5ad(f"{par['inp_CMtx']}{dataset_name}_CMtx.h5ad"), batch_key="dataset", join="inner")

    # Merge the AnnData objects
    gc.collect()

    return adata

### QC Check
def qc_check(adata):
    print('Shape before filtering:', adata.shape)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_cells(adata, max_genes=5000)
    # - in filtering, consider the number of donors
    n_donors = adata.obs['donor_id'].nunique()
    min_cells_per_donor = 10
    min_cells = int(n_donors * min_cells_per_donor)
    min_cells = max(min_cells, 10)

    # Apply filters
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_genes(adata, min_counts=1)
    print('Shape after filtering:', adata.shape)

    return adata
def qc_post_annotation(adata, par):
    if 'condition' in adata.obs.columns and 'cell_type' in adata.obs.columns:
        print(f'Filtering based on condition + cell_type combinations')
        min_cells_per_group = 10
        group_counts = adata.obs.groupby(['condition', 'cell_type']).size()
        keep_groups = group_counts[group_counts >= min_cells_per_group].index
        mask = adata.obs.set_index(['condition', 'cell_type']).index.isin(keep_groups)
        adata = adata[mask].copy()
        print(f"Kept {len(keep_groups)} condition+cell_type groups (>= {min_cells_per_group} cells each) out of {len(group_counts)} total")
    sample_size = adata.obs.groupby('donor_age', as_index=False).size()
    sample_size = sample_size[sample_size['size']>par['n_cell_t']]
    adata = adata[adata.obs['donor_age'].isin(sample_size.donor_age)]
    print('size after filtering for donor sinlge cell count: ', adata.shape)

    return adata
def format_data(adata, dataset_name):
    if dataset_name == 'op':
        adata.obs = adata.obs.rename(columns={'sm_name':'perturbation'})
        adata.obs['is_control'] = adata.obs['perturbation'].isin(['Dimethyl Sulfoxide'])
        adata.obs['is_positive_control'] = adata.obs['perturbation'].isin(['Dabrafenib', 'Belinostat'])
        
        meta = pd.DataFrame({
            "donor_id": ['Donor 1', 'Donor 2', 'Donor 3'],
            "age": [45, 52, 45],
            "sex": ["Female", "Male", "Male"]
        })

        # join metadata into obs
        adata.obs = adata.obs.merge(meta, left_on='donor_id', right_on='donor_id', how='left')
        print(adata.obs)

    adata.obs.rename(columns={'perturbation':'condition', 'disease':'condition', 'treatment':'condition'}, inplace=True)
    adata.obs.rename(columns={'orig.ident': 'dataset'}, inplace=True)
    adata.obs = adata.obs.astype('str')
    adata.obs['donor_age'] = adata.obs['age'].astype(str) + '_' + adata.obs['donor_id'].astype(str)
    return adata
def annotate_celltypes(adata):
    print('Annotating cell types...')
    adata.layers['counts'] = adata.X.copy()
    original_cols = adata.obs.columns
    ### Celltype annotation via Celltypist:
    import celltypist
    from celltypist import models
    ### Normalization
    # Before normalization, we need to store the raw counts in the layers['counts'] metadata for later use in the differential expression analysis
    gc.collect()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    models.download_models(force_update = True)

    model = models.Model.load(model = 'Immune_All_Low.pkl') #Immune_All_Low good for Major cell types and Immune_All_High for subtypes
    # Please note that the adata.X should be log-normalized data!
    adata_for_celltypist = adata.copy()
    # Annotate cell types using CellTypist
    print('Annotating cell types using CellTypist...')
    print(adata_for_celltypist.shape)
    predictions = celltypist.annotate(
        adata_for_celltypist,
        model=model,
        majority_voting=True,
        use_GPU=False
    )
    print('Cell types annotated successfully!')
    # Update the AnnData object with predictions
    adata_for_celltypist = predictions.to_adata()


    ###Major CT
    mapping = {
        'Tcm/Naive helper T cells': 'CD4T',
        'CD16+ NK cells': 'NK',
        'Classical monocytes': 'MONO',
        'Tem/Temra cytotoxic T cells': 'CD8T',
        'Tem/Effector helper T cells': 'CD4T',
        'Tcm/Naive cytotoxic T cells': 'CD8T',
        'B cells': 'B',
        'Naive B cells': 'B',
        'Tem/Trm cytotoxic T cells': 'CD8T',
        'Memory B cells': 'B',
        'Non-classical monocytes': 'MONO',
        'MAIT cells': 'CD8T',  # or 'MAIT' if you want to keep it separate
        'Regulatory T cells': 'CD4T',
        'Cycling T cells' : 'CD4T',
        'DC2': 'DC',
        'pDC': 'DC',
        'Intermediate macrophages': 'MONO',
        'NK cells': 'NK',
        'Plasma cells': 'B',
        'HSC/MPP': 'HSC',
        'Age-associated B cells': 'B',
        'DC1': 'DC',
        'Megakaryocytes/platelets': 'Megakaryocyte',
        'Plasmablasts': 'B',
        'ILC': 'ILC',
        'CD8a/a': 'CD8T',
        'Double-positive thymocytes': 'T',
        'Late erythroid': 'Erythroid'
    }

    ###SubPopulation
    mapping_sub = {
        'Tcm/Naive helper T cells': 'Tcm_Naive_CD4',
        'CD16+ NK cells': 'CD16_NK',
        'Classical monocytes': 'Classic_MONO',
        'Tem/Temra cytotoxic T cells': 'Tem_Temra_CD8',
        'Tem/Effector helper T cells': 'Tem_Effector_CD4',
        'Tcm/Naive cytotoxic T cells': 'Tcm_Naive_CD8',
        'Naive B cells': 'Naive_B',
        'Tem/Trm cytotoxic T cells': 'Tem_Trm_CD8',
        'Memory B cells': 'Memory_B',
        'B cells': 'Bcells',
        'Non-classical monocytes': 'NonClassic_MONO',
        'MAIT cells': 'MAIT',  # or 'MAIT' if you want to keep it separate
        'Regulatory T cells': 'Treg',
        'DC2': 'DC2',
        'pDC': 'pDC',
        'Intermediate macrophages': 'Int_Macrophage',
        'NK cells': 'NK',
        'Plasma cells': 'Plasma_B',
        'HSC/MPP': 'HSC/MPP',
        'Age-associated B cells': 'Aged_B',
        'DC1': 'DC1',
        'Megakaryocytes/platelets': 'Platelet',
        'Plasmablasts': 'Plasmablasts_B',
        'ILC': 'ILC',
        'CD8a/a': 'CD8a/a',
        'Double-positive thymocytes': 'T',
        'Late erythroid': 'Late_Erythroid'
    }

    # Map Major and Sub cell types
    adata_for_celltypist.obs['Major_CT'] = adata_for_celltypist.obs['majority_voting'].apply(lambda x: mapping.get(x, 'Others'))
    adata_for_celltypist.obs['Sub_CT'] = adata_for_celltypist.obs['majority_voting'].apply(lambda x: mapping_sub.get(x, 'Others'))

    # - post process
    adata.obs = adata.obs.join(adata_for_celltypist.obs[['Major_CT', 'Sub_CT']])
    adata.X = adata.layers["counts"]
    del adata.layers

    adata.obs['cell_type'] = adata.obs['Major_CT']
    major_cell_types = ["MONO", "NK", "B", "CD8T", "CD4T"]
    adata = adata[adata.obs['cell_type'].isin(major_cell_types)]
    
    return adata



# def binarize_age(obs):
#     obs = obs.copy()
#     obs['donor_age'] = obs['age'].astype(str) + '_' + obs['donor_id'].astype(str)
#     obs['age'] = pd.to_numeric(obs['age'], errors='coerce')
#     # min_age = obs.age.min()
#     # bins = [min_age, 35, 45, 55, 65, 75, 100]  
#     # age_groups = ['34-', '35_44', '45_54', '55_64', '65_75', '75+']  
#     # obs['age_group'] = pd.cut(obs['age'], bins=bins, labels=age_groups, right=False)
#     return obs
