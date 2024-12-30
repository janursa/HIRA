def create_dataset12():

    import seaborn as sns
    import anndata as ad 
    import pandas as pd
    import matplotlib.pyplot as plt
    import scanpy as sc
    import numpy as np
    from scipy.sparse import csr_matrix
    import sys
    from tqdm import tqdm

    sys.path.insert(0, '../')
    from task_grn_inference.src.utils.util import colors_blind
    from task_grn_inference.src.exp_analysis.helper import Exp_analysis, plot_interactions, create_interaction_info, create_interaction_df, jaccard_similarity_net, cosine_similarity_net, calculate_feature_distance, plot_cumulative_density
    from task_grn_inference.src.utils.util import basic_qc, read_gmt
    sys.path.insert(0, './')
    from src.helper import exp_plots

    cell_type_mapping = {
        "CD4Naive": "CD4+ T cells",
        "CD4TCM": "CD4+ T cells",
        "CD4TEM": "CD4+ T cells",
        "CD4CTL": "CD4+ T cells",
        "Treg": "CD4+ T cells",
        "CD4Proliferating": "CD4+ T cells",
        "CD8Naive": "CD8+ T cells",
        "CD8TCM": "CD8+ T cells",
        "CD8TEM": "CD8+ T cells",
        "TRAV1-2- CD8+ T cells": "CD8+ T cells",
        "CD8Proliferating": "CD8+ T cells",
        "NK": "NK cells",
        "NK_CD56bright": "NK cells",
        "NKProliferating": "NK cells",
        "Bnaive": "B cells",
        "Bmemory": "B cells",
        "Bintermediate": "B cells",
        "Plasmablast": "B cells",
        "CD14Mono": "Myeloid cells",
        "CD16Mono": "Myeloid cells",
        "cDC1": "Myeloid cells",
        "cDC2": "Myeloid cells",
        "pDC": "Myeloid cells",
        "ASDC": "Myeloid cells",
        "gdT": "gd T cells",
        "MAIT": "MAIT cells",
        "HSPC": "Progenitor cells",
        "Platelet": "Platelet",
        "Eryth": "Erythroid cells",
        "ILC": "ILC",
        "Doublet": "Doublet",
        "dnT": "DN T cells",
        "DN T cells": "DN T cells"
    }
    def formatize_obs_1(obs):
        obs = obs[['Batch', 'Donor_id','Age','Cluster_names', 'Sex']]
        obs.columns = ['batch', 'donor_id', 'age', 'cell_type', 'sex']
        obs['age_donor'] = obs['age'].astype(str) + '_' + obs['donor_id']

        # - remove age over 75
        obs = obs[obs.age<=75]

        # - binarize to age groups 
        bins = [23, 35, 45, 55, 65, 76]  # Define bins for age groups
        age_groups = ['34-', '35_44', '45_54', '55_64', '65_75']  # Define corresponding labels
        obs['age_group'] = pd.cut(obs['age'], bins=bins, labels=age_groups, right=False)
        return obs

    def formatize_obs_2(obs):
        obs = obs[['batch_info', 'donor_id','age','ct_major_published', 'sex']]
        obs.columns = ['batch', 'donor_id', 'age', 'cell_type', 'sex']
        obs['sex'] = obs['sex'].str.replace('M', 'Male')
        obs['sex'] = obs['sex'].str.replace('F', 'Female')
        obs['age_donor'] = obs['age'].astype(str) + '_' + obs['donor_id']

        # - remove age over 75
        obs = obs[obs.age<75]

        # - binarize to age groups 
        bins = [23, 35, 45, 55, 65, 75]  # Define bins for age groups
        age_groups = ['34-', '35_44', '45_54', '55_64', '65_75']  # Define corresponding labels
        obs['age_group'] = pd.cut(obs['age'], bins=bins, labels=age_groups, right=False)
        return obs

    adata_1 = ad.read_h5ad('/vol/projects/CIIM/Healthy_Single_Cell_Data/initial_data_downloaded/pbmc_ageing/raw_counts_h5ad/pbmc_gex_raw_with_var_obs.h5ad')
    obs_1 = pd.read_csv('/vol/projects/CIIM/Healthy_Single_Cell_Data/initial_data_downloaded/pbmc_ageing/all_pbmcs/all_pbmcs_metadata.csv', index_col=0)
    obs_1 = formatize_obs_1(obs_1)

    adata_2  = ad.read_h5ad('/vol/projects/CIIM/Healthy_Single_Cell_Data/count_matrix/data1_CMtx.h5ad')
    obs_2 = adata_2.obs
    obs_2 = formatize_obs_2(obs_2)

    obs = pd.concat([obs_1, obs_2])
    obs['cell_type'] = obs['cell_type'].map(lambda name: cell_type_mapping.get(name,name))

    # - filter samples with low cell counts
    n_cell_t = 1000

    sample_size = obs.groupby('age_donor', as_index=False).size()
    sample_size = sample_size[sample_size['size']>n_cell_t]

    obs = obs[obs.age_donor.isin(sample_size.age_donor)]
    # - filter for sex 
    obs = obs[obs['sex']=='Male']
    # - equalize donor size
    donor_size = 60
    def reduce_donors(ss):
        if ss.nunique() >= donor_size:
            ss = np.random.choice(ss.unique(), size=donor_size, replace=False)
        else:
            pass
        return ss
    # obs.groupby('age_group')['donor_id'].apply(reduce_donors)
    donors_f = obs.groupby('age_group')['donor_id'].apply(reduce_donors)
    obs = obs[obs.donor_id.isin(np.concatenate(donors_f.values))]
    # - assign batches 
    def assign_batches(group):
        unique_donors = group['donor_id'].unique()
        np.random.shuffle(unique_donors)
        
        n_half = int(len(unique_donors) / 2)
        batch1 = unique_donors[:n_half]
        batch2 = unique_donors[n_half:]

        batch_names = ['batch_1'] * n_half + ['batch_2'] * (len(unique_donors) - n_half)
        map_donor_to_batch = {donor: batch for donor, batch in zip(unique_donors, batch_names)}
        group['batch_group'] = group['donor_id'].map(map_donor_to_batch)
        
        return group

    obs = obs.groupby('age_group', group_keys=False).apply(assign_batches)
    # - 
    for i, age_group in enumerate(obs['age_group'].unique()):
        obs_sub = obs[obs['age_group'] == age_group]

        adata_1_sub = adata_1[adata_1.obs.index.isin(obs_sub.index), :]
        adata_2_sub = adata_2[adata_2.obs.index.isin(obs_sub.index), :]
        adata_sub = ad.concat([adata_1_sub, adata_2_sub], axis=0, join="outer")
        
        adata_sub = basic_qc(adata_sub, min_genes_per_cell = 200, max_genes_per_cell = 5000, min_cells_per_gene = 1000)
        
        if i == 0:
            adata_all = adata_sub
        else:
            adata_all = ad.concat([adata_sub, adata_all], axis=0, join="outer")
        print(adata_all)
    adata_all.obs = adata_all.obs.merge(obs, left_index=True, right_index=True, how='left', suffixes=['','_'])
    adata_all.obs = adata_all.obs[['batch', 'donor_id', 'age', 'cell_type', 'sex', 'age_donor', 'age_group']]

    adata_all.write('input/dataset_12.h5ad') 
