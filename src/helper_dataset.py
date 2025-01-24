
import seaborn as sns
import anndata as ad 
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
from scipy.sparse import csr_matrix
import sys
from tqdm import tqdm
import argparse
sys.path.insert(0, '../')
from task_grn_inference.src.utils.util import colors_blind
from task_grn_inference.src.exp_analysis.helper import Exp_analysis, plot_interactions, create_interaction_info, create_interaction_df, jaccard_similarity_net, cosine_similarity_net, calculate_feature_distance, plot_cumulative_density
from task_grn_inference.src.utils.util import basic_qc, read_gmt
sys.path.insert(0, './')
from src.helper import cell_type_mapping, major_cell_types
def binarize_age(obs):
    obs['age_donor'] = obs['age'].astype(str) + '_' + obs['donor_id'].astype(str)
    # obs = obs[obs.age<=75]
    min_age = obs.age.min()
    bins = [min_age, 35, 45, 55, 65, 75, 100]  
    age_groups = ['34-', '35_44', '45_54', '55_64', '65_75', '75+']  
    obs['age_group'] = pd.cut(obs['age'], bins=bins, labels=age_groups, right=False)
    return obs
def formatize_obs_all(obs):
    major_cell_types_ali = ["MONO", "NK", "B", "CD8T", "CD4T"]
    obs = obs[['orig.ident', 'donor_id','age','Major_CT', 'sex']]
    obs.columns = ['dataset', 'donor_id', 'age', 'cell_type', 'sex']
    obs = obs[obs['cell_type'].isin(major_cell_types_ali)]
    obs = binarize_age(obs)
    return obs
def formatize_obs_pbmc_ageing(obs):
    obs = obs[['Batch', 'Donor_id','Age','Cluster_names', 'Sex']]
    obs.columns = ['batch', 'donor_id', 'age', 'cell_type', 'sex']
    obs['sex'] = obs['sex'].str.replace('Male','M')
    obs['sex'] = obs['sex'].str.replace('Female','F')
    obs = binarize_age(obs)
    obs['cell_type'] = obs['cell_type'].map(lambda name: cell_type_mapping.get(name,name))
    obs = obs[obs['cell_type'].isin(major_cell_types)]
    return obs
def formatize_obs_data1(obs):
    obs = obs.copy()
    obs = obs[['batch_info', 'donor_id','age','ct_major_published', 'sex']]
    obs.columns = ['batch', 'donor_id', 'age', 'cell_type', 'sex']
    obs = binarize_age(obs)
    obs['cell_type'] = obs['cell_type'].map(lambda name: cell_type_mapping.get(name,name))
    obs = obs[obs['cell_type'].isin(major_cell_types)]
    return obs
def clean_all_data():
    dataset_file  = "/vol/projects/CIIM/Healthy_Single_Cell_Data/output/processed_data/processed_data.h5ad"
    adata = ad.read_h5ad(dataset_file)
    adata.X = adata.layers["counts"]
    obs_cols = ['orig.ident', 'donor_id', 'age', 'Major_CT', 'sex']
    adata.obs = adata.obs[obs_cols]
    adata.var = adata.var[[]]
    del adata.layers 
    del adata.obsm
    del adata.varm
    del adata.uns
    del adata.obsp
def clean_pbmc_ageing():
    adata = sc.read_h5ad('/vol/projects/CIIM/Healthy_Single_Cell_Data/initial_data_downloaded/pbmc_ageing/raw_counts_h5ad/pbmc_gex_raw_with_var_obs.h5ad', backed='r')
    meta = pd.read_csv('/vol/projects/CIIM/Healthy_Single_Cell_Data/initial_data_downloaded/pbmc_ageing/all_pbmcs/all_pbmcs_metadata.csv', index_col=0)
    adata.obs = adata.obs[[]].join(meta[['orig.ident', 'Donor_id', 'Age', 'Cluster_names', 'Sex', 'Batch']], how='left')
    del adata.uns
    adata.write('/vol/projects/jnourisa/adata_pbmc_ageing.h5ad')
# def age_donor_assign(obs, sample_size=4000):
#     def stratified_sample_multiple_times(group, sample_size):
#         from sklearn.model_selection import StratifiedGroupKFold
#         cell_counts = group['cell_type'].value_counts()
#         total_cells = len(group)
#         samples = []
        
#         # Determine how many 4000-cell samples can be drawn
#         num_full_samples = total_cells // sample_size
#         remaining_cells = total_cells % sample_size  # Remaining cells after full sample extractions
        
#         def stratified_sample(group, target_count, replace=False):
#             # Same stratified sampling as before
#             cell_counts = group['cell_type'].value_counts()
#             proportions = cell_counts / cell_counts.sum()
#             samples = (proportions * target_count).round().astype(int)
#             samples = samples[samples > 0]
            
#             sampled_data = []
#             for cell_type, n_samples in samples.items():
#                 sampled_data.append(group[group['cell_type'] == cell_type].sample(n=n_samples, replace=replace))
#             return pd.concat(sampled_data)
#         # Sample full sets of 4000 cells
#         i = 0
#         for _ in range(num_full_samples):
#             sampled_data = stratified_sample(group, sample_size)
#             sampled_data['sample'] = sampled_data['age_donor']+f'_{i}'
#             samples.append(sampled_data)
#             group = group.drop(sampled_data.index)  # Drop the sampled cells to avoid resampling them
#             i+=1
#         # print(total_cells, len(samples))
        
#         # Sample the remaining cells if any
#         # print(sampled_data,remaining_cells)
#         if remaining_cells >= (sample_size*(3/4)):
#             sampled_data = group
#             sampled_data['sample'] = sampled_data['age_donor']+f'_{i}'
#             samples.append(sampled_data)
        
#         return pd.concat(samples)
#     obs = obs.groupby('age_donor').apply(lambda x: stratified_sample_multiple_times(x, sample_size=sample_size)).reset_index(level=0, drop=True)
#     obs['age_donor'] = obs['sample']
#     return obs
def process_obs(obs, par):
    print('Original size: ', obs.shape)
    obs['donor_id'] = obs['donor_id'].astype(str)
    # - filter samples with low cell counts
    sample_size = obs.groupby('age_donor', as_index=False).size()
    sample_size = sample_size[sample_size['size']>par['n_cell_t']]
    obs = obs[obs.age_donor.isin(sample_size.age_donor)]
    print('size after filtering for donor sinlge cell count: ', obs.shape)
    # - filter for sex 
    if par['only_male']:
        obs = obs[obs['sex']=='M']
        print('size after filtering for Male only: ', obs.shape)
    # - asign batches
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
    # - downsample -> first apply min donors by preserving donors with highest cell count, then down sample the cell count
    if par['downsample']:
        # - min donor size per age group
        min_donors = obs.groupby(['batch_group']).apply(lambda df: df.groupby('age_group')['donor_id'].nunique()).min(axis=1)
        print(min_donors)
        # - min sample size per age group
        min_sample_size = obs.groupby(['batch_group']).apply(lambda df: df.groupby('age_group').size()).min(axis=1)
        barcodes = []
        for batch_group in obs['batch_group'].unique():
            obs_batch = obs[obs['batch_group']==batch_group]
            
            min_donor = min_donors[batch_group]
            min_sample = min_sample_size[batch_group]
            for age_group in obs['age_group'].unique():
                obs_age = obs_batch[obs_batch['age_group'] == age_group]
                if par['equalize_donor_size']:
                    # - downsample based on donor count
                    selected_donors = obs_age.groupby('donor_id').size().sort_values()[::-1][:min_donor].index
                    obs_age = obs_age[obs_age['donor_id'].isin(selected_donors)]
                
                # - downsample based on count
                n_sample = len(obs_age)
                if n_sample > min_sample:
                    ratio_to_keep = min_sample/len(obs_age) 
                    # Group by 'donor_id' and 'cell_type', then sample based on the ratio_to_keep
                    obs_age = obs_age.groupby(['donor_id', 'cell_type'], group_keys=False).apply(
                        lambda group: group.sample(frac=ratio_to_keep, random_state=42)
                    )
                barcodes.append(obs_age.index)
        barcodes = np.concatenate(barcodes)
        obs = obs[obs.index.isin(barcodes)]
    assert not obs.isna().any().any()
    obs = obs[['batch_group', 'donor_id', 'age', 'cell_type', 'sex', 'age_donor', 'age_group', 'dataset']]

    return obs
def process_dataset(par):
    # - read dataset
    adata = ad.read_h5ad(par['dataset_file'], backed='r')
    if par['dataset']=='pbmc_ageing':
        obs = formatize_obs_pbmc_ageing(adata.obs.copy())
    elif par['dataset']=='data1':
        obs = formatize_obs_data1(adata.obs.copy())
    elif par['dataset']=='all':
        obs = formatize_obs_all(adata.obs.copy())
    else:
        raise ValueError('Invalid dataset name')    
    obs = process_obs(obs=obs.copy(), par=par)
    assert not obs.isna().any().any()
    return obs


if __name__ == '__main__':
    clean_all_data()