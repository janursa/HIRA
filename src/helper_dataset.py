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
    obs['age_donor'] = obs['age'].astype(str) + '_' + obs['donor_id']
    obs = obs[obs.age<=75]
    min_age = obs.age.min()
    bins = [min_age, 35, 45, 55, 65, 75]  
    age_groups = ['34-', '35_44', '45_54', '55_64', '65_75']  
    obs['age_group'] = pd.cut(obs['age'], bins=bins, labels=age_groups, right=False)
    return obs

def formatize_obs_dataset_1(obs, dataset):
    obs = obs[['Batch', 'Donor_id','Age','Cluster_names', 'Sex']]
    obs.columns = ['batch', 'donor_id', 'age', 'cell_type', 'sex']
    obs = binarize_age(obs)
    obs['dataset'] = dataset
    return obs

def formatize_obs_dataset_2(obs, dataset):
    obs = obs.copy()
    obs = obs[['batch_info', 'donor_id','age','ct_major_published', 'sex']]
    obs.columns = ['batch', 'donor_id', 'age', 'cell_type', 'sex']
    obs['sex'] = obs['sex'].str.replace('M', 'Male')
    obs['sex'] = obs['sex'].str.replace('F', 'Female')
    obs = binarize_age(obs)
    obs['dataset'] = dataset
    return obs


def age_donor_assign(obs, sample_size=4000):
    def stratified_sample_multiple_times(group, sample_size):
        from sklearn.model_selection import StratifiedGroupKFold

        cell_counts = group['cell_type'].value_counts()
        total_cells = len(group)
        samples = []
        
        # Determine how many 4000-cell samples can be drawn
        num_full_samples = total_cells // sample_size
        remaining_cells = total_cells % sample_size  # Remaining cells after full sample extractions
        
        def stratified_sample(group, target_count, replace=False):
            # Same stratified sampling as before
            cell_counts = group['cell_type'].value_counts()
            proportions = cell_counts / cell_counts.sum()
            samples = (proportions * target_count).round().astype(int)
            samples = samples[samples > 0]
            
            sampled_data = []
            for cell_type, n_samples in samples.items():
                sampled_data.append(group[group['cell_type'] == cell_type].sample(n=n_samples, replace=replace))
            return pd.concat(sampled_data)

        # Sample full sets of 4000 cells
        i = 0
        for _ in range(num_full_samples):
            sampled_data = stratified_sample(group, sample_size)
            sampled_data['sample'] = sampled_data['age_donor']+f'_{i}'
            samples.append(sampled_data)
            group = group.drop(sampled_data.index)  # Drop the sampled cells to avoid resampling them
            i+=1
        # print(total_cells, len(samples))
        
        # Sample the remaining cells if any
        # print(sampled_data,remaining_cells)
        if remaining_cells >= (sample_size*(3/4)):
            sampled_data = group
            sampled_data['sample'] = sampled_data['age_donor']+f'_{i}'
            samples.append(sampled_data)
        
        return pd.concat(samples)

    obs = obs.groupby('age_donor').apply(lambda x: stratified_sample_multiple_times(x, sample_size=sample_size)).reset_index(level=0, drop=True)
    obs['age_donor'] = obs['sample']

    return obs

def create_dataset(dataset='pbmc_ageing', only_male=True, save_file='input/dataset_1.h5ad', downsample=False):
    # - downsample -> first apply min donors by preserving donors with highest cell count, then down sample the cell count
    equalize_donor_size=True
    print(save_file, downsample)
    if dataset=='pbmc_ageing':
        adata = ad.read_h5ad('/vol/projects/CIIM/Healthy_Single_Cell_Data/initial_data_downloaded/pbmc_ageing/raw_counts_h5ad/pbmc_gex_raw_with_var_obs.h5ad')
        obs = pd.read_csv('/vol/projects/CIIM/Healthy_Single_Cell_Data/initial_data_downloaded/pbmc_ageing/all_pbmcs/all_pbmcs_metadata.csv', index_col=0)
        obs = formatize_obs_dataset_1(obs, dataset)
    elif dataset=='data1':
        adata  = ad.read_h5ad('/vol/projects/CIIM/Healthy_Single_Cell_Data/count_matrix/data1_CMtx.h5ad')
        obs = adata.obs
        obs = formatize_obs_dataset_2(obs, dataset)
    else:
        raise ValueError('define first')

    obs = pd.concat([obs]) # only data 1
    obs['cell_type'] = obs['cell_type'].map(lambda name: cell_type_mapping.get(name,name))
    obs= obs[obs['cell_type'].isin(major_cell_types)]

    # - filter samples with low cell counts
    n_cell_t = 500

    sample_size = obs.groupby('age_donor', as_index=False).size()
    sample_size = sample_size[sample_size['size']>n_cell_t]

    obs = obs[obs.age_donor.isin(sample_size.age_donor)]

    # - filter for sex 
    if only_male:
        obs = obs[obs['sex']=='Male']
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

    # - min donor size per age group
    min_donors = obs.groupby(['batch_group']).apply(lambda df: df.groupby('age_group')['donor_id'].nunique()).min(axis=1)
    print(min_donors)
    # - min sample size per age group
    min_sample_size = obs.groupby(['batch_group']).apply(lambda df: df.groupby('age_group').size()).min(axis=1)
    min_sample_size

    # - downsample -> first apply min donors by preserving donors with highest cell count, then down sample the cell count
    if downsample:
        barcodes = []
        for batch_group in obs['batch_group'].unique():
            obs_batch = obs[obs['batch_group']==batch_group]
            
            min_donor = min_donors[batch_group]
            min_sample = min_sample_size[batch_group]
            for age_group in obs['age_group'].unique():
                obs_age = obs_batch[obs_batch['age_group'] == age_group]
                if equalize_donor_size:
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



    def final_process(adata):
        adata = basic_qc(adata, min_genes_per_cell = 200, max_genes_per_cell = 5000, min_cells_per_gene = 5000)
        adata.obs = adata.obs.merge(obs, left_index=True, right_index=True, how='left', suffixes=['','_'])
        adata.obs = adata.obs[['batch_group', 'donor_id', 'age', 'cell_type', 'sex', 'age_donor', 'age_group', 'batch', 'dataset']]
        return adata

    adata = adata[adata.obs.index.isin(obs.index), :]
    adata = final_process(adata) # to make the memory consumption of merging more affordable

    assert not adata.obs.isna().any().any()
    del adata.raw
    adata.write(save_file)



if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help="Name of the datsaet (data1, of pbmc_ageing)."
    )

    parser.add_argument(
        '--save_file',
        type=str,
        required=True,
        help="Path to save the dataset file (e.g., 'input/dataset_1.h5ad')."
    )

    parser.add_argument(
        '--downsample',
        action='store_true',
        help="Whether to equalize donor sizes. Default is False."
    )


    args = parser.parse_args()
    save_file=args.save_file
    downsample=args.downsample
    dataset=args.dataset
    
    print(save_file, dataset, downsample)
    create_dataset(save_file=save_file, dataset=dataset, downsample=downsample)
