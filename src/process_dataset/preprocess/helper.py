
import sys
import os 
from tqdm import tqdm
import argparse
import anndata as ad
import numpy as np
import pandas as pd
def binarize_age(obs):
    obs = obs.copy()
    obs['age_donor'] = obs['age'].astype(str) + '_' + obs['donor_id'].astype(str)
    obs['age'] = pd.to_numeric(obs['age'], errors='coerce')
    # min_age = obs.age.min()
    # bins = [min_age, 35, 45, 55, 65, 75, 100]  
    # age_groups = ['34-', '35_44', '45_54', '55_64', '65_75', '75+']  
    # obs['age_group'] = pd.cut(obs['age'], bins=bins, labels=age_groups, right=False)
    return obs

def process_obs(obs, par):
    # - cleanup
    # obs_cols = ['orig.ident', 'donor_id', 'age', 'Major_CT', 'Sub_CT', 'sex']
    # obs = obs[obs_cols]
    obs['dataset'] = obs['orig.ident']
    obs['cell_type'] = obs['Major_CT']
    
    obs = binarize_age(obs)
    major_cell_types = ["MONO", "NK", "B", "CD8T", "CD4T"]
    obs = obs[obs['cell_type'].isin(major_cell_types)]

    # - actual processing
    print('Original size: ', obs.shape)
    obs['donor_id'] = obs['donor_id'].astype(str)
    # - filter samples with low cell counts
    sample_size = obs.groupby('age_donor', as_index=False).size()
    sample_size = sample_size[sample_size['size']>par['n_cell_t']]
    obs = obs[obs.age_donor.isin(sample_size.age_donor)]
    print('size after filtering for donor sinlge cell count: ', obs.shape)
    # - filter for sex 
    gender = par['gender']
    if gender == 'both':
        pass
    else:
        if gender in ['M', 'F']:
            if gender not in obs['sex'].unique():
                obs['sex'] = obs['sex'].map({'Male':'M', 'Female':'F'})
        
        assert gender in obs['sex'].unique(), f"{gender} not in obs['sex]"
        obs = obs[obs['sex']==gender]
        print('size after filtering for gene given: ', obs.shape)

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
    # obs = obs.groupby('age_group', group_keys=False).apply(assign_batches)
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
        
    # cols = ['batch_group', 'donor_id', 'age', 'cell_type', 'sex', 'age_donor', 'age_group', 'Major_CT', 'Sub_CT']
    # if 'dataset' in obs.columns:
    #     cols = obs.columns + ['dataset']
        
    # obs = obs[cols]

    return obs