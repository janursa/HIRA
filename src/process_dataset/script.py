
import sys
import os 
from tqdm import tqdm
import argparse
import anndata as ad
import numpy as np
import pandas as pd

## VIASH START
parser = argparse.ArgumentParser()

parser.add_argument(
    '--raw_dataset_file',
    type=str,
    required=True,
    help="The input dataset"
)
parser.add_argument('--processed_dataset_file', 
    type=str,
    required=True,
    help="Processed dataset file"
    )
parser.add_argument('--bulk_dataset_file', 
    type=str,
    required=True,
    help="Processed dataset file in bulk format"
    )
    
parser.add_argument('--n_cell_t', 
    type=int,
    required=True,
    help="Number of threshold for cell count per donor to include"
    )

parser.add_argument(
    '--downsample',
    action='store_true',
    help="Whether to equalize cell counts and donor sizes. Default is False."
)
parser.add_argument(
    '--only_male',
    action='store_true',
    help="Whether to subset the data to only males. Default is False."
)

args = parser.parse_args()

par_input = vars(args)

par = {
    'equalize_donor_size': True,
}

for key in par_input.keys():
    if par_input[key] is not None:
        par[key] = par_input[key]

## VIASH END
meta = {
    'resources_dir' : 'src/utils/'
}
sys.path.append(meta['resources_dir'])

from util import basic_qc, bulkify_adata

def binarize_age(obs):
    obs = obs.copy()
    obs['age_donor'] = obs['age'].astype(str) + '_' + obs['donor_id'].astype(str)
    # obs = obs[obs.age<=75]
    min_age = obs.age.min()
    bins = [min_age, 35, 45, 55, 65, 75, 100]  
    age_groups = ['34-', '35_44', '45_54', '55_64', '65_75', '75+']  
    obs['age_group'] = pd.cut(obs['age'], bins=bins, labels=age_groups, right=False)
    return obs

def process_obs(obs, par):
    # - cleanup
    obs_cols = ['orig.ident', 'donor_id', 'age', 'Major_CT', 'sex']
    obs = obs[obs_cols]
    obs.columns = ['dataset', 'donor_id', 'age', 'cell_type', 'sex']
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

    cols = ['batch_group', 'donor_id', 'age', 'cell_type', 'sex', 'age_donor', 'age_group']
    if 'dataset' in obs.columns:
        cols = cols + ['dataset']
        
    obs = obs[cols]

    return obs

def main(par):
    print('Processing dataset...')
    adata = ad.read_h5ad(par['raw_dataset_file'], backed='r')
    obs = adata.obs.copy()
    obs = process_obs(obs=obs, par=par)
    assert not obs.isna().any().any()
    obs = obs.reindex(adata.obs.index)

    adata.obs = obs.copy()
    print('Processed adata size: ', adata.shape)
    
    print(f"Saving adata in progress. Loading adata...")
    adata = adata.to_memory()
    adata = basic_qc(adata, min_cells_per_gene=100, min_genes_per_cell=10)
    print(f"Saving adata to {par['processed_dataset_file']}")
    adata.write(par['processed_dataset_file'])

    # - bulk dataset
    print('Creating bulk dataset...')
    adata = bulkify_adata(adata)
    adata.write(par['bulk_dataset_file'])

if __name__ == '__main__':
    print(par, flush=True)
    
    main(par)


        