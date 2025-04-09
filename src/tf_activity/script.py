import anndata as ad
import pandas as pd
import sys
import numpy as np
import scanpy as sc 
from scipy import stats
from scipy.stats import pearsonr
import os
import argparse
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from pandas.api.types import CategoricalDtype
from ciim.src.tf_activity.helper import calculate_tf_activity, find_central_tfs, \
    find_robust_predictors, linear_association_with_age, run_pseudotime_analysis, net_lambda,  \
    tf_all, summary_func, adata_lambda, compute_trend, \
    read_tf_acts, write_tf_acts, determine_stats_disease
from ciim.src.tf_activity.meta_analysis.helper import wrapper_meta_analysis

from ciim.src.common import cell_types, datasets, datasets_healthy, datasets_all, v_datasets, datasets_disease, mapping_minor_2_major

def determine_std(adata):
    # Ensure .X is dense
    if isinstance(adata.X, np.ndarray):
        X_dense = adata.X
    elif hasattr(adata.X, "todense"):  # Check if it's a sparse matrix
        X_dense = adata.X.todense().A
    else:
        raise TypeError("Unexpected type for adata.X: {}".format(type(adata.X)))

    # Create DataFrame
    df = pd.DataFrame(X_dense, index=adata.obs.index, columns=adata.var_names)
    genes = df.columns
    # Add metadata
    df["age"] = adata.obs["age"].values.astype(str)
    df["donor_id"] = adata.obs["donor_id"].values.astype(str)

    # Compute standard deviation for each gene per donor-age combination
    std_df = df.groupby(["donor_id", 'age']).std()
    cell_count = df.groupby(["donor_id", 'age']).size()
    std_adata = sc.AnnData(std_df.values)
    
    # Set the new AnnData object's obs and var from the std_df
    std_adata.obs = std_df.reset_index()[["donor_id", 'age']]
    std_adata.obs['cell_count'] = cell_count.values
    std_adata.obs['age'] = pd.to_numeric(std_adata.obs['age'], errors='coerce')
    std_adata.var = pd.DataFrame(index=genes)  # This sets the genes as the variables
    std_adata.X = np.nan_to_num(std_adata.X, nan=0)
    return std_adata

def wrapper_robust_predictor_tfs(par, cell_types, datasets, predictor_age_q=.8):
    predictor_tfs_store = []
    mapped_major_cell_types = np.unique(list(mapping_minor_2_major.values()))
    for cell_type in cell_types:
        print('Cell type:', cell_type)
        for dataset in datasets:
            major_cell_type = mapping_minor_2_major.get(cell_type, cell_type)
            if major_cell_type not in mapped_major_cell_types:
                print('No mapping for', cell_type, 'to major cell type. Skipping it.')
                continue

            tf_acts = read_tf_acts(dataset, major_cell_type, type=par['type'], read_dir=par['tf_acts_dir'])

            if par['gender_stratification'] != 'all':
                tf_acts.obs['sex'] = tf_acts.obs['sex'].apply(lambda name: {'Male': 'M', 'Female': 'F'}.get(name, name))
                assert par['gender_stratification'] in tf_acts.obs['sex'].unique(), f"{par['gender_stratification']} not in {tf_acts.obs['sex'].unique()}"
                tf_acts = tf_acts[tf_acts.obs['sex']==par['gender_stratification']]

            tf_acts = tf_acts[tf_acts.obs[par['cell_type_resolution']] == cell_type]
            if len(tf_acts) == 0:
                print('No tf_acts for', dataset, cell_type)
                continue
            top_tfs, r2 = find_robust_predictors(
                                    tf_acts, target='age', top_q=predictor_age_q 
                                )
            for tf in top_tfs:
                predictor_tfs_store.append({
                    'dataset': dataset,
                    'cell_type': cell_type,
                    'tf': tf,
                    'r2': r2
                })
    predictor_tfs_df = pd.DataFrame(predictor_tfs_store)
    return predictor_tfs_df

def wrapper_main(par):
    from ciim.src.tf_activity.helper import find_robust_predictors
    stats_all = pd.read_csv(par['stats_tfs'])
    cell_types = stats_all['cell_type'].unique()
    # - identify robust predictors
    predictor_tfs_df = wrapper_robust_predictor_tfs(par, cell_types, datasets=datasets)
    # - take the union of the robust predictors across datasets
    prior_tfs_dict = {}
    for cell_type in cell_types:
        df = predictor_tfs_df[predictor_tfs_df['cell_type'] == cell_type]
        df = df.groupby('tf').size().sort_values(ascending=False)
        intersect = df[df > 1].index
        union = df.index
        if len(intersect) == 0:
            print('No common predictors for', cell_type)
            continue
        print('Robust TFs', cell_type, ' intersection : ', len(intersect), ' union: ', len(union))
        prior_tfs_dict[cell_type] = union
    # - discovery 
    print('Discovery ...')
    
    stats_store = []
    for cell_type in prior_tfs_dict.keys():
        stats = stats_all[(stats_all['cell_type'] == cell_type) & (stats_all['tf'].isin(prior_tfs_dict[cell_type])) & (stats_all['dataset'].isin(datasets))]
        if len(stats) == 0:
            print('No stats for', cell_type, ' skipping it')
            continue
        if stats.groupby('tf').size().max()<2:
            print('Not enough mutual TFs for ', cell_type, ' skipping it')
            continue
        meta_stats = run_meta_analysis(stats, datasets, tmp_dir=par['tmp_dir'])
        
        stats_store.append(meta_stats)
    stats_discovery = pd.concat(stats_store)

    stats_discovery_sig = stats_discovery[stats_discovery['meta_p_adj'] < 0.05]
    stats_discovery_consistent = stats_discovery_sig[stats_discovery_sig['trend']!='Inconsistent'][['cell_type', 'tf', 'meta_p_adj', 'trend']].drop_duplicates()

    # - validation
    print('Validation ...')
    stats_store = []
    for cell_type in stats_discovery_consistent['cell_type'].unique():
        discovery_tfs = stats_discovery_consistent[stats_discovery_consistent['cell_type'] == cell_type]['tf'].unique()
        if len(discovery_tfs) == 0:
            print('No discovery TFs for cell type', cell_type)
            continue
        stats = stats_all[(stats_all['cell_type'] == cell_type) & (stats_all['tf'].isin(discovery_tfs))]
        stats = run_meta_analysis(stats, v_datasets, tmp_dir=par['tmp_dir'])
        stats_store.append(stats)
    stats_validation = pd.concat(stats_store)

    # - disease
    print('Disease ...')
    stats_validation_sig = stats_validation[stats_validation['meta_p_adj'] < 0.05]
    stats_validation_consistent = stats_validation_sig[stats_validation_sig['trend']!='Inconsistent']
    stats_store = []
    for cell_type in stats_validation_consistent['cell_type'].unique():
        valid_tfs = stats_validation_consistent[stats_validation_consistent['cell_type'] == cell_type]['tf'].unique()
        for dataset in datasets_disease:
            stats = stats_all[(stats_all['cell_type'] == cell_type) & (stats_all['tf'].isin(valid_tfs)) & (stats_all['dataset'] == dataset)]
            if len(stats) == 0:
                print('No stats for', cell_type, dataset)
                continue
            stats["p_value_adj"] = multipletests(stats["p_value"], method="fdr_bh")[1]
            # stats['trend'] = ["Increase in disease" if x > 0 else "Decrease in disease" for x in stats['slope']]
            stats["dataset"] = dataset 
            stats_store.append(stats)
        
    stats_disease = pd.concat(stats_store)

    # - combine
    stats_discovery['analysis'] = 'discovery'
    stats_validation['analysis'] = 'validation'
    stats_disease['analysis'] = 'disease'
    stats_combined = pd.concat([stats_discovery, stats_validation, stats_disease])

    #- save
    stats_combined.to_csv(par['stats_all'], index=False)


def wrapper_tf_activity(cell_types, datasets, type='bulk', tf_acts_dir='output/tf_activation/tf_acts/'):
    # --------- load data
    print('Loading data...')
    adata_dict = {dataset: adata_lambda(dataset, type) for dataset in datasets}

    # - calculate tf activity for all datasets
    print('Calculating TF activity...')
    stats_store = []
    for cell_type in tqdm(cell_types, desc='cell types'):
        # ----------- calculate tf activity for all datasets
        for dataset in datasets:
            adata = adata_dict[dataset][adata_dict[dataset].obs['cell_type']==cell_type]
            net = net_lambda(dataset, cell_type)

            if adata.shape[0] < 10:
                continue
            tf_acts = calculate_tf_activity(adata, net)
            
            tf_acts = tf_acts[tf_acts.obs['age'].isna()==False] # there is a bug in the code that causes age to be NaN
            # print(cell_type, tf_acts.shape)
            write_tf_acts(tf_acts, dataset, cell_type, type, tf_acts_dir)
# - add meta p values
def run_meta_analysis(stats_all, datasets, tmp_dir='output/tmp/'):
    assert len(datasets) > 1, 'Meta analysis is not needed for one dataset'
    assert stats_all.shape[0]> 0, 'No stats for meta analysis'
    stats_all = stats_all[stats_all['dataset'].isin(datasets)]

    from ciim.src.tf_activity.meta_analysis.helper import wrapper_meta_analysis    
    
    print('Meta analysis...')
    stats_all_c = stats_all.copy()
    stats_all_c.rename(columns={'p_value': 'pvalue', 'tf':'gene'}, inplace=True)
    meta_analysis_type = 'max'
    df_meta_all = wrapper_meta_analysis(stats_all_c, type=meta_analysis_type, temp_dir=tmp_dir)
    df_meta_all.rename(columns={'gene': 'tf'}, inplace=True)
    if df_meta_all.shape[0]==0:
        print(f"No meta analysis results for {stats_all['cell_type'].unique()} {datasets}")
        return None
    assert 'meta_p_adj' not in stats_all.columns, 'meta_p_adj already in df_meta_all'
    stats_all = stats_all.merge(df_meta_all, on=['tf', 'cell_type'], how='left')

    pval_col = 'meta_p_adj'
    stats_all = compute_trend(stats_all, pval_col=pval_col, slope_col='slope')
    # - save
    return stats_all
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

def tf_association_with_age(par, cell_types, datasets, tfs_dict=None):
    from ciim.src.tf_activity.helper import read_tf_acts
    # - calculate tf activity for all datasets
    print('TF association with age/disease...')
    stats_store = []
    for cell_type in tqdm(cell_types, desc='cell types'):
        # ----------- calculate tf activity for all datasets
        for dataset in datasets:
            tf_acts = read_tf_acts(dataset, cell_type, par['type'], read_dir=par['tf_acts_dir'])
            if par['gender_stratification'] != 'all':
                tf_acts.obs['sex'] = tf_acts.obs['sex'].apply(lambda name: {'Male': 'M', 'Female': 'F'}.get(name, name))
                assert par['gender_stratification'] in tf_acts.obs['sex'].unique(), f"{par['gender_stratification']} not in {tf_acts.obs['sex'].unique()}"
                tf_acts = tf_acts[tf_acts.obs['sex']==par['gender_stratification']]
            if 'disease' in tf_acts.obs.columns:
                disease_flag = True
            else:
                disease_flag = False

            # - add which cell type resolution to run the analysis
            tf_acts.obs['cell_type_resolution'] = tf_acts.obs[par['cell_type_resolution']]
            
            
            for cell_type_resolution in tf_acts.obs['cell_type_resolution'].unique():
                tf_acts_sub = tf_acts[tf_acts.obs['cell_type_resolution']==cell_type_resolution]
                if tf_acts_sub.shape[0] < 10:
                    print('Not enough samples for', cell_type, dataset, cell_type_resolution)
                    continue
                # - for std, determine std and then use it for the rest of the analysis
                if type == 'std':
                    tf_acts_sub = determine_std(tf_acts_sub)
                
                if disease_flag:
                    stats = determine_stats_disease(tf_acts)
                else:
                    stats = linear_association_with_age(tf_acts_sub, genes=tf_acts_sub.var_names)
                
                stats['dataset'] = dataset
                stats['cell_type'] = cell_type_resolution
                
                stats_store.append(stats)
        
    stats_all = pd.concat(stats_store)
    return stats_all
def wrapper_target_association_age(par):
    stats_store = []
    for dataset in tqdm(datasets_healthy, desc='datasets'):
    # for dataset in ['data12']:
        adata = adata_lambda(dataset)
        for cell_type in tqdm(cell_types, desc='cell types'):
        # for cell_type in tqdm(['CD8T'], desc='cell types'):
            net = net_lambda(dataset, cell_type)
            targets = net['target'].unique()
            adata_t = adata[adata.obs['cell_type']==cell_type].copy()
            stats = linear_association_with_age(adata_t, targets)
            stats.rename(columns={'tf': 'target'}, inplace=True) 
            stats['dataset'] = dataset
            stats['cell_type'] = cell_type
            stats_store.append(stats)
    stats_target_all = pd.concat(stats_store)
    stats_target_all.to_csv(par['stats_targets'], index=False)

def run_workflow_bulk(par):
    print(par)
    if True: # Run once for bulk and sc
        # - step 1: calculate TF activity
        if True: 
            print('Calculating TF activity...')
            wrapper_tf_activity(cell_types, datasets_all, type=par['type'], tf_acts_dir=par['tf_acts_dir'])
        # - step 4:
        if True:
            print('Target association with age...')
            wrapper_target_association_age(par)

        
    # - step 2: calculate TF association with age
    if True:
        stats_tfs_all = tf_association_with_age(par, cell_types, datasets=datasets_all)
        stats_tfs_all.to_csv(par['stats_tfs'], index=False)
    if True:
        print('TF discovery/validation...')
        wrapper_main(par)
    
    

if __name__ == '__main__':
    if False:
        # ----- bulk TF activity
        par = {
            'type': 'bulk',
            'cell_type_resolution': 'Major_CT',
            'gender_stratification': 'all',
            'tf_acts_dir': 'output/tf_activation/tf_acts/', 
            'stats_tfs': 'output/tf_activation/stats_tfs_minor.csv',
            'stats_targets': 'output/tf_activation/stats_targets_bulk.csv',
            'stats_all': 'output/tf_activation/stats_all_bulk.csv', 
            'tmp_dir': 'output/tmp/',
        }
        os.makedirs(par['tf_acts_dir'], exist_ok=True)
        run_workflow_bulk(par)
    if False:
        # ----- bulk TF activity: minor
        par = {
            'type': 'bulk_minor',
            'cell_type_resolution': 'Sub_CT',
            'gender_stratification': 'all',
            'tf_acts_dir': 'output/tf_activation/tf_acts/', 
            'stats_tfs': 'output/tf_activation/stats_tfs_bulk_minor.csv',
            'stats_targets': 'output/tf_activation/stats_targets_bulk_minor.csv',
            'stats_all': 'output/tf_activation/stats_all_bulk_minor.csv', 
            'tmp_dir': 'output/tmp/',
        }
        os.makedirs(par['tf_acts_dir'], exist_ok=True)
        run_workflow_bulk(par)
    if False:
        # ----- bulk TF activity: gender
        for gender in ['M', 'F']:
            
            par = {
                'type': 'bulk',
                'cell_type_resolution': 'Major_CT',
                'gender_stratification': gender,
                'tf_acts_dir': 'output/tf_activation/tf_acts/', 
                'stats_tfs': f'output/tf_activation/stats_tfs_bulk_{gender}.csv',
                'stats_targets': f'output/tf_activation/stats_targets_bulk_{gender}.csv',
                'stats_all': f'output/tf_activation/stats_all_bulk_{gender}.csv', 
                'tmp_dir': 'output/tmp/',
            }
            os.makedirs(par['tf_acts_dir'], exist_ok=True)
            run_workflow_bulk(par)
    if True:
        # ----- sc TF activity
        par = {
            'type': 'sc',
            'cell_type_resolution': 'Major_CT',
            'gender_stratification': 'all',
            'tf_acts_dir': 'output/tf_activation/tf_acts/', 
            'stats_tfs': 'output/tf_activation/stats_tfs_bulk_sc.csv',
            'stats_targets': 'output/tf_activation/stats_targets_sc.csv',
            'stats_all': 'output/tf_activation/stats_all_sc.csv', 
            'tmp_dir': 'output/tmp/',
        }
        os.makedirs(par['tf_acts_dir'], exist_ok=True)
        run_workflow_bulk(par)
        
    