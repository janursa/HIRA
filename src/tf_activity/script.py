import anndata as ad
import pandas as pd
import sys
import numpy as np
import scanpy as sc 
from scipy import stats
from scipy.stats import pearsonr

import argparse
meta = {
    'helper_dir': '/home/jnourisa/projs/ongoing/ciim/src/tf_activity',

}
sys.path.append(meta['helper_dir'])
from helper import calculate_tf_activity, convert_long_table_2_adata, find_central_tfs, \
                   find_robust_predictors, determine_stats, run_pseudotime_analysis, net_lambda, \
                    adata_cell_type_lambda, tf_all, summary_func, adata_sc_lambda



def determine_stats_tf_activation_bulk(par):
    include_pseudo = False
    # - global vars
    cell_type = par['cell_type']
    datasets = par['datasets']
    # --------- load data
    nets_dict = {dataset: net_lambda(dataset, cell_type) for dataset in datasets}
    if par['use_pseudobulk']:
        adata_dict = {dataset: adata_cell_type_lambda(dataset, cell_type) for dataset in datasets}
    else:
        adata_dict = {dataset: adata_sc_lambda(dataset) for dataset in datasets}
        adata_dict = {dataset: adata[adata.obs['cell_type']==cell_type] for dataset, adata in adata_dict.items()}
        
    
    # ----------- calculate tf activity for all datasets
    #  stored as dict where each item is in adata format (obs: age, donor_id, cell_count) and X: tf activity
    tf_acts_dict = {}
    for dataset in datasets:
        tf_acts = calculate_tf_activity(adata_dict[dataset], nets_dict[dataset])
        tf_acts_dict[dataset] = tf_acts

    # ------------ identify central TFs 
    print("Running central TFs analysis")
    top_central_tfs = {}
    for dataset in datasets:
        top_central_tfs[dataset] = find_central_tfs(nets_dict[dataset], par)

    # -------------- identify robust predictors of age 
    print("Running age prediction analysis")
    covariates = ["age", "donor_id"]
    target = 'age'
    top_predictors_age = {}
    for dataset in datasets:
        adata = tf_acts_dict[dataset]
        top_predictors_age[dataset] = find_robust_predictors(
            adata, covariates, target, top_q=par['predictor_age_q']
        )
    # -------------- identify robust predictors of pseudo age
    if include_pseudo:
        print("Running pseudotime prediction analysis")
        target = 'dpt_pseudotime'
        top_predictors_pseudo = {}

        for dataset in datasets:
            # - run pseudotime analysis 
            adata = tf_acts_dict[dataset]
            # adata.obs.index = adata.obs.index.astype(int)
            run_pseudotime_analysis(adata, seed=32)
            # - remove those samples with invalid pseudotime (seems like it's only for MONO)
            invalid_pseudotime_samples = adata.obs['dpt_pseudotime']==np.inf
            print(f"Removing {invalid_pseudotime_samples.sum()} samples with invalid pseudotime")
            adata = adata[~invalid_pseudotime_samples, :]
            # - find robust predictors
            top_predictors_pseudo[dataset] = find_robust_predictors(
                adata, covariates, target, top_q=par['predictor_age_q']
            )
    
    # -------------- combined all identified TFs and calculate stats (p-value, effect size) with age trajectory
    stats_store = []
    for dataset in datasets:
        adata = tf_acts_dict[dataset]
        if include_pseudo:
            top_tfs = list(set(top_central_tfs[dataset]) | set(top_predictors_age[dataset]) | set(top_predictors_pseudo[dataset]))
        else:
            top_tfs = list(set(top_central_tfs[dataset]) | set(top_predictors_age[dataset]))
        stats = determine_stats(adata, top_tfs)
        stats['dataset'] = dataset
        
        stats_store.append(stats)
    stats_df = pd.concat(stats_store)

    # -------------- meta analysis to find mutual top TFs
    
    # stats_df = summary_func(stats_df)
    # stats_df = stats_df_raw[stats_df_raw['meta_p_value'] < 0.05].reset_index(drop=True)

    
    
    return stats_df


def determine_stats_std_tf_activation(par):
    # - global vars
    cell_type = par['cell_type']

    datasets = par['datasets']
    print(f"Running for {cell_type}")
    # --------- load data
    nets_dict = {dataset: net_lambda(dataset, cell_type) for dataset in datasets}
    
    adata_dict = {dataset: adata_sc_lambda(dataset) for dataset in datasets}
    adata_dict = {dataset: adata[adata.obs['cell_type']==cell_type] for dataset, adata in adata_dict.items()}
        
    # ----------- calculate tf activity for all datasets
    #  stored as dict where each item is in adata format (obs: age, donor_id, cell_count) and X: tf activity
    tf_acts_dict = {}
    for dataset in datasets:
        print(f"Calculating TF activity for {dataset}")
        tf_acts = calculate_tf_activity(adata_dict[dataset], nets_dict[dataset])
        tf_acts.write(f"{par['temp_dir']}/{dataset}_{cell_type}_tf_acts.h5ad")
        tf_acts_dict[dataset] = tf_acts

    # ------------ calculate std of tf activity 
    print("Running std")
    tf_act_std_dict = {}
    for dataset, adata in tf_acts_dict.items():
        df = pd.DataFrame(adata.X, index=adata.obs.index, columns=adata.var_names)
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
        tf_act_std_dict[dataset] = std_adata

    # -------------- identify robust predictors of age 
    print("Running age prediction analysis")
    covariates = ["age", "donor_id"]
    target = 'age'
    top_predictors_age = {}
    for dataset in datasets:
        adata = tf_act_std_dict[dataset]
        top_predictors_age[dataset] = find_robust_predictors(
            adata, covariates, target, top_q=par['predictor_age_q']
        )
    
    
    # -------------- combined all identified TFs and calculate stats (p-value, effect size) with age trajectory
    stats_store = []
    for dataset in datasets:
        adata = tf_act_std_dict[dataset]
        top_tfs = top_predictors_age[dataset]
        stats = determine_stats(adata, top_tfs)
        stats['dataset'] = dataset
        
        stats_store.append(stats)
    stats_df = pd.concat(stats_store)

    return stats_df

def determine_stats_targets_bulk(targets, cell_type, datasets):
    """
        for a given cell type, compute the stats of the targets (p values for expression based change in ageing)
    """
    from src.tf_activity.helper import adata_cell_type_lambda, determine_stats
    stats_store = []
    for dataset in datasets:
        adata = adata_cell_type_lambda(dataset, cell_type)
        stats = determine_stats(adata, targets)
        stats['dataset'] = dataset
        stats_store.append(stats)
    stats = pd.concat(stats_store) # index 
    stats.rename(columns={'tf': 'target'}, inplace=True) #stats has target and their associated adj_p_value
    stats['cell_type'] = cell_type
    return stats
def determine_stats_std_targets_sc(targets, cell_type, datasets):
    """
        for a given cell type, compute the stats of the targets (p values for expression based change in ageing)
    """
    from src.tf_activity.helper import adata_cell_type_lambda, determine_stats
    stats_store = []

    for dataset in datasets:
        adata = adata_sc_lambda(dataset) 
        adata = adata[adata.obs['cell_type']==cell_type]
        stats = determine_stats(adata, targets)
        stats['dataset'] = dataset
        stats_store.append(stats)
    stats = pd.concat(stats_store) # index 
    stats.rename(columns={'tf': 'target'}, inplace=True) #stats has target and their associated adj_p_value
    stats['cell_type'] = cell_type
    return stats