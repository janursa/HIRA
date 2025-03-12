import anndata as ad
import pandas as pd
import sys
import numpy as np

import argparse
meta = {
    'helper_dir': '/home/jnourisa/projs/ongoing/ciim/src/tf_activity',

}
sys.path.append(meta['helper_dir'])
from helper import calculate_tf_activity, convert_long_table_2_adata, find_central_tfs, \
                   find_robust_predictors, determine_stats, run_pseudotime_analysis, net_lambda, \
                    adata_cell_type_lambda, tf_all, summary_func, adata_sc_lambda



def main(par):
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

