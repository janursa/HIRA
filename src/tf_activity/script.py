import anndata as ad
import pandas as pd
import sys
import numpy as np

import argparse
meta = {
    'helper_dir': '/home/jnourisa/projs/ongoing/ciim/src/tf_activity',

}
sys.path.append(meta['helper_dir'])
from helper import enrich_tfs, convert_long_table_2_adata, find_central_tfs, \
                   find_robust_predictors, determine_stats, run_pseudotime_analysis, net_lambda, adata_cell_type_lambda, tf_all



def main(par):
    include_pseudo = False
    # - global vars
    cell_type = par['cell_type']
    datasets = par['datasets']
    # --------- load data
    nets_dict = {dataset: net_lambda(dataset, cell_type) for dataset in datasets}
    adata_dict = {dataset: adata_cell_type_lambda(dataset, cell_type) for dataset in datasets}
    
    # ----------- calculate tf activity for all datasets
    #  stored as dict where each item is in adata format (obs: age, donor_id, cell_count) and X: tf activity
    tf_acts_dict = {}
    for dataset in datasets:
        tf_acts = enrich_tfs(adata_dict[dataset], nets_dict[dataset])
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
    def summary_func(df):
        pvals = df["p_value_adj"]
        effect = df["slope"]
        if effect.prod() < 0: # if the effect is in opposite direction, pvalue is none
            return None
        if len(pvals)<2:
            return 1
        else:
            return max(pvals)
            # combine_pvalues(pvals, method="fisher")[1]
    meta_p_values = (
        stats_df.groupby("tf")
        .apply(summary_func)  
        .reset_index(name='meta_p_value')
        
    )
    stats_df = stats_df.merge(meta_p_values, on='tf')
    stats_df = stats_df[stats_df['meta_p_value'] < 0.05].reset_index(drop=True)

    # --------------- add the type of discovery (age, centrality, pseudo) to the stats
    stats_df["type"] = [[] for _ in range(len(stats_df))]
    for idx, row in stats_df.iterrows():
        tf = row["tf"]
        dataset = row["dataset"]
        tf_types = []

        if tf in top_central_tfs.get(dataset, []):
            tf_types.append("central")
        if tf in top_predictors_age.get(dataset, []):
            tf_types.append("age")
        if include_pseudo:
            if tf in top_predictors_pseudo.get(dataset, []):
                tf_types.append("pseudo")

        stats_df.at[idx, "type"] = tf_types
    # --------------- add the stats obtained from expression data (rather than TF activity)
    sig_tfs_all = stats_df['tf'].unique()
    stats_store = []
    for dataset in datasets:
        adata = adata_dict[dataset]
        stats = determine_stats(adata, sig_tfs_all)
        stats['dataset'] = dataset
        stats_store.append(stats)
    expression_stats_df = pd.concat(stats_store)
    
    if expression_stats_df.shape[0] > 0:
        meta_p_values = (
            expression_stats_df.groupby("tf")
            .apply(summary_func)  
            .reset_index(name='meta_p_value')
            
        )
        expression_stats_df = expression_stats_df.merge(meta_p_values, on='tf')
        stats_df = stats_df.merge(expression_stats_df[['tf', 'slope', 'p_value_adj', 'meta_p_value', 'dataset']], on=['tf', 'dataset'], suffixes=('', '_expression'), how='left')

    return stats_df

