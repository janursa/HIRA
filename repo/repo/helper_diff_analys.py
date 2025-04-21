
import sys
import subprocess
import os
import anndata as ad
import scanpy as sc 
import os
import anndata as ad
import numpy as np 
import pandas as pd 
import seaborn as sns
from scipy import stats
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import argparse

from scipy.stats import spearmanr
import sys
import matplotlib.pyplot as plt
import scanpy as sc 
# import decoupler as dc 
from scipy.stats import pearsonr
import json
import itertools
import warnings
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from scipy import stats

import numpy as np
from scipy.stats import spearmanr, t
from statsmodels.stats.multitest import multipletests


sys.path.insert(0, '../')
from task_grn_inference.src.utils.util import basic_qc, read_gmt
# from task_grn_inference.src.process_data.perturbation.opsca.script import sum_by
sys.path.insert(0, './')
from src.helper import get_genesets, get_gene2pathway, efficient_melting, determine_centrality, surrogate_names


def find_consistency_between_batches(df, key_value='centrality', top_n_genes=10, batch_col='batch_group'):
    df_pivot = df.pivot(index='gene', columns=[batch_col], values=key_value).fillna(0)
    

    prod_df = df_pivot.prod(axis=1).to_frame(name='prod')
    
    # print(df_pivot.merge(rank_df, left_index=True, right_index=True).sort_values('rank', ascending=False)[:top_n_genes].index.tolist())
    return df_pivot.merge(prod_df, left_index=True, right_index=True).sort_values('prod', ascending=False)[:top_n_genes].index.tolist()

def filter_noisy_genes(centrality_all, threshold=1):
    """
    this is essential to interpret centrality. it remove genes that are zero centrality in both batches for either of ref or sample:
        - if a gene is zero centrality in both batches, then it's biological meaningful, otherwise it's drop outs
        - if not remved, the normalized centrality will be a very high number, twisting the results 
    """
    filtered_dfs = []
    groups = centrality_all.groupby(['age_group', 'cell_type'])
    for (age_group, cell_type), df in groups: 
        c_ref = df.pivot(index='gene', columns='batch_group', values='centrality_ref')
        c_sample = df.pivot(index='gene', columns='batch_group', values='centrality_sample')

        # - genes with less than n centrality in 
        ref_quality_genes = c_ref[(c_ref >= threshold).all(axis=1)].index.values #
        sample_quality_genes = c_sample[(c_sample >= threshold).all(axis=1)].index.values
        passed_genes = np.intersect1d(ref_quality_genes, sample_quality_genes)

        # Filter the original DataFrame for passed genes
        filtered_df = df[df['gene'].isin(passed_genes)]
        filtered_dfs.append(filtered_df)

    return pd.concat(filtered_dfs, ignore_index=True)

def sort_based_on_consistency(df) -> list[str]:
    # Get unique batch groups
    batch_groups = df['batch_group'].unique()
    assert len(batch_groups) > 1, "At least two batch groups are required."

    # Pivot each batch group into a separate DataFrame
    pivoted_dfs = {
        group: df[df['batch_group'] == group].pivot_table(index='gene', columns='age_group', values='centrality')
        for group in batch_groups
    }

    # Find common genes across all batch groups
    common_genes = list(set.intersection(*(set(pivoted.index) for pivoted in pivoted_dfs.values())))

    # Subset and fill NaN values with 0 for consistency calculation
    pivoted_dfs = {group: pivoted.loc[common_genes].fillna(0) for group, pivoted in pivoted_dfs.items()}

    # Calculate pairwise cosine similarity for each gene across all batch groups
    scores = {}
    for gene in common_genes:
        similarities = []
        for group1 in batch_groups:
            for group2 in batch_groups:
                if group1 != group2:
                    sim = cosine_similarity(
                        [pivoted_dfs[group1].loc[gene]],
                        [pivoted_dfs[group2].loc[gene]]
                    )[0, 0]
                    similarities.append(sim)
        # Take the mean of all pairwise similarities for the gene
        scores[gene] = sum(similarities) / len(similarities)

    # Convert scores dictionary to a Series
    scores_series = pd.Series(scores, name='score')

    # Sort genes by their average consistency score
    sorted_genes = scores_series.sort_values(ascending=False).index.tolist()

    return sorted_genes


def metrics_consistency(batch1, batch2):
    ref_age = '34-'
    sign_scores = []
    corr_scores = []
    for index, row in batch1.iterrows():
        row_ref = row
        row_val = batch2.loc[index]
        norm_diff_ref = (row_ref-row_ref[ref_age])
        norm_diff_ref = norm_diff_ref/norm_diff_ref.abs().max()
        norm_diff_val = (row_val-row_val[ref_age])
        norm_diff_val= norm_diff_val/norm_diff_val.abs().max()

        from scipy.stats import spearmanr
        correlation, _ = spearmanr(norm_diff_ref, norm_diff_val)
        # if np.isnan(correlation):
        #     print(index, batch1)
        #     aaa
        corr_scores.append(correlation)
        sign_score = (np.sign(norm_diff_ref)==np.sign(norm_diff_val)).sum()-1
        sign_score = sign_score/(row_ref.shape[0]-1)
        sign_scores.append(sign_score)
    return corr_scores, sign_scores
def get_metadata(df, tf_all, ref_dataset='pbmc_ageing', val_datasets=['data1'], var_name='gene', value_name='centrality'):
    """
    
    """
    from sklearn.metrics.pairwise import cosine_similarity
    from src.helper import surrogate_names

    df_ref = df[(df['dataset'] == ref_dataset)]
    
    if 'batch_group' in df_ref.columns:
        df_ref_AllBatch = df_ref[df_ref['batch_group'] == 'all_batches']
    else:
        df_ref_AllBatch = df_ref
    df_ref_AllBatch_table = df_ref_AllBatch.pivot(index=var_name, columns='age_group', values=value_name).fillna(0)
    meta_data = {var_name:df_ref_AllBatch_table.index.values}
    # - label tfs 
    tf_col = df_ref_AllBatch_table.index.isin(tf_all).astype(int)
    meta_data['TF'] = tf_col
    # - significane of change
    norm_std = df_ref_AllBatch_table.std(axis=1)/df_ref_AllBatch_table.mean(axis=1)
    meta_data['Significance of change'] = norm_std.values
    if ('batch_group' in df_ref.columns):
        if df_ref['batch_group'].nunique()>1:
            # - similarity between two batches of ref dataset 
            batch_1 = df_ref[df_ref['batch_group']=='batch_1'].pivot(index=var_name, columns='age_group', values=value_name).fillna(0)
            batch_2 = df_ref[df_ref['batch_group']=='batch_2'].pivot(index=var_name, columns='age_group', values=value_name).fillna(0)
            batch_2 = batch_2.reindex(batch_1.index).fillna(0)
            assert len(batch_2) == len(batch_1)
            # similarity_matrix_batches = cosine_similarity(batch_1, batch_2)
            # similarity_matrix_batches = np.asarray([similarity_matrix_batches[idx, idx] for idx in range(batch_1.shape[0])])
            # meta_data['Consistency (batches)'] = similarity_matrix_batches
            corr_scores, sign_scores = metrics_consistency(batch_1, batch_2)
            meta_data['Spearman (batches)'] = corr_scores
            meta_data['Sign (batches)'] = sign_scores


    # similarity between dataset 1 and 2 
    print('Similarity between datasets')
    for val_dataset in val_datasets:
        df_val = df[(df['dataset'] == val_dataset)]
        assert df_val.shape[0]!=0, f'{val_dataset} not found in the data'
        if 'batch_group' in df_ref.columns:
            df_val_AllBatch_table = df_val[df_val['batch_group']=='all_batches'].pivot(index=var_name, columns='age_group', values=value_name).fillna(0)
        else:
            df_val_AllBatch_table = df_val.pivot(index=var_name, columns='age_group', values=value_name).fillna(0)
        df_val_AllBatch_table = df_val_AllBatch_table.reindex(df_ref_AllBatch_table.index).fillna(0)
        assert len(df_val_AllBatch_table) == len(df_ref_AllBatch_table)
        corr_scores, sign_scores = metrics_consistency(df_ref_AllBatch_table, df_val_AllBatch_table)
        meta_data[f'Spearman ({surrogate_names.get(val_dataset, val_dataset)})'] = corr_scores
        meta_data[f'Sign ({surrogate_names.get(val_dataset, val_dataset)})'] = sign_scores

    
    # combine metadata
    meta_df = pd.DataFrame(meta_data).set_index(var_name)      
    return meta_df

def determine_centrality_all(net_all, use_weight=False):
    ii = 0
    for (cell_type, batch_group, age_group, dataset), group_df in net_all.groupby(['cell_type', 'batch_group', 'age_group', 'dataset']):
        centrality_df = determine_centrality(group_df, use_weight=use_weight).reset_index()
        # print(centrality_df)
        # aa
        centrality_df['cell_type'] = cell_type
        centrality_df['batch_group'] = batch_group
        centrality_df['age_group'] = age_group
        centrality_df['dataset'] = dataset
        if ii == 0:
            centrality_all_df = centrality_df
        else:
            centrality_all_df = pd.concat([centrality_all_df, centrality_df])
        ii=+1
    centrality_all_df = centrality_all_df.rename(columns={'index':'gene'})
    return centrality_all_df
def determine_centrality_consistency(net_all):
    net_all = net_all[net_all['batch_group']!='all_batches']

    # - read the inputs and calculate diff links
    net_all = net_all.reset_index()
    net_all['cell_type'] = net_all['cell_type'].astype(str)

    # - calculaye centrality 
    centrality_all_df = determine_centrality_all(net_all)
    # - centrality related scores
    def func_score(df, covariates, value='-log10_pvalue'):
        df_pivot = df.pivot(index='batch_group',columns=covariates, values=value)
        corr_matrix, p_values = spearmanr(df_pivot.loc['batch_1'].fillna(0).values.flatten(), df_pivot.loc['batch_2'].fillna(0).values.flatten(), nan_policy='raise')

        return corr_matrix
    centrality_scores_celltypes = centrality_all_df.groupby('cell_type').apply(lambda df: func_score(df, covariates=['gene', 'age_group'], value='centrality')).reset_index(name='score')
    centrality_scores_age_group = centrality_all_df.groupby('age_group').apply(lambda df: func_score(df, covariates=['gene', 'cell_type'], value='centrality')).reset_index(name='score')
    centrality_consistency_scores = pd.concat([centrality_scores_age_group, centrality_scores_celltypes])

    return centrality_consistency_scores

def enrich_pathway_interaction(df_subset, df_all, pathway_df):
    '''Calculates enriched pathway interacttions between df_subset and df_all. Both these dfs should have two columns of g1 and g2. All genes given should be present in pathway_df, which has gene as index and corrosponding pathway'''
    # Map genes to their pathways as sets for efficient lookup
    gene_to_pathways = pathway_df.groupby(pathway_df.index)['pathway'].apply(set)

    # Create pathway interaction pairs for the full dataset
    pathway_interactions_all = [
        (p1, p2)
        for g1, g2 in zip(df_all['g1'], df_all['g2'])
        for p1 in gene_to_pathways[g1]
        for p2 in gene_to_pathways[g2]]

    # Create and count pathway interaction pairs
    pathway_interactions_all_df = pd.DataFrame(
        pathway_interactions_all, columns=['pathway_1', 'pathway_2']
    )
    full_counts = pathway_interactions_all_df.value_counts().reset_index()
    full_counts.columns = ['pathway_1', 'pathway_2', 'count']


    # Create pathway interaction pairs for the full dataset
    pathway_interactions_subset = [
        (p1, p2)
        for g1, g2 in zip(df_subset['g1'], df_subset['g2'])
        for p1 in gene_to_pathways[g1]
        for p2 in gene_to_pathways[g2]]

    # Create and count pathway interaction pairs
    pathway_interactions_subset_df = pd.DataFrame(
        pathway_interactions_subset, columns=['pathway_1', 'pathway_2']
    )
    subset_counts = pathway_interactions_subset_df.value_counts().reset_index()
    subset_counts.columns = ['pathway_1', 'pathway_2', 'count']

    # Merge observed and expected counts
    merged_counts = subset_counts.merge(
        full_counts, on=['pathway_1', 'pathway_2'], how='outer', suffixes=('_observed', '_expected')
    ).fillna(0)

    
    # Check if p_value column is empty
    if merged_counts.empty:
        # If no p-values are available, return an empty DataFrame with the expected structure
        return merged_counts
    
    # Normalize by total pairs
    merged_counts['freq_observed'] = merged_counts['count_observed'] / len(pathway_interactions_subset)
    merged_counts['freq_expected'] = merged_counts['count_expected'] / len(pathway_interactions_all)

    merged_counts['sign'] = merged_counts['freq_observed'] - merged_counts['freq_expected']

    # fisher exact test 
    from scipy.stats import fisher_exact
    import statsmodels.stats.multitest as smm

    # Fisher's exact test for each pathway interaction
    p_values = []
    odds_ratios = []

    for _, row in merged_counts.iterrows():
        observed = row['count_observed']
        expected = row['count_expected']
        total_observed = len(pathway_interactions_subset)
        total_expected = len(pathway_interactions_all)

        contingency_table = [
            [observed, total_observed - observed],
            [expected, total_expected - expected],
        ]
        odds_ratio, p_value = fisher_exact(contingency_table)
        p_values.append(p_value)
        odds_ratios.append(odds_ratio)

    merged_counts['p_value'] = p_values
    merged_counts['odds_ratio'] = odds_ratios

    # Adjust p-values for multiple comparisons
    merged_counts['p_adj'] = smm.multipletests(merged_counts['p_value'], method='fdr_bh')[1]
    # -log10_pvalue
    merged_counts['-log10_pvalue'] = -np.log10(merged_counts['p_adj'])

    return merged_counts

def consistency_metrics(net_all_file):
    import networkx as nx

    # - read the inputs and calculate diff links
    net_all = pd.read_csv(net_all_file, index_col=0)
    net_all = net_all.reset_index()
    net_all['cell_type'] = net_all['cell_type'].astype(str)

    # - sig analysis
    net_all['sig'] = (net_all['adj_pvalue']<0.001) & (net_all['diff']>0.05)
    net_all[['g1', 'g2']] = net_all['link'].str.split('_', expand=True)

    net_all_sig = net_all[net_all['sig']]

    print('ratio of sig: ', net_all_sig.shape[0]/net_all.shape[0])

    pathway_df = get_gene2pathway()

    # - calculate pathway interaction enrichment scores 

    groups_all = net_all.groupby(['cell_type','age_group'])
    groups_sig = net_all_sig.groupby(['cell_type','age_group'])

    i_all = 0
    for age_group in sorted(net_all['age_group'].unique()):
        for cell_type in net_all['cell_type'].unique():
            try:
                df_subset_sig = groups_sig.get_group((cell_type, age_group))
                df_subset_all = groups_all.get_group((cell_type, age_group))
            except:
                continue

            for i_batch, batch_group in enumerate(net_all['batch_group'].unique()):
                df_subset_sig_b = df_subset_sig[df_subset_sig['batch_group'].eq(batch_group)]
                df_subset_all_b = df_subset_all[df_subset_all['batch_group'].eq(batch_group)]
                # - find pathway interaction enrich scores
                interaction_score_df = enrich_pathway_interaction(df_subset_sig_b, df_subset_all_b, pathway_df)
                interaction_score_df['cell_type'] = cell_type
                interaction_score_df['age_group'] = age_group
                interaction_score_df['batch_group'] = batch_group

                if i_all == 0:
                    interaction_score_all = interaction_score_df
                else:
                    interaction_score_all = pd.concat([interaction_score_all, interaction_score_df], axis=0).reset_index(drop=True)
                i_all+=1
    interaction_score_all['link'] = interaction_score_all['pathway_1'] + ' -- ' + interaction_score_all['pathway_2']

    # - filter to keep only those that are sig across both batch groups
    result_list = []

    for age_group in interaction_score_all['age_group'].unique():
        for cell_type in interaction_score_all['cell_type'].unique():
            # Filter data for the specific combination of age_group and cell_type
            subset = interaction_score_all[
                (interaction_score_all['age_group'] == age_group) &
                (interaction_score_all['cell_type'] == cell_type)
            ]
            
            # Pivot the data to prepare for mutual significance check
            df_pivot = subset.pivot(
                index='batch_group', 
                columns='link', 
                values='p_adj'
            )
            
            # Identify mutual significant links across batch_group
            mask_mutual_sig = (df_pivot < 0.05).all(axis=0)
            significant_links = mask_mutual_sig[mask_mutual_sig].index  # Links that are mutual
            
            # Filter the original subset for the significant links
            filtered_subset = subset[subset['link'].isin(significant_links)]
            
            # Store the filtered subset
            result_list.append(filtered_subset)

    filtered_interaction_score_all = pd.concat(result_list, ignore_index=True) #only keep the sig ones

    # - calculaye centrality 
    centrality_df = determine_centrality(net_all_sig)

    # - pathway related scores
    def func_score(df, covariates, value='-log10_pvalue'):
        df_pivot = df.pivot(index='batch_group',columns=covariates, values=value)
        corr_matrix, p_values = spearmanr(df_pivot.loc['batch_1'].fillna(0).values.flatten(), df_pivot.loc['batch_2'].fillna(0).values.flatten(), nan_policy='raise')

        return corr_matrix

    pathways_scores_overall = func_score(filtered_interaction_score_all, covariates=['link', 'age_group', 'cell_type'])
    pathways_scores_celltypes = filtered_interaction_score_all.groupby('cell_type').apply(lambda df: func_score(df, covariates=['link', 'age_group'])).reset_index(name='score')
    pathways_scores_age_group = filtered_interaction_score_all.groupby('age_group').apply(lambda df: func_score(df, covariates=['link', 'cell_type'])).reset_index(name='score')
    pathway_scores_all = pd.concat([pathways_scores_celltypes, pathways_scores_age_group])
    pathway_scores_all['type'] = 'pathway'

    # - centrality related scores
    centrality_scores_overall = func_score(centrality_df, covariates=['gene', 'age_group', 'cell_type'], value='centrality')
    centrality_scores_celltypes = centrality_df.groupby('cell_type').apply(lambda df: func_score(df, covariates=['gene', 'age_group'], value='centrality')).reset_index(name='score')
    centrality_scores_age_group = centrality_df.groupby('age_group').apply(lambda df: func_score(df, covariates=['gene', 'cell_type'], value='centrality')).reset_index(name='score')
    centrality_scores_all = pd.concat([centrality_scores_age_group, centrality_scores_celltypes])
    centrality_scores_all['type'] = 'centrality'

    # - jaccard sim
    def func_score(df, covariates, value='-diff'):
        df_pivot = df.pivot(index='batch_group', columns=covariates, values=value).fillna(0)
        jaccard_sim = (df_pivot!=0).all(axis=0).sum()/df_pivot.shape[1]
        return jaccard_sim

    jacc_sim_overall = func_score(net_all_sig, covariates=['link', 'age_group', 'cell_type'], value='diff')
    jacc_sim_celltypes = net_all_sig.groupby('cell_type').apply(lambda df: func_score(df, covariates=['link', 'age_group'], value='diff')).reset_index(name='score')
    jacc_sim_age_group = net_all_sig.groupby('age_group').apply(lambda df: func_score(df, covariates=['link', 'cell_type'], value='diff')).reset_index(name='score')
    jacc_sim_all = pd.concat([jacc_sim_celltypes, jacc_sim_age_group])
    jacc_sim_all['type'] = 'link_jac_sim'

    # - combine the scores
    scores_all = pd.concat([pathway_scores_all, centrality_scores_all, jacc_sim_all])

    rr = {'net_all_sig':net_all_sig, 'interaction_score_all':interaction_score_all, 'filtered_interaction_score_all':filtered_interaction_score_all, 'centrality_df':centrality_df, 'scores_all':scores_all}
    return rr

def run_DEA():
    par = {
        'dataset_file': 'input/dataset_1_2.h5ad',
        'save_file': 'output/diff_genes/de_genes_dataset1_2.csv',
        'ref_age_group': '34-'
    }

    adata = ad.read_h5ad(par['dataset_file'])
    if 'batch_group' not in adata.obs:
        adata.obs['batch_group'] = 'all'
    
    # - fix the granualariy of the cell typs based on geneRNIB
    adata.obs['cell_type_major'] = adata.obs['cell_type'].map(map_cell_type_genernib)
    adata.obs['cell_type_major'].unique()

    # mask_genes = adata.var_names.isin(target_genes) 
    # - pseudobulk
    adata.obs['sum_by'] = adata.obs['age_group'].astype(str) + '_' + adata.obs['cell_type'].astype(str) + '_' + adata.obs['batch_group'].astype(str) + '_' + adata.obs['donor_id'].astype(str)
    adata.obs['sum_by'] = adata.obs['sum_by'].astype('category')
    adata_bulk = sum_by(adata, 'sum_by')

    # - normalize
    X_norm = sc.pp.normalize_total(adata_bulk, inplace=False)['X']
    adata_bulk.layers['X_norm'] = sc.pp.log1p(X_norm, copy=True)

    adata_bulk.layers['counts'] = adata_bulk.X
    adata_bulk.X = adata_bulk.layers['X_norm']

    # - run for each group
    results = []

    for batch_group in adata_bulk.obs['batch_group'].unique():
        batch_mask = adata_bulk.obs['batch_group'] == batch_group
        
        
        for cell_type in adata_bulk.obs['cell_type_major'].unique():
            cell_type_mask = adata_bulk.obs['cell_type_major'] == cell_type
            adata_bulk_sub = adata_bulk[batch_mask & cell_type_mask, :]
            if adata_bulk_sub.shape[0] == 0:
                continue
            

            for age_group in adata_bulk.obs['age_group'].unique():
                print(batch_group, cell_type, age_group)
                # Pseudobulking by mean for each ['donor_id', 'age_group', 'cell_type']
                

                # Prepare data for DE analysis
                adata_ref = adata_bulk_sub[adata_bulk_sub.obs['age_group'] == par['ref_age_group'], :]
                adata_test = adata_bulk_sub[adata_bulk_sub.obs['age_group'] == age_group, :]
                
                # Apply a rank-based test (e.g., Mann-Whitney U test)
                pvals = []
                for gene in adata_bulk.var_names:
                    if gene in adata_ref.var_names and gene in adata_test.var_names:
                        gene_mask = adata_bulk.var_names == gene
                        x_test = adata_test.X[:, gene_mask].todense().A.flatten()
                        x_ref = adata_ref.X[:, gene_mask].todense().A.flatten()
                        # print(x_ref)
                        stat, p_value = stats.mannwhitneyu(x_ref, x_test, alternative='two-sided')
                        pvals.append((gene, p_value, x_test.mean() , x_ref.mean()))
                
                # Collect results
                df_results = pd.DataFrame(pvals, columns=['gene', 'p_value', 'mean_test', 'mean_ref'])
                df_results['age_group'] = age_group
                df_results['batch_group'] = batch_group
                df_results['cell_type'] = cell_type
                results.append(df_results)

    # Concatenate and save
    final_results = pd.concat(results, ignore_index=True)
    final_results.to_csv(par['save_file'], index=False)

