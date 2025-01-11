
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
from task_grn_inference.src.process_data.perturbation.opsca.script import sum_by
sys.path.insert(0, './')
from src.helper import get_genesets, get_gene2pathway, efficient_melting, determine_centrality, surrogate_names, colors_blind


def find_consistency_between_batches(df, key_value='centrality', top_n_genes=10):
    df_pivot = df.pivot(index='gene', columns=['batch_group'], values=key_value)

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
def determine_diff_centrality_all(centrality_df):
    diff_centrality_all = []

    batch_groups = centrality_df.batch_group.unique()
    cell_types = centrality_df.cell_type.unique()
    age_groups = centrality_df.age_group.unique()

    for batch_group in batch_groups:
        for cell_type in cell_types:
            mask = (centrality_df.cell_type==cell_type)&(centrality_df.batch_group==batch_group)
            centrality_sub = centrality_df[mask]
            df_ref = centrality_sub[centrality_sub.age_group=='34-'][['gene','centrality']].set_index('gene')
            for age_group in age_groups:
                if age_group == '34-':
                    continue
                df_sample = centrality_sub[centrality_sub.age_group==age_group][['gene','centrality']].set_index('gene')
                df_merged = df_sample.merge(df_ref, left_index=True, right_index=True, suffixes=['_sample','_ref'], how='outer').fillna(0)

                # - main
                pseudocount = 1e-6
                c_ref = df_merged['centrality_ref'] + pseudocount
                c_sample = df_merged['centrality_sample'] + pseudocount
                df_merged['diff'] = c_sample-c_ref
                df_merged['fold_change'] = np.where(
                    c_sample > c_ref,
                    c_sample / c_ref,
                    c_ref / c_sample
                )
                df_merged['signed_fold_change'] = np.sign(df_merged['diff'])*df_merged['fold_change']
                df_merged['log2_fold_change'] = np.log2(df_merged['fold_change'])
                df_merged['signed_log2_fold_change'] = np.sign(df_merged['diff'])*df_merged['log2_fold_change']
                

                df_merged['batch_group'] = batch_group
                df_merged['cell_type'] = cell_type
                df_merged['age_group'] = age_group

                diff_centrality_all.append(df_merged)

    diff_centrality_all = pd.concat(diff_centrality_all)

    
    return diff_centrality_all

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

def plot_centrality_heatmap_metadata(df, metadata,length=8, width=3, cluster_offset=0.1):
    import scipy.cluster.hierarchy as sch
    from matplotlib.gridspec import GridSpec
    # - main data (used for clustering)

    data_heatmap = df.copy()
    data_heatmap = data_heatmap.pivot(index='gene', columns='age_group', values='centrality').fillna(0)
    # - Perform hierarchical clustering on the first batch data
    linkage = sch.linkage(data_heatmap, method='ward')
    dendrogram = sch.dendrogram(linkage, no_plot=True)
    cluster_order = [data_heatmap.index[i] for i in dendrogram['leaves'][::-1]]
   
    # Plot 
    fig = plt.figure(figsize=(width, length))  # Adjust overall figure size
    gs = GridSpec(1, 4, figure=fig)

    # - Dendrogram subplot
    axes = []
    ax_dendro = fig.add_subplot(gs[0, 0])
    axes.append(ax_dendro)
    sch.dendrogram(linkage, labels=data_heatmap.index, orientation='left', ax=ax_dendro)
    ax_dendro.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax_dendro.tick_params(left=False, bottom=False, right=False, top=False) 
    ax_dendro.set_xticks([])

    # Heatmap subplot
    ax = fig.add_subplot(gs[0, 1])
    data_heatmap = data_heatmap.reindex(cluster_order) 
    normalized_data = data_heatmap.div(data_heatmap.max(axis=1), axis=0)

    sns.heatmap(
        normalized_data,
        annot=data_heatmap,  
        fmt=".0f", 
        cmap="viridis", 
        ax=ax,
        cbar=None,
        annot_kws={"size": 8}
    )
    # ax.set_title(batch)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_yticklabels([])
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xticks(np.array(range(len(normalized_data.columns)))+.5)  # Set the correct number of ticks
    ax.set_xticklabels(normalized_data.columns, rotation=45, ha='right')  # Set the labels explicitly
    ax.set_xlabel('Age group', loc='left')
    ax_main=ax

    # Metadata subplot (part 1) -> consistency and significance
    meta_df = metadata.reindex(cluster_order) 
    meta_df_1 = meta_df[[col for col in meta_df.columns if col !='TF']]
    normalized_data = meta_df_1.div(meta_df_1.max(axis=0), axis=1)
    ax = fig.add_subplot(gs[0, 2])
    sns.heatmap(
        normalized_data,
        annot=meta_df_1,  
        fmt=".02f", 
        cmap=None, 
        ax=ax,
        cbar=None,
        annot_kws={"size": 8}
    )
    ax.set_yticks([])
    ax.set_ylabel('')
    ax.set_xticks(np.array(range(len(normalized_data.columns)))+.5)  
    ax.set_xticklabels(normalized_data.columns, rotation=45, ha='right')  
    ax_meta_1 = ax

    # Metadata subplot (part 2) -> TF
    meta_df_2 = meta_df[[col for col in meta_df.columns if col =='TF']]
    meta_df_2_show = meta_df_2.copy()
    meta_df_2_show['TF'] = meta_df_2_show['TF'].map({1:'True', 0:''})

    # meta_df_2_show = 
    ax = fig.add_subplot(gs[0, 3])
    from matplotlib.colors import ListedColormap

    sns.heatmap(
        meta_df_2,
        annot = meta_df_2_show,
        ax=ax,
        fmt="s",
        cmap=ListedColormap(["white", colors_blind[1]]),
        annot_kws={"size": 8},
        cbar=False
    )
    ax.set_yticks([])
    ax.set_ylabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax_meta_2 = ax
    
    ax_dendro.set_position([0.01, 0.1, 0.22, 0.75])  
    ax_main.set_position([0.6+cluster_offset, 0.1, 0.6, 0.75]) 
    ax_meta_1.set_position([0.6+0.6+.05+cluster_offset, 0.1, 0.6, 0.75])  
    ax_meta_2.set_position([0.6+0.6+0.6+.08+cluster_offset, 0.1, 0.1, 0.75])  

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
        corr_scores.append(correlation)
        sign_score = (np.sign(norm_diff_ref)==np.sign(norm_diff_val)).sum()-1
        sign_score = sign_score/4
        sign_scores.append(sign_score)
    return corr_scores, sign_scores
def get_metadata(df, tf_all, col_comparision='batch_group', ref_dataset='pbmc_ageing', val_dataset='data1'):
    """
    
    """
    from sklearn.metrics.pairwise import cosine_similarity
    df_ref = df[(df['dataset'] == ref_dataset)]
    df_val = df[(df['dataset'] == val_dataset)]
    
    df_ref_AllBatch = df_ref[df_ref['batch_group'] == 'all_batches']
    df_ref_AllBatch_table = df_ref_AllBatch.pivot(index='gene', columns='age_group', values='centrality').fillna(0)
    meta_data = {'gene':df_ref_AllBatch_table.index.values}
    # - label tfs 
    tf_col = df_ref_AllBatch_table.index.isin(tf_all).astype(int)
    meta_data['TF'] = tf_col
    # - significane of change
    norm_std = df_ref_AllBatch_table.std(axis=1)/df_ref_AllBatch_table.mean(axis=1)
    meta_data['Significance of change'] = norm_std.values
    # - similarity between two batches of ref dataset 
    batch_1 = df_ref[df_ref['batch_group']=='batch_1'].pivot(index='gene', columns='age_group', values='centrality').fillna(0)
    batch_2 = df_ref[df_ref['batch_group']=='batch_2'].pivot(index='gene', columns='age_group', values='centrality').fillna(0)
    batch_2 = batch_2.reindex(batch_1.index).fillna(0)
    assert len(batch_2) == len(batch_1)
    # similarity_matrix_batches = cosine_similarity(batch_1, batch_2)
    # similarity_matrix_batches = np.asarray([similarity_matrix_batches[idx, idx] for idx in range(batch_1.shape[0])])
    # meta_data['Consistency (batches)'] = similarity_matrix_batches
    corr_scores, sign_scores = metrics_consistency(batch_1, batch_2)
    meta_data['Spearman (batches)'] = corr_scores
    meta_data['Sign (batches)'] = sign_scores


    # similarity between dataset 1 and 2 
    df_val_AllBatch_table = df_val[df_val['batch_group']=='all_batches'].pivot(index='gene', columns='age_group', values='centrality').fillna(0)
    df_val_AllBatch_table = df_val_AllBatch_table.reindex(df_ref_AllBatch_table.index).fillna(0)
    assert len(df_val_AllBatch_table) == len(df_ref_AllBatch_table)
    corr_scores, sign_scores = metrics_consistency(df_val_AllBatch_table, df_ref_AllBatch_table)
    meta_data['Spearman (datasets)'] = corr_scores
    meta_data['Sign (datasets)'] = sign_scores

    
    # combine metadata
    meta_df = pd.DataFrame(meta_data).set_index('gene')      
    return meta_df

def plot_joint_centrality_heatmap(df, title='', length=8, width=3, cluster_offset=0.1, ref_batch='all_batches'):

    import scipy.cluster.hierarchy as sch
    from matplotlib.gridspec import GridSpec
    batch_groups = sorted(df['batch_group'].unique())

    # - Create a pivot table for the first batch (used for clustering)
    assert ref_batch in df['batch_group'].unique()
    first_batch_data = df[df['batch_group'] == ref_batch]
    df_pivot = first_batch_data.pivot(index='gene', columns='age_group', values='centrality').fillna(0)

    # - Perform hierarchical clustering on the first batch data
    linkage = sch.linkage(df_pivot, method='ward')
    dendrogram = sch.dendrogram(linkage, no_plot=True)
    cluster_order = [df_pivot.index[i] for i in dendrogram['leaves'][::-1]]


    # Plot 
    fig = plt.figure(figsize=(width * len(batch_groups) + 3, length))  # Adjust overall figure size
    gs = GridSpec(1, len(batch_groups) + 1, figure=fig)

    # - Dendrogram subplot
    axes = []
    ax_dendro = fig.add_subplot(gs[0, 0])
    axes.append(ax_dendro)
    sch.dendrogram(linkage, labels=df_pivot.index, orientation='left', ax=ax_dendro)
    ax_dendro.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax_dendro.tick_params(left=False, bottom=False, right=False, top=False) 
    ax_dendro.set_xticks([])
    

    # Heatmap subplots
    for i, batch in enumerate(batch_groups):
        ax = fig.add_subplot(gs[0, i + 1])
        axes.append(ax)

        batch_data = df[df['batch_group'] == batch]
        df_pivot = batch_data.pivot(index='gene', columns='age_group', values='centrality').fillna(0)
        df_pivot = df_pivot.reindex(cluster_order)  # Reorder genes based on clustering
        

        # Normalize data
        normalized_data = df_pivot.div(df_pivot.max(axis=1), axis=0)

        # Plot heatmap
        sns.heatmap(
            normalized_data,
            annot=df_pivot,  
            fmt=".0f", 
            cmap="viridis", 
            ax=ax,
            cbar=None,
            annot_kws={"size": 8}
        )
        ax.set_title(batch)
        ax.set_ylabel('')
        ax.set_xlabel('')
        # if i!=0:
        ax.set_yticklabels([])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    axes[0].set_position([0.01, 0.1, 0.17, 0.75])  
    axes[1].set_position([0.29+cluster_offset, 0.1, 0.17, 0.75])  
    axes[2].set_position([0.48+cluster_offset, 0.1, 0.17, 0.75])
    axes[3].set_position([0.67+cluster_offset, 0.1, 0.17, 0.75]) 
    
    axes[1].set_xlabel('Age group')
    plt.suptitle(title, fontsize=14)
    # plt.tight_layout()
    # plt.tight_layout()
    # plt.show()
    
def plot_joint_centrality_wrapper(centrality_df, key_value, top_n_genes, title, horizontal=True):
    # - identify top genes consistent between two batches 
    top_genes = centrality_df.groupby(['age_group']).apply(lambda df: find_consistency_between_batches(df, key_value, top_n_genes))
    top_genes = list(set([gene for sublist in top_genes for gene in sublist]))
    color_palette = sns.color_palette('tab20', len(top_genes))  # 'husl' is an example; choose any palette
    color_map = {gene:color for gene, color in zip(top_genes, color_palette)}

    plot_joint_centrality(centrality_df, title=title, color_map=color_map, key_value=key_value, top_n_genes=top_n_genes, horizontal=horizontal)
    return top_genes
def plot_joint_centrality(centrality_df, color_map, title='', key_value='centrality', top_n_genes=10, horizontal=True):   
    
    def add_show_name(df):
        df['show_name'] = 'Others'
        genes = find_consistency_between_batches(df, key_value, top_n_genes)
        mask = df['gene'].isin(genes)
        df.loc[mask, 'show_name'] = df.loc[mask, 'gene']
        return df
    age_groups = centrality_df['age_group'].unique()
    if horizontal:
        fig, axes = plt.subplots(1, len(age_groups), figsize=(4*len(age_groups), 3.5), sharey=False, dpi=100)
    else:
        fig, axes = plt.subplots(len(age_groups), 1, figsize=(4, 3*len(age_groups)), sharey=False, dpi=100)
    
    for ii, age_group in enumerate(age_groups):
        c_age = centrality_df.groupby('age_group').get_group((age_group))
        c_age = add_show_name(c_age)
        c_age_pivot = c_age.pivot(index='gene', columns=['batch_group'], values=key_value).reset_index()
        c_age_pivot = c_age_pivot.merge(c_age[['gene','show_name','is_tf']], on='gene', how='left').reset_index()
        c_age_pivot = c_age_pivot.dropna(axis=0)
        
        ax = axes[ii]
        sns.scatterplot(c_age_pivot, x='batch_1', y='batch_2', 
                hue='show_name', 
                ax=ax, palette={'Others':'grey', **color_map}, 
                style='is_tf'
                )
        ax.set_title(f'Age: {age_group}')
        ax.legend(loc=(1.05, 0), fontsize=8)
        # ax.get_legend().remove()
    if horizontal:
        plt.suptitle(title, y=1.1)
    else:
        plt.suptitle(title)
    plt.tight_layout()
    
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
def evaluate_batch_effect_on_hub_genes():
    adata = ad.read_h5ad('input/dataset_1_2.h5ad')
    # - subset to only one age group
    adata = adata[adata.obs['age_group']=='34-']
    # - subset to only one cell type
    adata.obs['cell_type_major'] = adata.obs['cell_type'].map(map_cell_type_genernib)
    adata = adata[adata.obs['cell_type_major'] =='T cells']
    # - determine the net for top 5 batches
    top_batches = adata.obs.groupby('donor_id').size().sort_values()[::-1][:5].index
    for ii, batch in enumerate(top_batches):
        print(batch)
        adata_sample = adata[adata.obs['donor_id'] == batch]

        adata_sample = basic_qc(adata_sample, min_cells_per_gene=500, min_genes_per_cell=10)

        if (adata_sample.shape[0]==0):
            continue
            
        if adata_sample.shape[1]==0:
            continue

        # - normalize 
        X_norm = sc.pp.normalize_total(adata_sample, inplace=False)['X']
        adata_sample.layers['X_norm'] = sc.pp.log1p(X_norm, copy=True)

        # - actual subset 
        expression_sample = adata_sample.layers['X_norm']
        gene_names = adata_sample.var_names

        # - infer grn
        net = infer_grn(expression_sample, gene_names)

        # net = net[net['source'].isin(tf_all)]

        net['weight'] = pd.to_numeric(net['weight'], errors='coerce')

        # net_short = net.loc[net['weight'].abs().nlargest(par['n_max_links']).index]
        net = net[net['weight'].abs()>.05]

        net['batch'] = batch

        if ii == 0:
            net_all = net 
        else:
            net_all = pd.concat([net_all, net])

    # - for the reference net, determine the hub genes
    net_ref = pd.read_csv(os.path.abspath(f"{par['read_dir']}/net_T cells_34-_all_batches.csv"), index_col=0)
    centrality_ref = determine_centrality_weight(net_ref)
    hub_genes_ref = centrality_ref.sort_values('centrality', ascending=False)[:100].index

    # - for each batch, calculate the recall of the hub genes
    recall_list = []
    for batch in net_all['batch'].unique():
        net = net_all[net_all['batch'] == batch]
        centrality_ref = determine_centrality_weight(net)
        hub_genes = centrality_ref.sort_values('centrality', ascending=False)[:100].index
        recall_list.append(np.intersect1d(hub_genes, hub_genes_ref).shape)
        print(recall_list)
    return recall_list
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
def sparse_corrcoef(A, B=None):

    if B is not None:
        A = sparse.vstack((A, B), format='csr')

    A = A.astype(np.float64)
    n = A.shape[1]

    # Compute the covariance matrix
    rowsum = A.sum(1)
    centering = rowsum.dot(rowsum.T.conjugate()) / n
    C = (A.dot(A.T.conjugate()) - centering) / (n - 1)

    # The correlation coefficients are given by
    # C_{i,j} / sqrt(C_{i} * C_{j})
    d = np.diag(C)
    coeffs = C / np.sqrt(np.outer(d, d))

    return coeffs
def infer_grn(X, gene_names):
    from scipy.stats import spearmanr
    std_devs = sparse_std(X)
    mask_zero_std = std_devs == 0
    gene_names = gene_names[~mask_zero_std]
    X_filtered = X[:, ~mask_zero_std]
    if False:
        corr, _ = spearmanr(X_filtered, nan_policy='raise')
    else:
        print('start corr calculation')
        corr = sparse_corrcoef(X_filtered.T)
        print(corr.shape)
    try:
        net = efficient_melting(corr.A, gene_names)
    except:
        net = efficient_melting(corr, gene_names)
    return net 

def plot_gene_centrality_vs_expression(df_merged):
    hub_genes = df_merged[df_merged['is_hub_gene']]['gene'].unique()

    # - Generate a color palette
    color_palette = sns.color_palette('tab20', len(hub_genes))  # 'husl' is an example; choose any palette
    color_map = {gene:color for gene, color in zip(hub_genes, color_palette)}

    # - actual plot
    fig, axes = plt.subplots(1,2, figsize=(8,3.5), sharey=True, dpi=100)
    df_merged['show_name'] = 'Others'
    df_merged.loc[df_merged['is_hub_gene'], 'show_name'] = df_merged.loc[df_merged['is_hub_gene'], 'gene']
    hue_order = df_merged['show_name'].unique()

    for i, batch in enumerate(df_merged['batch_group'].unique()):
        ax = axes[i]
        df_merged_b = df_merged[df_merged['batch_group'].eq(batch)]

        sns.scatterplot(df_merged_b, x='-fc_log10_pvalue', y='diff', hue='show_name', hue_order=hue_order,ax=ax, palette={'Others':'grey', **color_map}, style='is_tf')
        if i == 0:
            ax.get_legend().remove()
        else:  
            ax.legend(loc=(1.05, 0))
        ax.set_title(batch)
        # ax.set_xscale('log')
    plt.tight_layout()

def infer_grns_all(input_file,folder_tag='grn'):
    
    par = {
            'dataset_file':input_file,
            'save_dir': f'output/{folder_tag}/',
            'min_cells_per_gene': 500,
            'weight_t': .05,
            # 'tf_all': f'../task_grn_inference/resources/prior/tf_all.csv'
        }
    par['net_all'] = f"{par['save_dir']}/net_all.csv"
    os.makedirs(par['save_dir'], exist_ok=True)
    # - dependencies
    adata = ad.read_h5ad(par['dataset_file'])

    batches = ['batch_1', 'batch_2', 'all_batches']
    cell_types = list(adata.obs['cell_type'].unique())
    # batches = ['all_batches']
    # cell_types = ['all_celltypes']

    # - infer grns 
    i_exp = 0
    grns_store = []
    for i_cell_type, cell_type in enumerate(cell_types):
        if cell_type == 'all_celltypes':
            cell_type_mask = np.asarray([True for i in range(adata.shape[0])])
        else:
            cell_type_mask = (adata.obs['cell_type'] == cell_type)   
        
        for age_group in list(adata.obs['age_group'].unique()): # only age groups for -1 cell type
            age_group_mask = (adata.obs['age_group'] == age_group)
            for batch_group in batches:
                if batch_group == 'all_batches':
                    batch_group_mask = np.asarray([True for i in range(adata.shape[0])])
                else:
                    batch_group_mask = (adata.obs['batch_group'] == batch_group)
                save_file_name = os.path.abspath(f"{par['save_dir']}/net_{cell_type}_{age_group}_{batch_group}.csv")

                print(save_file_name)
                if os.path.exists(save_file_name):
                    print(f"File already exists. Skipping: {save_file_name}")
                    continue
                print(cell_type, age_group, batch_group)
                mask_sample = batch_group_mask & age_group_mask & cell_type_mask
                print(cell_type, age_group, batch_group)

                adata_sample = basic_qc(adata[mask_sample, :], min_cells_per_gene=par['min_cells_per_gene'], min_genes_per_cell=10)

                if (adata_sample.shape[0]==0):
                    continue
                    
                if adata_sample.shape[1]==0:
                    continue

                # - normalize 
                X_norm = sc.pp.normalize_total(adata_sample, inplace=False)['X']
                adata_sample.layers['X_norm'] = sc.pp.log1p(X_norm, copy=True)

                # - actual subset 
                expression_sample = adata_sample.layers['X_norm']
                gene_names = adata_sample.var_names

                # - infer grn
                net = infer_grn(expression_sample, gene_names)

                # net = net[net['source'].isin(tf_all)]

                net['weight'] = pd.to_numeric(net['weight'], errors='coerce')

                # net_short = net.loc[net['weight'].abs().nlargest(par['n_max_links']).index]
                net = net[net['weight'].abs()>par['weight_t']]

                # - save 
                net.to_csv(save_file_name)
                net['batch_group'] = batch_group
                net['cell_type'] = cell_type
                net['age_group'] = age_group

                grns_store.append(net)
                i_exp+=1
    grns = pd.concat(grns_store)
    grns.to_csv(par['net_all'])
def infer_grns_selected():
    map_cell_type_genernib = {
        'CD4+ T cells': 'T cells',
        'TRAV1-2- CD8+ T cells': 'T cells',
        'gd T cells': 'T cells',
        'DN T cells': 'T cells',
        'MAIT cells': 'T cells',
        'Progenitor cells': 'Myeloid cells',
        'B cells': 'B cells',
        'NK cells': 'NK cells',
        'Myeloid cells': 'Myeloid cells'
    }

    par = {
            'dataset_file': 'input/dataset_1_2.h5ad',
            'save_dir': f'output/grns',
            'min_cells_per_gene': 500,
            'n_max_links': 200_000,
            'tf_all': f'../task_grn_inference/resources/prior/tf_all.csv'
        }
    os.makedirs(par['save_dir'], exist_ok=True)
    # - dependencies
    adata = ad.read_h5ad(par['dataset_file'])
    tf_all = np.loadtxt(par['tf_all'], dtype=str)
    # - fix the granualariy of the cell typs based on geneRNIB
    adata.obs['cell_type_major'] = adata.obs['cell_type'].map(map_cell_type_genernib)
    adata.obs['cell_type_major'].unique()

    # Map 'age' to 'age_group_major' based on the threshold of 50
    adata.obs['age_group_major'] = adata.obs['age'].apply(lambda x: 'young' if x < 50 else 'old')
    adata.obs['age_group_major'].unique()

    # - infer grns for 10 conditions
    i_exp = 0
    # for i_cell_type, cell_type in enumerate(list(adata.obs['cell_type_major'].unique())+['all_celltypes']):
    for i_cell_type, cell_type in enumerate(['all_celltypes']):
        if cell_type == 'all_celltypes':
            cell_type_mask = np.asarray([True for i in range(adata.shape[0])])
        else:
            cell_type_mask = (adata.obs['cell_type_major'] == cell_type)   
        
        for age_group in list(adata.obs['age_group_major'].unique())+['all_ages']: # only age groups for -1 cell type
            if age_group == 'all_ages':
                age_group_mask = np.asarray([True for i in range(adata.shape[0])])
            else:
                if cell_type != 'all_celltypes': # only for all cell types
                    continue 
                age_group_mask = (adata.obs['age_group_major'] == age_group)
            for batch_group in ['all_batches', 'batch_1']:
                
                if batch_group == 'all_batches':
                    batch_group_mask = np.asarray([True for i in range(adata.shape[0])])
                else:
                    if cell_type != 'all_celltypes': # only for all cell types
                        continue 
                    batch_group_mask = (adata.obs['batch_group'] == batch_group)

                mask_sample = batch_group_mask & age_group_mask & cell_type_mask
                print(cell_type, age_group, batch_group)

                adata_sample = basic_qc(adata[mask_sample, :], min_cells_per_gene=par['min_cells_per_gene'], min_genes_per_cell=10)

                if (adata_sample.shape[0]==0):
                    continue
                    
                if adata_sample.shape[1]==0:
                    continue

                # - normalize 
                X_norm = sc.pp.normalize_total(adata_sample, inplace=False)['X']
                adata_sample.layers['X_norm'] = sc.pp.log1p(X_norm, copy=True)

                # - actual subset 
                # expression_sample = adata_sample.layers['X_norm'].todense().A
                expression_sample = adata_sample.layers['X_norm']
                gene_names = adata_sample.var_names

                # - infer grn
                net = infer_grn(expression_sample, gene_names)

                net = net[net['source'].isin(tf_all)]

                net['weight'] = pd.to_numeric(net['weight'], errors='coerce')

                net_short = net.loc[net['weight'].abs().nlargest(par['n_max_links']).index]

                # - save 
                net_short.to_csv(f"{par['save_dir']}/net_{cell_type}_{age_group}_{batch_group}.csv")
                i_exp+=1
def sparse_std(X):
    from sklearn.preprocessing import StandardScaler
    scalar = StandardScaler(with_mean=False)
    scalar.fit(X)
    X_var = scalar.var_
    return X_var

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


if __name__ == '__main__': 
    if True: #- GRN inference
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--data_file',
            type=str,
            required=True,
            help="Path to save the dataset file (e.g., 'input/dataset_1.h5ad')."
        )
        
        args = parser.parse_args()
        data_file=args.data_file

        file_name = data_file.split('/')[-1].split('.')[0]
        folder_tag = f'grns/{file_name}'
        print(folder_tag, data_file)
        

        infer_grns_all(input_file=data_file, folder_tag=folder_tag)


    # run_DEA()