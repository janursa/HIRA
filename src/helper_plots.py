
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



def plot_pathway_interactions(pathway_interactions_df, ax):
    # plot
    all_pathways = pd.Index(np.unique(np.union1d(pathway_interactions_df['pathway_1'], pathway_interactions_df['pathway_2'])))
    pathway_matrix = pd.pivot(
        pathway_interactions_df,
        index='pathway_1',
        columns='pathway_2',
        values='-log10_pvalue'
    ).fillna(0).reindex(index=all_pathways, columns=all_pathways, fill_value=0)
    # Symmetrize the matrix if pathways are bidirectional
    pathway_matrix = pathway_matrix + pathway_matrix.T - np.diag(np.diag(pathway_matrix))

    sns.heatmap(
        pathway_matrix, 
        cmap='viridis', 
        square=True, 
        annot=False, 
        cbar=None,
        ax=ax
    )
    ax.set_xlabel('')
    ax.set_ylabel('')


def exp_plots(groups, cell_type=True):
        
    # - plot dist of sex and cell types
    def norm_size(series):
        normalized = series.value_counts(normalize=True)
        normalized.index.name='index'
        return normalized
    
    cellcount_dist = groups.size().reset_index(name='cell_count')
    donor_dist = groups['donor_id'].nunique().reset_index(name='donor_n')
    age_donor_dist = groups['age_donor'].nunique().reset_index(name='age_donor')
    age2donor_dist = groups.apply(lambda df: df.groupby('donor_id')['age'].nunique()).reset_index(name='count')


    sex_ratio = groups['sex'].apply(norm_size).reset_index(name='ratio')
    sex_ratio = sex_ratio.rename(columns={'index':'sex'})

    cell_type_ratio = groups['cell_type'].apply(norm_size).reset_index(name='ratio')
    cell_type_ratio = cell_type_ratio.rename(columns={'index':'cell_type'})

    cell_count_donors = groups.apply(lambda df: df.groupby('age_donor').size()).reset_index(name='count')
    

    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 7), gridspec_kw={'width_ratios':[1, 1, 1, 2]})


    # distribution of cell count
    ax = axes[0][0]
    sns.barplot(data=cellcount_dist, x='age_group', y='cell_count', ax=ax)
    ax.set_title('Cell count')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)

    # distribution of age 2 donor 
    ax = axes[1][2]
    sns.stripplot(data=age2donor_dist, x='age_group', y='count', ax=ax)
    ax.set_title('Age 2 donor')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)

    # distribution of donors
    ax = axes[0][1]
    sns.barplot(data=donor_dist, x='age_group', y='donor_n', ax=ax)
    ax.set_title('Donor count')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)

    # distribution of age donor
    ax = axes[0][2]
    sns.barplot(data=age_donor_dist, x='age_group', y='age_donor', ax=ax)
    ax.set_title('Age donor count')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)

    # Plot for sex
    ax = axes[1][0]
    sns.barplot(data=sex_ratio, x='age_group', y='ratio', hue='sex', ax=ax)
    ax.set_title('Ratio of Sex per Age Group')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Ratio')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc=(.5,.5))

    # Plot for cell type
    if cell_type:
        ax = axes[0][3]
        sns.barplot(data=cell_type_ratio, x='age_group', y='ratio', hue='cell_type', ax=ax)
        ax.set_title('Ratio of Cell Type per Age Group')
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Ratio')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(loc=(1.1,.2))

    # cell count distribution for donor-age 
    ax = axes[1][1]
    sns.stripplot(data=cell_count_donors, x='age_group', y='count', ax=ax)
    ax.set_title('Counts seg. by donor-age')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Cell count')
    ax.tick_params(axis='x', rotation=45)


    # Adjust layout
    plt.tight_layout()

    # Show the plots
    plt.show()
def plot_centrality_cluster(groups, normalize=True, save_name='output/figs/degree.png', figsize=(4, 20), degree_t = 10, target_groups=['45_54','65_75']):
    def link_to_centrality(df):
        df['weight'] = 1
        gg_mat = link_to_matrix(df)
        # Calculate degree centrality (sum of weights for each node)
        degree = gg_mat.sum(axis=1)  # sum by row to get degree centrality
        
        return degree

    # Apply the function to the groups
    centrality_df = groups.apply(link_to_centrality).reset_index(level=0, name='degree').pivot(columns='age_group', values='degree').fillna(0)

    
    centrality_df = centrality_df[(centrality_df>degree_t)[target_groups].any(axis=1)]

    if normalize:
        # Normalize the data (optional, for better visual contrast)
        df = (centrality_df - centrality_df.min()) / (centrality_df.max() - centrality_df.min())
    else:
        df = centrality_df

    # Create the clustermap with row clustering
    g = sns.clustermap(df, cmap="Blues", row_cluster=True, col_cluster=False, 
                    figsize=figsize, linewidths=0.01, xticklabels=True, yticklabels=True, 
                    cbar_pos=(0.95, 0.2, 0.03, 0.45))
    # g.ax_heatmap.set_position([0.05, 0.05, 0.8, 0.8])
    # Make the x and y tick labels (gene names) smaller
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), fontsize=5)  # Adjust x-axis gene names
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), fontsize=5)  # Adjust y-axis gene names
    plt.tight_layout()
    # g.fig.suptitle('', y=0.8)

    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
def heatmap_pathways(g_g_matrix:pd.DataFrame, pathway_df:pd.DataFrame):
    '''
    g_g_matrix: df with genes names on index and column
    pathway_df: gene to pathway connection df, where index is gene name and pathway column is pathways
    '''
    # Sort the genes based on the pathways for better visualization

    pathway_df = pathway_df[pathway_df.index.isin(g_g_matrix.index)]
    sorted_genes = pathway_df.index
    expanded_g_g_matrix = pd.DataFrame(index=sorted_genes, columns=sorted_genes)

    # Loop over the gene pairs to populate expanded_g_g_matrix from g_g_matrix
    for i, gene_i in enumerate(sorted_genes):
        for j, gene_j in enumerate(sorted_genes):
            # Use g_g_matrix values if both genes are present, otherwise keep NaN
            if gene_i in g_g_matrix.index and gene_j in g_g_matrix.columns:
                expanded_g_g_matrix.iloc[i, j] = g_g_matrix.loc[gene_i, gene_j]
            else:
                expanded_g_g_matrix.iloc[i, j] =0  # or use 0 if you prefer

    # Convert the matrix to numeric, ensuring all values are floats
    expanded_g_g_matrix = expanded_g_g_matrix.apply(pd.to_numeric, errors='coerce')

    # Fill NaNs with a small value or 0 (depending on your preferences)
    expanded_g_g_matrix = expanded_g_g_matrix.fillna(0)  # or use np.nan if you prefer

    pathway_colors = sns.color_palette('Set2', len(pathway_df['pathway'].unique()))
    pathway_color_map = dict(zip(pathway_df['pathway'].unique(), pathway_colors))
    row_colors = pathway_df['pathway'].map(pathway_color_map)

    # Set figure size
    figsize = (10, 10)

    # Create the heatmap with the gene-to-gene connection strengths
    g = sns.clustermap(expanded_g_g_matrix, row_colors=row_colors, col_colors=row_colors, 
                    cmap="coolwarm", linewidths=0.01, xticklabels=True, yticklabels=True, 
                    row_cluster=False, col_cluster=False,
                    cbar_pos=(.1, 0.2, 0.03, 0.6))  # cbar_pos to the left
    # Make the x and y tick labels (gene names) smaller
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), fontsize=4)  # Adjust x-axis gene names
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), fontsize=4)  # Adjust y-axis gene names

    # Adjust the figure size
    g.fig.set_size_inches(figsize)

    # Create a custom legend showing pathway names and their colors
    for pathway, color in pathway_color_map.items():
        plt.plot([], [], marker="o", ms=10, ls="", mec=None, color=color, label=f"{pathway}")

    # Customize plot labels and add the legend on the right
    plt.legend(loc='lower left', title="Pathways", bbox_to_anchor=(30, 0.5), borderaxespad=0)

    # Show the plot
    plt.tight_layout()
def G_plot(df, ax=None):
    import networkx as nx

    df[['source', 'target']] = df['link'].str.split('_', expand=True)
    if 'mean' not in df.columns:
        df['mean'] = 1
    G = nx.from_pandas_edgelist(df, 'source', 'target', edge_attr='mean')
    pos = nx.circular_layout(G)  
    nx.draw(G, pos, with_labels=True, node_size=200, node_color='lightblue', font_size=6, font_weight='normal', edge_color='gray', ax=ax)


def plot_exp(groups, suptitle):
    def deg_cent(df):
        df['weight'] = 1
        gg_mat = link_to_matrix(df)
        degree = gg_mat.sum()
        return degree
    def sum_cent(df):
        df['weight'] = df['mean']
        gg_mat = link_to_matrix(df)
        sum_mean = gg_mat.abs().sum()
        return sum_mean

    n_sig_edges = groups.size().reset_index(name='# sig edges')
    mean_corr = groups.apply(lambda df: df['mean']).reset_index(name='mean_corr')
    diff_corr = groups.apply(lambda df: df['diff']).reset_index(name='diff_corr')
    cent_degree = groups.apply(deg_cent).reset_index(name='degree')
    cent_sum = groups.apply(sum_cent).reset_index(name='sum_mean')
    

    # Create subplots
    fig, axes = plt.subplots(1, 5, figsize=(22, 4), gridspec_kw={'width_ratios':[1, 1, 1, 1, 1]})

    # Plot # of edges
    sns.barplot(data=n_sig_edges, x='age_group', y='# sig edges',ax=axes[0])
    axes[0].set_title('# sig edges per group')
    axes[0].set_xlabel('Age Group')
    axes[0].set_ylabel('# sig edges')
    axes[0].tick_params(axis='x', rotation=45)

    # Plot mean corr for each group
    ax = axes[1]
    sns.stripplot(data=mean_corr, x='age_group', y='mean_corr',ax=ax)
    ax.set_title('Mean corr')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Mean corr.')
    ax.tick_params(axis='x', rotation=45)

    # Plot diff corr for each group
    ax = axes[4]
    sns.stripplot(data=diff_corr, x='age_group', y='diff_corr',ax=ax)
    ax.set_title('Diff corr')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Diff corr.')
    ax.tick_params(axis='x', rotation=45)


    # Plot distribution of centrality degree
    ax = axes[2]
    sns.violinplot(data=cent_degree, x='age_group', y='degree',ax=ax)
    ax.set_title('Centrality degree per group')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Degree')
    ax.tick_params(axis='x', rotation=45)

    # Plot distribution of centrality sum
    ax = axes[3]
    sns.violinplot(data=cent_sum, x='age_group', y='sum_mean',ax=ax)
    ax.set_title('Centrality sum per group')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Sum of mean corr')
    ax.tick_params(axis='x', rotation=45)


    fig.suptitle(suptitle, fontsize=16)

    # Adjust layout
    plt.tight_layout()

def plot_centrality_by_age(cell_type, centrality_df):
    """
    Plots heatmaps showing the changes in centrality over age for a given cell type and two batch groups.

    Parameters:
    - cell_type (str): The cell type to filter and plot.
    - centrality_df (pd.DataFrame): The DataFrame containing centrality data with columns:
      ['gene', 'age_group', 'centrality', 'batch_group', 'cell_type'].
    """
    top_genes = (
        centrality_df.groupby('gene')['centrality']
        .std()
        .nlargest(20)
        .index
    )

    # Filter centrality data for top genes
    centrality_df = centrality_df[centrality_df['gene'].isin(top_genes)]
    # Filter the dataframe for the specified cell type
    cell_df = centrality_df[centrality_df['cell_type'] == cell_type]
    
    # Ensure there are entries to plot
    if cell_df.empty:
        print(f"No data available for cell type: {cell_type}")
        return
    
    # Create the subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    
    for i, (batch_group, batch_df) in enumerate(cell_df.groupby('batch_group')):
        # Pivot the table for heatmap
        pivot_df = batch_df.pivot(index='gene', columns='age_group', values='centrality')
        
        # Plot the heatmap
        ax = axes[i]
        sns.heatmap(
            pivot_df,
            cmap='viridis',
            annot=False,
            ax=ax,
            cbar=True
        )
        ax.set_title(f'Batch: {batch_group}')
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Gene')
    
    # Add an overall title
    fig.suptitle(f'Centrality Change Over Age for {cell_type}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    plt.show()

def plot_pathway_change_age(filtered_interaction_score_all, cell_type):
    # Filter data for the current cell type
    subset = filtered_interaction_score_all[
        filtered_interaction_score_all['cell_type'] == cell_type
    ]
    # Separate data by batch groups
    batches = subset['batch_group'].unique()
    fig, axes = plt.subplots(1, 2, figsize=(8, 6), sharey=True)

    for i, batch_group in enumerate(sorted(batches)):
        batch_subset = subset[subset['batch_group'] == batch_group]
        
        # Pivot the data for the heatmap
        batch_subset_pivot = batch_subset.pivot(
            index='link', 
            columns='age_group', 
            values='-log10_pvalue',
        )
        
        # Calculate the variation across age groups
        batch_subset_pivot['variation'] = batch_subset_pivot.var(axis=1)  # Use std() or ptp() if preferred
        
        # Select the top 20 links with the highest variation
        top_links = batch_subset_pivot.nlargest(20, 'variation').drop(columns=['variation'])
        
        # Plot the heatmap
        sns.heatmap(
            top_links, 
            cmap="viridis", 
            cbar_kws={'label': '-log10(p_adj)'}, 
            linewidths=0.5, 
            ax=axes[i]
        )
        axes[i].set_title(f"{batch_group} (Top 20 Links)")
        axes[i].set_xlabel("Age Group")
        axes[i].set_ylabel("Links" if i == 0 else "")  # Only show y-label for the left plot

    # Set the overall title and layout
    plt.suptitle(f"Heatmap of -log10(p_adj) for Cell Type: {cell_type}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
def plot_pathway_change_celltypes(filtered_interaction_score_all, age_group):
    subset = filtered_interaction_score_all[
        filtered_interaction_score_all['age_group'] == age_group
    ]
    
    # Compute -log10(p_adj)
    subset['-log10_pvalue'] = -np.log10(subset['p_adj'])
    
    # Separate data by batch groups
    batches = subset['batch_group'].unique()
    fig, axes = plt.subplots(1, 2, figsize=(8, 6), sharey=True)
    
    for i, batch_group in enumerate(sorted(batches)):
        batch_subset = subset[subset['batch_group'] == batch_group]
        
        # Pivot the data for the heatmap
        heatmap_data = batch_subset.pivot_table(
            index='link', 
            columns='cell_type', 
            values='-log10_pvalue',
            aggfunc='mean'  # In case there are duplicates
        )
        
        # Calculate the variation across cell types
        heatmap_data['variation'] = heatmap_data.var(axis=1)  # Use std() or ptp() if preferred
        
        # Select the top 20 links with the highest variation
        top_links = heatmap_data.nlargest(20, 'variation').drop(columns=['variation'])
        
        # Plot the heatmap
        sns.heatmap(
            top_links, 
            cmap="viridis", 
            cbar_kws={'label': '-log10(p_adj)'}, 
            linewidths=0.5, 
            ax=axes[i]
        )
        axes[i].set_title(f"{batch_group} (Top 20 Links)")
        axes[i].set_xlabel("Cell Type")
        axes[i].set_ylabel("Links" if i == 0 else "")  # Only show y-label for the left plot
    
    # Set the overall title and layout
    plt.suptitle(f"Heatmap of -log10(p_adj) for Age Group: {age_group}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
