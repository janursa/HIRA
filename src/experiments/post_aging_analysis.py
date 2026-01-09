#!/usr/bin/env python
"""

This script performs post-run analysis for age-associated TFs.
generating various visualizations including:

"""

import argparse
import os
import sys
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
pd.set_option("display.max_columns", None)

# Import common utilities and configuration
from ciim.src.config import (
    PLOTS_DIR, 
    CELL_TYPES, 
    DISCOVERY_COHORTS,
    palette_trend,
    surrogate_names
)
from ciim.src.config import get_config
from ciim.src.feature_association.helper import retrieve_sig_stats

from ciim.src.utils.util import retrieve_net_consensus
from ciim.src.config import palette_cell_types, palette_datasets, palette_trend_2, colors_blind
from ciim.src.feature_association.helper import retrieve_sig_net, retrieve_sig_stats, retrieve_features_stats
from ciim.src.feature_association.plots import plot_sig_tfs_stats


warnings.filterwarnings("ignore")

# Set matplotlib defaults
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "Arial"



## Heatmap of sig TFs across cell types and cohorts
def plot_heatmap_overal():
    from ciim.src.feature_association.plots import plot_overall_heatmap

    stats_all['cell_type'] = pd.Categorical(stats_all['cell_type'], categories=CELL_TYPES, ordered=True)
    stats_all['dataset'] = pd.Categorical(stats_all['dataset'], categories=DISCOVERY_COHORTS, ordered=True)

    plot_overall_heatmap(stats_all, 
                        sig_dots_y_offset=3, 
                        first_col='cell_type',  first_col_palette=palette_cell_types,
                        second_col='dataset', second_col_palette=palette_datasets,
                        bbox_to_anchor=(1.05, 1.05),
                        bbox_to_anchor_col2=(1.05, .85),
                        bbox_to_anchor_col1=(1.05, 0.37),
                        figsize=(4, 6),
                        map_names={**{'cell_type':'Cell type', 'dataset': 'Dataset'}, **surrogate_names})
    file_name = f"{PLOTS_DIR}/overall_heatmap.png"
    print(f"Saving figure to {file_name}")
    plt.savefig(file_name, bbox_inches='tight', dpi=300)


## Sig TFs counts
def plot_sig_tf_counts(args):
    aging_stats_sig = retrieve_sig_stats(data_type=args.data_type, filter_inconsistent=True).drop_duplicates(subset=['cell_type', 'gene'])
    # plt.savefig(f'../output/tf_activity/trends.png', dpi=300)
    aging_stats_sig['cell_type'] = pd.Categorical(aging_stats_sig['cell_type'], categories=CELL_TYPES, ordered=True)
    plot_sig_tfs_stats(aging_stats_sig, figsize=(2, 1.5), palette=palette_trend_2)
    # plt.title(f'Number of aging TFs', pad=15, fontsize=10, weight='bold')
    file_name = f"{PLOTS_DIR}/aging_tfs_count.png"
    print(f"Saving figure to {file_name}")
    plt.savefig(file_name, bbox_inches='tight', dpi=300, transparent=True)


## Identify sig networks
def plot_sig_networks(data_type = 'bulk'):
    from ciim.src.feature_association.helper import determine_sig_network
    
    

    if False:
        if True:
            determine_sig_network(data_type, min_degree=3)
        sig_net = retrieve_sig_net()
        sig_net_size = sig_net.groupby('cell_type').size()
        sig_net_size = sig_net_size.reindex(CELL_TYPES, fill_value=0).reset_index()
        sig_net_size.columns = ['cell_type', 'edge_count']

        # Ensure categorical order for plotting
        sig_net_size['cell_type'] = pd.Categorical(sig_net_size['cell_type'], categories=CELL_TYPES, ordered=True)
        sig_net_size = sig_net_size.sort_values('cell_type')

        # Plot
        fig, ax = plt.subplots(figsize=(1.5, 2))
        ax.barh(sig_net_size['cell_type'], sig_net_size['edge_count'], color=colors_blind[5], alpha=.5)

        # Aesthetics
        ax.invert_yaxis()
        ax.set_xlabel('TF-target pair')
        ax.set_ylabel('')
        ax.spines[['top', 'right']].set_visible(False)
        ax.margins(y=0.1, x=0.1)
        # ax.set_xscale('log')
        ax.set_title('Aging GRNs size', weight='bold', fontsize=10, pad=15)


    if False:
        sig_net = retrieve_sig_net()
        n_top = 20
        cell_type = 'CD8T'
        df = sig_net[sig_net['cell_type'] == cell_type]

        # Compute top source and target degrees
        source_counts = df['source'].value_counts().head(n_top).reset_index()
        source_counts.columns = ['source', 'degree']
        source_info = df[['source', 'trend_source']].drop_duplicates(subset='source')
        source_df = source_counts.merge(source_info, on='source', how='left')

        target_counts = df['target'].value_counts().head(n_top).reset_index()
        target_counts.columns = ['target', 'degree']
        target_info = df[['target', 'trend_target']].drop_duplicates(subset='target')
        target_df = target_counts.merge(target_info, on='target', how='left')

        # Setup plot
        fig, axes = plt.subplots(1, 2, figsize=(3, 4), sharey=False)

        # Plot sources
        source_colors = source_df['trend_source'].map(palette_trend).values
        target_colors = target_df['trend_target'].map(palette_trend).values[::-1]
        axes[0].barh(source_df['source'], source_df['degree'], color=source_colors)
        axes[0].set_title('TFs', fontsize=10)
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Out-degree')
        axes[0].spines[['top', 'right']].set_visible(False)

        # Plot targets
        axes[1].barh(target_df['target'][::-1], target_df['degree'][::-1], color=target_colors)
        axes[1].set_title('Targets', fontsize=10)
        axes[1].set_xlabel('In-degree')
        axes[1].spines[['top', 'right']].set_visible(False)

        # Title and layout
        fig.suptitle(f'Aging TFs and targets: {cell_type}', fontsize=10, weight='bold', y=.95)
        from matplotlib.patches import Patch

        legend_elements = [Patch(facecolor=color, label=label) for label, color in palette_trend_2.items()]
        # fig.legend(handles=legend_elements, bbox_to_anchor=(1.5, .8), fontsize=10, frameon=False, title='Trend', title_fontsize=10)
        fig.tight_layout()
def plot_central_features(data_type = 'bulk'):
    if True:
        from matplotlib.patches import Patch

        # Parameters
        n_top = 10
        cell_types_selected = ['CD4T', 'CD8T', 'NK', 'MONO']
        focus = 'source'  # can be 'source' or 'target'
        palette = palette_trend_2  # assume this dict is defined

        # Collect top TFs across cell types
        rows = []
        for ct in cell_types_selected:
            # Get consensus network for this cell type
            df_ct = retrieve_net_consensus(cell_type=ct)
            
            # Count degree for sources
            counts = df_ct[focus].value_counts().head(n_top).reset_index()
            counts.columns = [focus, 'degree']
            
            # Get trend information from stats_all
            stats_ct = stats_all[stats_all['cell_type'] == ct].copy()
            if focus == 'source':
                # Merge with TF stats to get trend
                stats_ct = stats_ct[['gene', 'slope']].drop_duplicates(subset='gene')
                stats_ct.columns = ['source', 'slope']
                merged = counts.merge(stats_ct, on='source', how='left')
                # Handle missing slopes - assign a small positive value to avoid NaN
                merged['slope'] = merged['slope'].fillna(0.001)
                # Map to palette_trend_2 keys
                merged['trend_source'] = merged['slope'].apply(
                    lambda x: 'Increase in aging' if x > 0 else 'Decrease in aging'
                )
                trend_col = 'trend_source'
            else:
                # For targets, we would need target stats
                # For now, set a default trend
                merged = counts.copy()
                merged['trend_target'] = 'Increase in aging'  # default
                trend_col = 'trend_target'
            
            merged['cell_type'] = ct
            
            
            # Calculate normalized centrality (normalize by max degree so top TF = 1.0)
            max_degree = merged['degree'].max()
            merged['normalized_centrality'] = merged['degree'] / max_degree if max_degree > 0 else 0
            
            rows.append(merged)

        plot_df = pd.concat(rows, ignore_index=True)

        # Determine trend column based on focus
        trend_col = 'trend_source' if focus == 'source' else 'trend_target'
        
        # Sort TFs within each cell type
        col_c = 'normalized_centrality'
        plot_df[focus] = plot_df[focus].astype(str)
        plot_df = plot_df.sort_values(by=['cell_type', col_c], ascending=[True, False])

        # Setup plot
        n_cols = len(cell_types_selected)
        fig, axes = plt.subplots(1, n_cols, figsize=(1.5*n_cols, 2.1 ), sharex=False)

        if n_cols == 1:
            axes = [axes]
        i = 0
        for ax, ct in zip(axes, cell_types_selected):
            df_ct = plot_df[plot_df['cell_type'] == ct]
            colors = df_ct[trend_col].map(palette)
            ax.barh(df_ct[focus], df_ct[col_c], color=colors, alpha=0.5)
            ax.set_title(f'{ct}', fontsize=10)
            ax.invert_yaxis()
            if i==0:
                ax.set_xlabel('Out-degree centrality' if focus == 'source' else 'In-degree centrality', fontsize=8)
            else:
                ax.set_xlabel('')
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
            ax.margins(x=0.1, y=0.1)
            ax.spines[['top', 'right']].set_visible(False)
            i+=1
        # Title and legend
        # fig.suptitle(f'Central aging {"TFs" if focus == "source" else "targets"}', fontsize=12, weight='bold', y=0.96)
        legend_elements = [Patch(facecolor=color, label=label) for label, color in palette.items()]
        # fig.legend(handles=legend_elements, bbox_to_anchor=(1.01, 0.8), loc='upper left', title='Trend', title_fontsize=10, frameon=False)
        fig.tight_layout()
        file_name = os.path.join(PLOTS_DIR, f'central_aging_{focus}.png')
        print(f"Saving figure to {file_name}")
        plt.savefig(file_name, bbox_inches='tight', dpi=300, transparent=True)
def plot_interaction_of_aging_TFs_between_cell_types(data_type = 'bulk'):
    from grn_benchmark.src.exp_analysis.helper import plot_interactions, create_interaction_df

    stats_sig = retrieve_sig_stats(data_type=data_type).drop_duplicates(subset=['cell_type', 'gene'])
    df_dict = stats_sig.groupby(['cell_type'])['gene'].apply(list).to_dict()
    interaction_main_df = create_interaction_df(df_dict)
    aa = plot_interactions(interaction_main_df, min_subset_size=5, min_degree=1, color_map=palette_cell_types)
    file_name = f"{PLOTS_DIR}/interactions.png"
    print(f"Saving figure to {file_name}")
    plt.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')
    # plt.title(surrogate_names[race], pad=40, fontsize=10, fontweight='bold')

    from ciim.src.feature_association.plots import plot_features_vs_datasets
    ttypes = ['CD8T', 'CD4T', 'NK']
    mask = interaction_main_df[ttypes].sum(axis=1)==len(ttypes)
    features = mask[mask].index.unique()
    print(len(features))
    for cell_type in ttypes:
        plot_features_vs_datasets(cell_type=cell_type, data_type=data_type, datasets=DISCOVERY_COHORTS, features=features, 
                                    feature_type='tf_activity', sizes=(90, 100), min_degree=1, race='both', 
                                    filter_meta_significant=True,
                                    )
        file_name = f"{PLOTS_DIR}/features_vs_datasets_{cell_type}.png"
        print(f"Saving figure to {file_name}")
        plt.savefig(file_name, bbox_inches='tight', dpi=300, transparent=True)
    if False: ### Shared sig TFs between CD8T and CD4T
        from ciim.src.feature_association.plots import plot_joint_scatter
        ttypes = ['CD8T', 'CD4T']
        mask = interaction_main_df[ttypes].sum(axis=1)==len(ttypes)
        features = mask[mask].index
        # Get significant TFs for CD4T and CD8T cells
        df = stats_sig[stats_sig['gene'].isin(features)].drop_duplicates(subset=['cell_type', 'gene'])
        df['neg_log10_adj_pval'] = -np.log10(df['meta_p_adj'])

        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        plot_joint_scatter(df, vars=['CD4T', 'CD8T'], annotate=True, ax=ax)
        ax.margins(x=0.1, y=0.1)
def gsea_analysis():
    from ciim.src.pathway_analysis.util import get_genesets, pathway_kde_func, get_hallmark, gsea_func, wrapper_gsea

    wrapper_gsea(stats_sig)
    file_name = f"{PLOTS_DIR}/gsea_tf_activity.png"
    print(f"Saving figure to {file_name}")
    plt.savefig(file_name, bbox_inches='tight', dpi=200)

def plot_case_tf(data_type='bulk'):
    from ciim.src.feature_association.plots import plot_feature_values_all_datasets, plot_feature_values_per_datasets
    n_top_targets = 10
    selected_cell_types = ['CD8T', 'CD4T', 'NK'] # ['CD8T', 'CD4T', 'NK'] #Tcm_Naive_CD8
    n_panels = len(selected_cell_types)
    datasets = DISCOVERY_COHORTS
    plot_both = True
    show_cbar=False
    for case_tf in ['SATB1', 'GATA3']:  # 'TCF7' 'SATB1' 
        for i, cell_type in enumerate(selected_cell_types):
            if i == 0:
                show_ylabels=True
            else:
                show_ylabels=False
            
            if plot_both:
                fig, axes = plt.subplots(2, 1, figsize=(2, 2.2), sharex=True)
            else:
                fig, ax = plt.subplots(1, 1, figsize=(2, .7))

            ax = axes[0] if plot_both else ax
            plot_feature_values_all_datasets(cell_type, feature=case_tf, feature_type='tf_activity', data_type=data_type, datasets=datasets, show_cbar=show_cbar, ax=ax,
                                            show_ylabels=show_ylabels)
            if plot_both:
                ax.set_xlabel('')
                ax = axes[1]
                plot_feature_values_all_datasets(cell_type, feature=case_tf, feature_type='gene_expression', 
                                                data_type=data_type, datasets=datasets, show_cbar=show_cbar, ax=ax, show_ylabels=show_ylabels)
            
            plt.suptitle(f'{cell_type}', y=1.05)
            file_name = f"{PLOTS_DIR}/case_tf_{case_tf}_{cell_type}.png"
            print(f"Saving figure to {file_name}")
            plt.savefig(file_name, bbox_inches='tight', dpi=300)

def plot_sig_genes_counts_hallmarks(data_type='bulk'):
    from ciim.src.config import PRIOR_DIR
    # Load aging hallmark genes with gene set information
    gene_col = 'gene' 
    geneset_col = 'gene_set'
    aging_hallmark_path = f'{PRIOR_DIR}/aging_hallmark_genes.csv'
    if not os.path.exists(aging_hallmark_path):
        raise FileNotFoundError(f'Aging hallmark genes file not found at {aging_hallmark_path}')
    
    aging_hallmark_df = pd.read_csv(aging_hallmark_path)   
    
    # Load significant stats for aging hallmarks
    stats_sig = retrieve_sig_stats(data_type, feature_type='aging_hallmarks', filter_inconsistent=True)
    # Merge with gene set information
    stats_with_geneset = stats_sig.merge(
        aging_hallmark_df[[gene_col, geneset_col]],
        left_on='gene',  # In the stats, genes are stored in 'gene' column
        right_on='gene',
        how='left'
    )
    
    # Calculate counts per cell type and gene set
    stats_with_geneset['trend'] = stats_with_geneset['slope'].apply(
        lambda x: 'Increase in aging' if x > 0 else 'Decrease in aging'
    )
    
    # Count significant genes per cell type and gene set
    count_df = stats_with_geneset.groupby(['cell_type', geneset_col, 'trend']).size().reset_index(name='count')

    # Pivot to get separate columns for increase and decrease
    count_pivot = count_df.pivot_table(
        index=['cell_type', geneset_col],
        columns='trend',
        values='count',
        fill_value=0
    ).reset_index()
    
    # Get unique cell types and gene sets
    cell_types_unique = count_pivot['cell_type'].unique()
    genesets_unique = sorted(count_pivot[geneset_col].unique())
    
    # Create one plot per cell type
    from matplotlib.patches import Patch
    
    for ct in cell_types_unique:
        ct_data = count_pivot[count_pivot['cell_type'] == ct]
        
        # Align data with genesets_unique order
        increase_counts = []
        decrease_counts = []
        
        for gs in genesets_unique:
            gs_data = ct_data[ct_data[geneset_col] == gs]
            if len(gs_data) > 0:
                increase_counts.append(gs_data['Increase in aging'].values[0] if 'Increase in aging' in gs_data.columns else 0)
                decrease_counts.append(gs_data['Decrease in aging'].values[0] if 'Decrease in aging' in gs_data.columns else 0)
            else:
                increase_counts.append(0)
                decrease_counts.append(0)
        
        # Plot
        fig, ax = plt.subplots(figsize=(max(6, len(genesets_unique) * 0.6), 4))
        
        x = np.arange(len(genesets_unique))
        width = 0.35
        
        # Plot bars for increase and decrease
        bars1 = ax.bar(x - width/2, increase_counts, width, 
                        label='Increase in aging', 
                        color=palette_trend_2['Increase in aging'], 
                        alpha=0.7)
        bars2 = ax.bar(x + width/2, decrease_counts, width, 
                        label='Decrease in aging', 
                        color=palette_trend_2['Decrease in aging'], 
                        alpha=0.7)
        
        # Aesthetics
        ax.set_xlabel('Gene Set', fontsize=10)
        ax.set_ylabel('Number of Significant Genes', fontsize=10)
        ax.set_title(f'Aging-associated genes: {ct}', fontsize=12, weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(genesets_unique, rotation=45, ha='right', fontsize=9)
        ax.legend(frameon=False, loc='upper left')
        ax.spines[['top', 'right']].set_visible(False)
        ax.margins(x=0.02)
        
        plt.tight_layout()
        file_name = f"{PLOTS_DIR}/aging_hallmarks_by_geneset_{ct}.png"
        print(f"Saving figure to {file_name}")
        plt.savefig(file_name, bbox_inches='tight', dpi=300, transparent=True)
        plt.close()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-type', type=str, required=True, help='Feature type to analyze')
    parser.add_argument('--data-type', type=str, default='bulk', required=False, help='Data type to analyze')
    args = parser.parse_args()
    feature_type = args.feature_type
    data_type = args.data_type

    stats_all = retrieve_features_stats(data_type, feature_type=feature_type)

    # - add sig signs
    stats_sig = retrieve_sig_stats(data_type, feature_type=feature_type)
    tuple_index = stats_sig.set_index(['cell_type', 'gene', 'dataset']).index

    stats_all = stats_all.set_index(['cell_type', 'gene', 'dataset'])
    stats_all['is_significant'] = False
    stats_all.loc[tuple_index, 'is_significant'] = True

    stats_all = stats_all.reset_index()[['gene', 'cell_type', 'dataset', 'slope', 'is_significant']].drop_duplicates()

    if feature_type == 'tf_activity':
        plot_heatmap_overal()
        plot_sig_tf_counts(args)
        plot_central_features()
        plot_interaction_of_aging_TFs_between_cell_types()
        gsea_analysis()
        plot_case_tf()

        # plot_sig_networks()
    elif feature_type == 'aging_hallmarks':
        plot_sig_genes_counts_hallmarks()
    else:
        raise ValueError('Unknown feature type')