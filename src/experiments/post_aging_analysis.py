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

# Import common utilities and configuration
from ciim.src.common import (
    PLOTS_DIR, 
    SAVE_DIR,
    cell_types, 
    datasets_all,
    palette_trend,
    palette_disease_effect,
    palette_treatment,
    surrogate_names,
    mapping_minor_2_major
)
from ciim.src.feature_association.config import get_config
from ciim.src.feature_association.helper import retrieve_sig_stats
from ciim.src.feature_association.plots import (
    heamap_plot_minor_cell_types,
    plot_overlap,
    plot_analysis_and_centrality,
    plot_donor_level_perturbation_effect
)
from ciim.src.feature_association.disease import plot_healthy_disease_trend
from ciim.src.utils.util import retrieve_net_consensus
from ciim.src.pathway_analysis.util import pathway_kde_func
from ciim.src.pathway_analysis.plots import plot_pathway_kde
from ciim.src.common import palette_cell_types, palette_datasets, palette_trend_2, colors_blind
from ciim.src.feature_association.helper import retrieve_sig_net, calculate_tf_activity, retrieve_sig_stats, retrieve_stats_features
from ciim.src.feature_association.plots import plot_net_nx, plot_sig_tfs_stats, plot_analysis_and_centrality, plot_overlap


warnings.filterwarnings("ignore")

# Set matplotlib defaults
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "Arial"

type = 'bulk'
stats_all = retrieve_stats_features(type, feature_type='tf_activity', condition='healthy')

# - add sig signs
stats_sig = retrieve_sig_stats(type)

tuple_index = stats_sig.set_index(['cell_type', 'tf', 'dataset']).index

stats_all = stats_all.set_index(['cell_type', 'tf', 'dataset'])
stats_all['is_significant'] = False
stats_all.loc[tuple_index, 'is_significant'] = True

stats_all = stats_all.reset_index()[['tf', 'cell_type', 'dataset', 'slope', 'is_significant']].drop_duplicates()


## Heatmap of sig TFs across cell types and cohorts
def plot_heatmap_overal():
    from ciim.src.feature_association.plots import plot_overall_heatmap

    stats_all['cell_type'] = pd.Categorical(stats_all['cell_type'], categories=cell_types, ordered=True)
    stats_all['dataset'] = pd.Categorical(stats_all['dataset'], categories=datasets_all, ordered=True)

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
def plot_sig_tf_counts():
    aging_stats_sig = retrieve_sig_stats(type='bulk', filter_inconsistent=True).drop_duplicates(subset=['cell_type', 'tf'])
    # plt.savefig(f'../output/tf_activity/trends.png', dpi=300)
    aging_stats_sig['cell_type'] = pd.Categorical(aging_stats_sig['cell_type'], categories=cell_types, ordered=True)
    plot_sig_tfs_stats(aging_stats_sig, figsize=(2, 1.5), palette=palette_trend_2)
    # plt.title(f'Number of aging TFs', pad=15, fontsize=10, weight='bold')
    file_name = f"{PLOTS_DIR}/aging_tfs_count.png"
    print(f"Saving figure to {file_name}")
    plt.savefig(file_name, bbox_inches='tight', dpi=300, transparent=True)


## Identify sig networks
def plot_sig_networks(type = 'bulk'):
    from ciim.src.feature_association.helper import determine_sig_network
    
    

    if False:
        if True:
            determine_sig_network(type, min_degree=3)
        sig_net = retrieve_sig_net()
        sig_net_size = sig_net.groupby('cell_type').size()
        sig_net_size = sig_net_size.reindex(cell_types, fill_value=0).reset_index()
        sig_net_size.columns = ['cell_type', 'edge_count']

        # Ensure categorical order for plotting
        sig_net_size['cell_type'] = pd.Categorical(sig_net_size['cell_type'], categories=cell_types, ordered=True)
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
def plot_central_features(type = 'bulk'):
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
                stats_ct = stats_ct[['tf', 'slope']].drop_duplicates(subset='tf')
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
def plot_interaction_of_aging_TFs_between_cell_types(type = 'bulk'):
    from grn_benchmark.src.exp_analysis.helper import plot_interactions, create_interaction_df

    stats_sig = retrieve_sig_stats(type=type).drop_duplicates(subset=['cell_type', 'tf'])
    df_dict = stats_sig.groupby(['cell_type'])['tf'].apply(list).to_dict()
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
        plot_features_vs_datasets(cell_type=cell_type, type=type, datasets=datasets_all, features=features, 
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
        df = stats_sig[stats_sig['tf'].isin(features)].drop_duplicates(subset=['cell_type', 'tf'])
        df['neg_log10_adj_pval'] = -np.log10(df['meta_p_adj'])

        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        plot_joint_scatter(df, vars=['CD4T', 'CD8T'], annotate=True, ax=ax)
        ax.margins(x=0.1, y=0.1)
def gsea_analysis():
    from ciim.src.pathway_analysis.util import get_genesets, pathway_kde_func, get_hallmark, gsea_func, wrapper_gsea

    wrapper_gsea(stats_sig, feature_type='tf_activity')

def plot_case_tf():
    case_tf = 'SATB1'  # 'TCF7' 'SATB1' 
    type = 'bulk'
    n_top_targets = 10
    selected_cell_types = ['CD8T', 'CD4T', 'NK'] # ['CD8T', 'CD4T', 'NK'] #Tcm_Naive_CD8
    n_panels = len(selected_cell_types)
    from ciim.src.feature_association.plots import plot_feature_values_all_datasets, plot_feature_values_per_datasets

    datasets = datasets_all

    plot_both = True
    show_cbar=False


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
        plot_feature_values_all_datasets(cell_type, feature=case_tf, feature_type='tf_activity', type=type, datasets=datasets, show_cbar=show_cbar, ax=ax,
                                        show_ylabels=show_ylabels)
        if plot_both:
            ax.set_xlabel('')
            ax = axes[1]
            plot_feature_values_all_datasets(cell_type, feature=case_tf, feature_type='gene_expression', 
                                            type=type, datasets=datasets, show_cbar=show_cbar, ax=ax, show_ylabels=show_ylabels)
        
        plt.suptitle(f'{cell_type}', y=1.05)
        file_name = f"{PLOTS_DIR}/case_tf_{case_tf}_{cell_type}.png"
        print(f"Saving figure to {file_name}")
        plt.savefig(file_name, bbox_inches='tight', dpi=300)

    if False:
        nets_sig = pd.read_csv(f'{SAVE_DIR}/sig_nets/sig_nets_{type}_{race}.csv')
        nets_sig = nets_sig[(nets_sig['source']==case_tf) & (nets_sig['cell_type'].isin(selected_cell_types))].copy()
        n_targets = 10

        stats_targets = retrieve_sig_stats(type, feature_type='gene_expression', race=race)

        targets_store = []
        for cell_type in selected_cell_types:
            net = nets_sig[nets_sig['cell_type']==cell_type]
            targets = net.sort_values('weight', ascending=False, key=abs).head(n_targets)['target'].unique()
            targets_store.append(targets)
        all_targets = np.concatenate(targets_store)

        net_store = []
        for cell_type in selected_cell_types:
            net = select_top_targets(cell_type, case_tf, datasets, all_targets, n_top=n_top_targets)
            net['cell_type'] = cell_type
            net_store.append(net)
        net_tf = pd.concat(net_store)

        from ciim.src.feature_association.plots import wrapper_flesh_out_tf_interactions, plot_tf_interactions_plus_target_stats_binary

        n_top_genes_per_dataset=10
        # - select 
        shared_state = net_tf.groupby(['cell_type','target'])['dataset'].nunique()
        selected_tuple = shared_state[shared_state>=3].index
        net_tf = net_tf.set_index(['cell_type', 'target']).loc[selected_tuple].reset_index()

        n_targets = net_tf['target'].nunique()
        net_tf['target'] = net_tf['target'].astype('category')
        net_tf['dataset'] = pd.Categorical(net_tf['dataset'], categories=datasets_all, ordered=True)
        fig, axes = plt.subplots(n_panels, 1, figsize=(n_targets*.16 +1, len(datasets_all)*.5 + 1), sharex=True)

        for i, (cell_type) in enumerate(net_tf['cell_type'].unique()):
            ax = axes[i] if n_panels > 1 else axes
            if i == 1:
                show_legend = True
            else:
                show_legend = False
            net = net_tf[net_tf['cell_type'] == cell_type]   
            net = net[net['neg_log10_adj_pval'] > 1.4]
            
            plot_tf_interactions_plus_target_stats_binary(net.copy(), ax=ax, show_legend=show_legend, sizes=(50, 200), annotate_sig=False, annotate_targets=True)

            ax.margins(y=0.2, x=0.1)
            ax.set_title(cell_type)
            ax.set_xlabel('Target genes')
            ax.grid(True, linestyle='--', alpha=0.5)
            if i != n_panels - 1:
                ax.set_ylabel('')
        plt.suptitle(f'Targets of {case_tf}', y=1.1, weight='bold', fontsize=12)
        plt.subplots_adjust(hspace=0.6)
    if False:
        type = 'bulk'
        cell_type = 'CD4T' #'Tcm_Naive_CD8'
        n_top=10
        # features = ['FOXO1', 'FOXO3', 'NFE2L2', 'TP53', 'SIRT1', 'HIF1A']
        # features = ['GATA3', 'S100A4', 'GZMK', 'KLF6', 'COTL1', 'TCF7', 'ANXA1']
        # features = ['GATA3', 'S100A4', 'GZMK', 'COTL1']
        features = ['FOS', 'FOSB', 'JUND']
        from ciim.src.feature_association.plots import wrapper_draw_net
        for cell_type in ['MONO']:
            wrapper_draw_net(cell_type, datasets_all, features, min_degree=3, indivitual_net=False, draw_evidence=True, draw_collectri=True, figsize=(2.5, 2.5), figsize_collectri=(4, 4), 
                            offset_evidence=.13, arc_offset=.05, only_promotor_based=False)
if __name__ == "__main__":
    plot_heatmap_overal()
    plot_sig_tf_counts()
    # plot_sig_networks()
    plot_central_features()
    plot_interaction_of_aging_TFs_between_cell_types()
    gsea_analysis()
    plot_case_tf()