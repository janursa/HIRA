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
from hiara.src.config import (
    PLOTS_DIR, 
    CELL_TYPES, 
    DISCOVERY_COHORTS,
    palette_trend,
    surrogate_names
)
from hiara import retrieve_net_consensus
from hiara import palette_cell_types, palette_datasets, palette_trend_2, colors_blind
from hiara import retrieve_sig_stats, retrieve_stats
from hiara.src.feature_association.plots import plot_sig_tf_counts


warnings.filterwarnings("ignore")

# Set matplotlib defaults
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "Arial"

def plot_sig_genes_counts_hallmarks(data_type='bulk'):
    from hiara.src.config import PRIOR_DIR
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
    cell_types_unique = count_pivot['cell_type'].unique()
    genesets_unique = sorted(count_pivot[geneset_col].unique())
    for ct in cell_types_unique:
        ct_data = count_pivot[count_pivot['cell_type'] == ct]
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

def gsea_analysis():
    from hiara.src.pathway_analysis.util import get_genesets, pathway_kde_func, get_hallmark, gsea_func, wrapper_gsea

    wrapper_gsea(stats_sig)
    file_name = f"{PLOTS_DIR}/gsea_tf_activity.png"
    print(f"Saving figure to {file_name}")
    plt.savefig(file_name, bbox_inches='tight', dpi=200)
## Heatmap of sig TFs across cell types and cohorts
def plot_heatmap_overal(stats_aging):
    from hiara.src.feature_association.plots import plot_overall_heatmap

    stats_aging['cell_type'] = pd.Categorical(stats_aging['cell_type'], categories=CELL_TYPES, ordered=True)
    stats_aging['dataset'] = pd.Categorical(stats_aging['dataset'], categories=DISCOVERY_COHORTS, ordered=True)

    plot_overall_heatmap(stats_aging, 
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
def plot_central_features(stats_aging, cell_types):
    if True:
        # Parameters
        n_top = 10
        focus = 'source'  # can be 'source' or 'target'
        palette = palette_trend_2  # assume this dict is defined

        # Collect top TFs across cell types
        rows = []
        for ct in cell_types:
            stats_ct = stats_aging[stats_aging['cell_type'] == ct].drop_duplicates(subset='gene')[['gene', 'slope', 'trend', 'cell_type']]
            aging_genes = stats_ct['gene'].unique()
            net_ct = retrieve_net_consensus(cell_type=ct)
            net_ct = net_ct[net_ct[focus].isin(aging_genes)]
            centrality = net_ct[focus].value_counts().reset_index(name='degree')    
            max_degree = centrality['degree'].max()
            centrality['normalized_centrality'] = centrality['degree'] / max_degree if max_degree > 0 else 0 
            centrality = centrality.head(n_top) 
            
            # Merge with TF stats to get trend
            merged = centrality.merge(stats_ct, left_on=focus, right_on='gene', how='left')
            rows.append(merged)
        plot_df = pd.concat(rows, ignore_index=True)
        col_c = 'normalized_centrality'
        plot_df[focus] = plot_df[focus].astype(str)
        plot_df = plot_df.sort_values(by=['cell_type', col_c], ascending=[True, False])
        n_cols = len(cell_types)
        fig, axes = plt.subplots(1, n_cols, figsize=(1.5*n_cols, 2.1 ), sharex=False)

        if n_cols == 1:
            axes = [axes]
        i = 0
        for ax, ct in zip(axes, cell_types):
            df_ct = plot_df[plot_df['cell_type'] == ct]
            colors = df_ct['trend'].map(palette)
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
        fig.tight_layout()
        file_name = os.path.join(PLOTS_DIR, f'central_aging_{focus}.png')
        print(f"Saving figure to {file_name}")
        plt.savefig(file_name, bbox_inches='tight', dpi=300, transparent=True)
def plot_interaction_of_aging_TFs_between_cell_types(args):
    from grn_benchmark.src.exp_analysis.helper import plot_interactions, create_interaction_df

    stats_sig = retrieve_sig_stats(data_type=args.data_type).drop_duplicates(subset=['cell_type', 'gene'])
    df_dict = stats_sig.groupby(['cell_type'])['gene'].apply(list).to_dict()
    interaction_main_df = create_interaction_df(df_dict)
    aa = plot_interactions(interaction_main_df, min_subset_size=5, min_degree=1, color_map=palette_cell_types)
    file_name = f"{PLOTS_DIR}/interactions.png"
    print(f"Saving figure to {file_name}")
    plt.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')
    # plt.title(surrogate_names[race], pad=40, fontsize=10, fontweight='bold')

    from hiara.src.feature_association.plots import plot_features_vs_datasets
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
        from hiara.src.feature_association.plots import plot_joint_scatter
        ttypes = ['CD8T', 'CD4T']
        mask = interaction_main_df[ttypes].sum(axis=1)==len(ttypes)
        features = mask[mask].index
        # Get significant TFs for CD4T and CD8T cells
        df = stats_sig[stats_sig['gene'].isin(features)].drop_duplicates(subset=['cell_type', 'gene'])
        df['neg_log10_adj_pval'] = -np.log10(df['meta_p_adj'])

        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        plot_joint_scatter(df, vars=['CD4T', 'CD8T'], annotate=True, ax=ax)
        ax.margins(x=0.1, y=0.1)

def plot_case_tf(args):
    from hiara.src.feature_association.plots import plot_feature_values_all_datasets, plot_feature_values_per_datasets
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-type', type=str, required=True, help='Feature type to analyze')
    parser.add_argument('--data-type', type=str, default='bulk', required=False, help='Data type to analyze')
    parser.add_argument('--skip-pathway', action='store_true', help='Skip pathway analysis')

    args = parser.parse_args()
    feature_type = args.feature_type
    data_type = args.data_type
    skip_pathway = args.skip_pathway

    stats_aging = retrieve_stats(data_type=data_type, feature_type=feature_type)
    stats_aging_sig = retrieve_sig_stats(data_type=data_type, feature_type=feature_type)

    # - add sig signs
    stats_sig = retrieve_sig_stats(data_type, feature_type=feature_type)
    tuple_index = stats_sig.set_index(['cell_type', 'gene', 'dataset']).index

    stats_aging = stats_aging.set_index(['cell_type', 'gene', 'dataset'])
    stats_aging['is_significant'] = False
    stats_aging.loc[tuple_index, 'is_significant'] = True

    stats_aging = stats_aging.reset_index()[['gene', 'cell_type', 'dataset', 'slope', 'is_significant']].drop_duplicates()

    if feature_type == 'tf_activity':
        plot_heatmap_overal(stats_aging)
        plot_sig_tf_counts(args)
        plot_central_features(stats_aging_sig, cell_types=['CD4T', 'CD8T', 'NK', 'MONO'])
        plot_interaction_of_aging_TFs_between_cell_types(args)
        plot_case_tf(args)
        if not skip_pathway:
            gsea_analysis()

        # plot_sig_networks()
    elif feature_type == 'aging_hallmarks':
        plot_sig_genes_counts_hallmarks()
    else:
        raise ValueError('Unknown feature type')