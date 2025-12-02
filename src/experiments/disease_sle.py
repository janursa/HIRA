#!/usr/bin/env python
"""
SLE Disease Analysis - Post-run visualization script

This script performs post-run analysis for SLE (Systemic Lupus Erythematosus) disease data,
generating various visualizations including:
- Overview heatmap of cell types
- Overlap analysis with aging TFs
- Age-stratified analysis
- Case TF trends
- Pathway analysis
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
    surrogate_names,
    mapping_minor_2_major
)
from ciim.src.feature_association.config import get_config
from ciim.src.feature_association.helper import retrieve_sig_stats
from ciim.src.feature_association.plots import (
    heamap_plot_minor_cell_types,
    plot_overlap,
    plot_analysis_and_centrality
)
from ciim.src.feature_association.disease import plot_healthy_disease_trend
from ciim.src.utils.util import retrieve_net_consensus
from ciim.src.pathway_analysis.util import pathway_kde_func
from ciim.src.pathway_analysis.plots import plot_pathway_kde

warnings.filterwarnings("ignore")

# Set matplotlib defaults
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "Arial"


def load_stats(dataset, data_type, feature_type):
    """Load statistics from saved CSV file."""
    cfg = get_config(dataset)
    stats_path = f'{SAVE_DIR}/stats/stats_{dataset}_{data_type}_{feature_type}_{cfg.test_type}.csv'
    
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    
    stats = pd.read_csv(stats_path)
    stats = stats[(~stats['slope_condition'].isna())].copy()
    return stats, cfg


def plot_overview_heatmap(stats_sig, output_dir):
    """Generate overview heatmap of minor cell types."""
    print("Generating overview heatmap...")
    
    stats_sig['cell_type'] = pd.Categorical(stats_sig['cell_type'], categories=cell_types, ordered=True)
    stats_sig['major_cell_type'] = stats_sig['cell_type'].astype(
        CategoricalDtype(categories=['CD4T', 'CD8T', 'NK', 'MONO', 'B'], ordered=True)
    )
    
    stats_sig_all = stats_sig[stats_sig['age_group'] == 'Both age groups']
    
    heamap_plot_minor_cell_types(
        stats_sig_all, 
        slope_col='slope_condition', 
        palette=palette_disease_effect, 
        figsize=(2, 3), 
        sig_dots_y_offset=3, 
        annotate_x_ticks=False, 
        map_names={'cell_type': 'Sub type', 'major_cell_type': 'Cell type'},
        dendrogram_visible=False, 
        show_legend=False
    )
    
    output_path = os.path.join(output_dir, 'overview_sle.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_aging_overlap(stats_sig, included_cell_types, disease_name, data_type, agreement, output_dir):
    """Plot overlap between disease and aging TFs."""
    print("Generating aging overlap plot...")
    
    aging_stats_sig = retrieve_sig_stats(type='bulk').drop_duplicates(subset=['cell_type', 'tf'])
    aging_stats_sig = aging_stats_sig[['tf', 'cell_type', 'slope']]
    
    stats_sig_filtered = stats_sig[stats_sig['cell_type'].isin(included_cell_types)].copy()
    stats_sig_filtered['cell_type'] = pd.Categorical(
        stats_sig_filtered['cell_type'], 
        categories=included_cell_types, 
        ordered=True
    )
    
    plot_overlap(
        stats_sig_filtered[['tf', 'cell_type', 'slope_condition']], 
        aging_stats_sig[['tf', 'cell_type', 'slope']], 
        col='cell_type', 
        how='left', 
        agreement=agreement, 
        legend=True, 
        figsize=(1.5, 2), 
        legend_loc=(1, 0.5)
    )
    
    output_path = os.path.join(output_dir, f'disease_aging_overlap_{disease_name}_{data_type}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_age_stratified_analysis(stats, stats_sig, disease_name, target_cell_types, top_aging_tfs, output_dir):
    """Plot age-stratified analysis comparing disease and aging TFs."""
    print("Generating age-stratified analysis...")
    
    palette_all = {**palette_trend, **palette_disease_effect}
    age_groups = stats['age_group'].unique()
    
    aging_stats_sig = retrieve_sig_stats(type='bulk').drop_duplicates(subset=['cell_type', 'tf'])
    aging_stats_sig = aging_stats_sig[['tf', 'cell_type', 'slope']]
    
    stats['dataset'] = stats['dataset'].apply(lambda name: surrogate_names.get(name, name))
    
    for cell_type in target_cell_types:
        print(f"  Processing cell type: {cell_type}")
        
        # --- Aging TFs ---
        aging_df = aging_stats_sig[aging_stats_sig['cell_type'] == cell_type]
        net = retrieve_net_consensus(datasets_all, cell_type)
        c_df = net.groupby('source').size().reset_index(name='degree')
        aging_df = aging_df.merge(c_df, left_on='tf', right_on='source', how='left')[['tf', 'slope', 'degree']]
        aging_df['degree'] = aging_df['degree'].div(aging_df['degree'].max())  # Normalize degree
        aging_df['trend'] = ['Increase in aging' if x > 0 else 'Decrease in aging' for x in aging_df['slope']]
        aging_df['analysis'] = 'Age-associated'
        
        # --- Disease TFs ---
        stats_store = []
        for age_group in age_groups:
            stats_d = stats[
                (stats['age_group'] == age_group) &
                (stats['cell_type'] == cell_type) &
                (~stats['slope_condition'].isna())
            ].copy()
            stats_d = stats_d[['tf', 'cell_type', 'slope_condition', 'p_value_adj']].drop_duplicates()
            stats_d['analysis'] = age_group
            stats_d['trend'] = ['Increase in disease' if x > 0 else 'Decrease in disease' for x in stats_d['slope_condition']]
            stats_store.append(stats_d)
        
        stats_condition = pd.concat(stats_store)
        stats_condition['analysis'] = stats_condition['analysis'].astype(
            pd.CategoricalDtype(categories=age_groups, ordered=True)
        )
        
        # --- Restrict to overlapping TFs ---
        common_tfs = set(aging_df['tf']) & set(stats_condition['tf'])
        aging_df = aging_df[aging_df['tf'].isin(common_tfs)]
        
        # Order TFs by degree (descending: top = most central)
        top_tfs = aging_df.sort_values('degree', ascending=False).head(top_aging_tfs)['tf'].unique()
        tf_order = list(top_tfs)  # already in correct top-to-bottom order
        
        # --- Combine and filter ---
        aging_df = aging_df[aging_df['tf'].isin(top_tfs)]
        stats_combined = pd.concat([aging_df, stats_condition])
        df = stats_combined[stats_combined['tf'].isin(top_tfs)].copy()
        
        # Set TF categorical order
        df['tf'] = pd.Categorical(df['tf'], categories=tf_order, ordered=True)
        
        plot_analysis_and_centrality(
            df, 
            all_groups=age_groups,  
            figsize=(2.1, 4), 
            palette_all=palette_all, 
            plot_centrality=False, 
            show_legend=False
        )
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'aging_{disease_name}_overlap_{cell_type}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
        plt.close()
        print(f"    Saved: {output_path}")


def plot_case_tf_trends(dataset, data_type, case_tfs, cell_type, output_dir):
    """Plot healthy vs disease trends for specific TFs."""
    print("Generating case TF trend plots...")
    
    condition_col = 'condition' if dataset == 'SLE_European' else 'Max_WHO_Group'
    
    for i, case_tf in enumerate(case_tfs):
        fig, ax = plt.subplots(1, 1, figsize=(2, .6), sharey=False, sharex=False)
        
        plot_healthy_disease_trend(
            dataset=dataset, 
            data_type=data_type, 
            cell_type=cell_type, 
            case_tf=case_tf, 
            condition_col=condition_col, 
            ax=ax
        )
        
        ax.set_title(f'{case_tf}', fontsize=10, pad=10, weight='bold')
        if i == 0:
            ax.set_xlabel('')
        
        output_path = os.path.join(output_dir, f'healthy_disease_trend_{case_tf}_{cell_type}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
        plt.close()
        print(f"  Saved: {output_path}")


def plot_pathway_analysis(dataset, data_type, feature_type, pathway_cell_types, output_dir):
    """Perform pathway analysis comparing aging and disease."""
    print("Generating pathway analysis...")
    
    cfg = get_config(dataset)
    
    # Determine feature column based on feature type
    feature_col = 'target' if feature_type == 'gene_expression' else 'tf'
    
    # Load stats
    stats_path = f'{SAVE_DIR}/stats/stats_{dataset}_{data_type}_{feature_type}_{cfg.test_type}.csv'
    
    if not os.path.exists(stats_path):
        print(f"  Warning: Stats not found at {stats_path}")
        print("  Skipping pathway analysis.")
        return
    
    stats = pd.read_csv(stats_path)
    
    # Get aging stats with the same feature type
    aging_stats_sig = retrieve_sig_stats(
        type=data_type, 
        race="both", 
        feature_type=feature_type
    ).drop_duplicates(subset=["cell_type", feature_col])
    aging_stats_sig = aging_stats_sig[[feature_col, "cell_type", "slope", 'p_value_adj']]
    
    # Filter significant disease stats
    stats_sig = stats[(stats['p_value_adj'] < 0.05)]
    stats_sig = stats_sig[
        (~stats_sig['slope_condition'].isna()) & 
        (stats_sig['age_group'] == 'Younger than 50')
    ]
    
    # Pathway enrichment with appropriate feature column
    res_aging = pathway_kde_func(aging_stats_sig, min_genes=10, feature_col=feature_col)
    res_aging_sig = res_aging[res_aging['p_adj'] < 0.05]
    
    stats_sig['slope'] = stats_sig['slope_condition']
    res_condition = pathway_kde_func(stats_sig, min_genes=10, feature_col=feature_col)
    res_condition_sig = res_condition[res_condition['p_adj'] < 0.05]
    
    sets = np.concatenate([res_aging_sig['gene_set'].unique(), res_condition_sig['gene_set'].unique()])
    
    plot_pathway_kde(
        df_aging=aging_stats_sig,
        res_aging=res_aging,
        df_condition=stats_sig,
        res_cond=res_condition,
        cell_types=pathway_cell_types,
        sets=sets,
        min_genes=10,
        max_height=0.1,
        row_spacing=0.4,
        cell_spacing=1.5,
        feature_col=feature_col,
    )
    
    output_path = os.path.join(output_dir, 'sle_aging_pathway.png')
    plt.savefig(output_path, bbox_inches="tight", dpi=300, transparent=True)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='SLE Disease Analysis - Post-run visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--dataset',
        type=str,
        default='SLE_European',
        help='Dataset name (default: SLE_European)'
    )
    parser.add_argument(
        '--data-type',
        type=str,
        default='bulk',
        help='Data type: bulk or sc (default: bulk)'
    )
    parser.add_argument(
        '--feature-type',
        type=str,
        default='tf_activity',
        help='Feature type: tf_activity or gene_expression (default: tf_activity)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for plots (default: PLOTS_DIR from common.py)'
    )
    parser.add_argument(
        '--cell-types',
        type=str,
        nargs='+',
        default=['CD4T', 'CD8T'],
        help='Cell types to analyze (default: CD4T CD8T)'
    )
    parser.add_argument(
        '--case-tfs',
        type=str,
        nargs='+',
        default=['LEF1', 'ZEB2'],
        help='Case TFs for trend plotting (default: LEF1 ZEB2)'
    )
    parser.add_argument(
        '--case-cell-type',
        type=str,
        default='CD8T',
        help='Cell type for case TF analysis (default: CD8T)'
    )
    parser.add_argument(
        '--top-aging-tfs',
        type=int,
        default=15,
        help='Number of top aging TFs to show (default: 15)'
    )
    parser.add_argument(
        '--agreement',
        type=str,
        default='same',
        choices=['same', 'opposite'],
        help='Agreement type for overlap analysis (default: same)'
    )
    parser.add_argument(
        '--skip-pathway',
        action='store_true',
        help='Skip pathway analysis'
    )
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else PLOTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("SLE Disease Analysis - Post-run Visualization")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Data type: {args.data_type}")
    print(f"Feature type: {args.feature_type}")
    print(f"Output directory: {output_dir}")
    print(f"Cell types: {', '.join(args.cell_types)}")
    print("="*60)
    
    # Load statistics
    print("\nLoading statistics...")
    stats, cfg = load_stats(args.dataset, args.data_type, args.feature_type)
    stats_sig = stats[stats['p_value_adj'] < 0.05].copy()
    stats_sig['major_cell_type'] = stats_sig['cell_type'].apply(
        lambda x: mapping_minor_2_major.get(x, x)
    )
    
    print(f"Total significant TFs per cell type:")
    print(stats_sig.groupby(['cell_type'])['tf'].nunique())
    print()
    
    # 1. Overview heatmap
    plot_overview_heatmap(stats_sig, output_dir)
    
    # 2. Aging overlap
    plot_aging_overlap(
        stats_sig, 
        args.cell_types, 
        args.dataset, 
        args.data_type, 
        args.agreement,
        output_dir
    )
    
    # 3. Age-stratified analysis
    plot_age_stratified_analysis(
        stats, 
        stats_sig, 
        args.dataset,
        args.cell_types,
        args.top_aging_tfs,
        output_dir
    )
    
    # 4. Case TF trends
    plot_case_tf_trends(
        args.dataset,
        args.data_type,
        args.case_tfs,
        args.case_cell_type,
        output_dir
    )
    
    # 5. Pathway analysis (optional)
    if not args.skip_pathway:
        plot_pathway_analysis(
            args.dataset,
            args.data_type,
            args.feature_type,
            args.cell_types,
            output_dir
        )
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"All plots saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
