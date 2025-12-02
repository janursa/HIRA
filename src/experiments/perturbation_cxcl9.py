#!/usr/bin/env python
"""
Perturbation Analysis (CXCL9) - Post-run visualization script

This script performs post-run analysis for perturbation experiments (CXCL9, OP, etc.),
generating various visualizations including:
- Coverage/overlap with aging TFs
- Overview heatmap
- Age-stratified TF analysis
- Donor-level perturbation effects
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
    palette_treatment,
    surrogate_names
)
from ciim.src.feature_association.config import get_config
from ciim.src.feature_association.helper import retrieve_sig_stats
from ciim.src.feature_association.plots import (
    heamap_plot_minor_cell_types,
    plot_overlap,
    plot_analysis_and_centrality,
    plot_donor_level_perturbation_effect
)
from ciim.src.utils.util import retrieve_net_consensus

warnings.filterwarnings("ignore")

# Set matplotlib defaults
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "Arial"


def load_stats(dataset, data_type, feature_type):
    """Load perturbation statistics from saved CSV file."""
    cfg = get_config(dataset)
    stats_path = f'{SAVE_DIR}/stats/stats_{dataset}_{data_type}_{feature_type}_{cfg.test_type}.csv'
    
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    
    stats = pd.read_csv(stats_path)
    return stats, cfg


def plot_coverage_overlap(stats_sig, dataset, output_dir):
    """Plot overlap between perturbation and aging TFs."""
    print("Generating coverage/overlap plots...")
    
    aging_stats_sig = retrieve_sig_stats(type='bulk').drop_duplicates(subset=["cell_type", "tf"])
    aging_stats_sig = aging_stats_sig[["tf", "cell_type", "slope"]]
    
    # Dataset-specific configurations
    if dataset == 'CXCL9':
        treatments = ['RPMI', 'LPS']
    elif dataset == 'op':
        treatments = ['Dimethyl Sulfoxide']  # Single control
    elif dataset == 'parsebioscience':
        treatments = ['PBS']
    else:
        print(f"  Warning: Unknown dataset {dataset}, skipping overlap plots")
        return
    
    for treatment in treatments:
        if dataset == 'CXCL9':
            condition_filter = f'Ruxolitinib (ctr: {treatment})'
        else:
            condition_filter = stats_sig['condition'].unique()[0]  # Use first condition
        
        df_sub = stats_sig[stats_sig['condition'] == condition_filter]
        
        if len(df_sub) == 0:
            print(f"  Warning: No data for condition {condition_filter}")
            continue
        
        plot_overlap(
            df_sub[['tf', 'cell_type', 'slope_condition']], 
            aging_stats_sig[['tf', 'cell_type', 'slope']], 
            col='cell_type', 
            how='left', 
            agreement='opposite', 
            legend=True, 
            figsize=(1.5, 2), 
            legend_loc=(1, 0.5)
        )
        plt.title('')
        plt.legend().remove()
        
        name = f'{dataset}_{treatment}'
        output_path = os.path.join(output_dir, f'drug_aging_overlap_{name}.png')
        plt.savefig(output_path, bbox_inches="tight", dpi=300, transparent=True)
        plt.close()
        print(f"  Saved: {output_path}")


def plot_overview_heatmap(stats, output_dir):
    """Generate overview heatmap of perturbation effects."""
    print("Generating overview heatmap...")
    
    # Get unique conditions
    conditions = stats['condition'].unique()
    print(f"  Found {len(conditions)} condition(s): {', '.join(conditions)}")
    
    # Plot one heatmap per condition (or just first if many)
    conditions_to_plot = conditions[:1] if len(conditions) > 2 else conditions
    
    for condition in conditions_to_plot:
        stats_cond = stats[stats['condition'] == condition].copy()
        
        stats_cond['cell_type'] = pd.Categorical(stats_cond['cell_type'], categories=cell_types, ordered=True)
        stats_cond['major_cell_type'] = stats_cond['cell_type'].astype(
            CategoricalDtype(categories=['CD4T', 'CD8T', 'NK', 'MONO', 'B'], ordered=True)
        )
        
        heamap_plot_minor_cell_types(
            stats_cond, 
            slope_col='slope_condition', 
            palette=palette_treatment, 
            figsize=(2, 3), 
            sig_dots_y_offset=3, 
            annotate_x_ticks=False, 
            map_names={'cell_type': 'Sub type', 'major_cell_type': 'Cell type'},
            dendrogram_visible=False, 
            show_legend=False
        )
        
        # Clean condition name for filename
        condition_clean = condition.replace('(', '').replace(')', '').replace(':', '').replace(' ', '_')
        output_path = os.path.join(output_dir, f'overview_perturbation_{condition_clean}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
        plt.close()
        print(f"  Saved: {output_path}")


def plot_aging_perturbation_comparison(stats, dataset, target_cell_types, top_aging_tfs, output_dir):
    """Plot comparison between aging and perturbation TFs."""
    print("Generating aging vs perturbation comparison...")
    
    palette_all = {**palette_trend, **palette_treatment}
    stats_df = stats.copy()
    
    for cell_type in target_cell_types:
        print(f"  Processing cell type: {cell_type}")
        
        # Aging TFs
        stats_aging = retrieve_sig_stats(type='bulk').drop_duplicates(subset=['tf', 'cell_type'])
        aging_df = stats_aging[stats_aging['cell_type'] == cell_type]
        aging_df['analysis'] = 'Age-associated'
        
        # Perturbation TFs
        comparisons = stats_df['condition'].unique()
        stats_store = []
        for comparison in comparisons:
            stats_d = stats_df[
                (stats_df['condition'] == comparison) & 
                (stats_df['cell_type'] == cell_type)
            ].copy()
            stats_d['is_significant'] = stats_d['p_value_adj'] < 0.05
            stats_d['analysis'] = comparison
            stats_d['trend'] = [
                'Increase after treatment' if x > 0 else 'Decrease after treatment' 
                for x in stats_d['slope_condition']
            ]
            stats_store.append(stats_d)
        
        if not stats_store:
            print(f"    Warning: No data for {cell_type}")
            continue
            
        stats_condition = pd.concat(stats_store)
        
        # Subset to overlapping TFs
        drug_tfs = stats_condition['tf'].unique()
        aging_tfs = aging_df['tf'].unique()
        common_tfs = np.intersect1d(drug_tfs, aging_tfs)
        
        if len(common_tfs) == 0:
            print(f"    Warning: No common TFs for {cell_type}")
            continue
        
        # Merge
        df = pd.concat([aging_df, stats_condition])
        df = df[df['tf'].isin(common_tfs)]
        
        df['analysis'] = pd.Categorical(
            df['analysis'], 
            categories=['Age-associated'] + list(comparisons), 
            ordered=True
        )
        
        # Add centrality information
        net = retrieve_net_consensus(datasets_all, cell_type)
        c_df = net.groupby('source').size().reset_index(name='degree')
        df = df.merge(c_df, left_on='tf', right_on='source', how='left')
        
        top_tfs = df.drop_duplicates(subset=['cell_type', 'tf']).sort_values(
            'degree', ascending=False
        ).head(top_aging_tfs)['tf'].unique()
        df = df[df['tf'].isin(top_tfs)].sort_values('degree', ascending=False)
        df['degree'] = df['degree'].div(df['degree'].max())  # Normalize degree
        
        plot_analysis_and_centrality(
            df, 
            all_groups=comparisons, 
            palette_all=palette_all, 
            figsize=(2.5, 4), 
            ax2_margins={'x': 0.2, 'y': 0.02}, 
            hide_ylabels=False, 
            plot_centrality=True, 
            show_legend=True
        )
        plt.suptitle(f'{cell_type}', y=.98, weight='bold')
        plt.tight_layout()
        plt.title('')
        
        output_path = os.path.join(output_dir, f'aging_drug_trend_{cell_type}_{dataset}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
        plt.close()
        print(f"    Saved: {output_path}")


def plot_donor_level_effects(stats, dataset, target_cell_types, output_dir):
    """Plot donor-level perturbation effects for case TFs."""
    print("Generating donor-level perturbation plots...")
    
    # Dataset-specific configurations
    if dataset == 'CXCL9':
        comparisons = ['Ruxolitinib (ctr: RPMI)', 'Ruxolitinib (ctr: LPS)']
        bbox_to_anchor = (1, 1)
    elif dataset == 'op':
        comparisons = ['Ruxolitinib']
        bbox_to_anchor = (1, 1)
    elif dataset == 'parsebioscience':
        comparisons = ['IL-10']
        bbox_to_anchor = (1, 1)
    else:
        print(f"  Warning: Unknown dataset {dataset}, skipping donor-level plots")
        return
    
    for comparison in comparisons:
        for cell_type in target_cell_types:
            # Cell type specific TFs
            if cell_type == 'CD4T':
                case_tfs = ['KLF6', 'PRDM1']
            elif cell_type == 'CD8T':
                case_tfs = ['LEF1', 'ZEB2']
            else:
                print(f"    Warning: No case TFs defined for {cell_type}")
                continue
            
            fig, axes = plt.subplots(1, 2, figsize=(2.5, 1.2), sharex=False, sharey=False)
            
            for i, case_tf in enumerate(case_tfs):
                stats_case = stats[
                    (stats['condition'] == comparison) & 
                    (stats['tf'] == case_tf) & 
                    (stats['cell_type'] == cell_type)
                ].copy()
                
                if len(stats_case) == 0:
                    print(f"    Warning: No data for {case_tf} in {cell_type}")
                    continue
                
                ax = axes[i]
                
                # Map comparison to control and treatment labels
                if comparison == 'Ruxolitinib (ctr: LPS)':
                    ctr = '24 h LPS'
                    treatment = '24 h LPS + ruxolitinib'
                elif comparison == 'Ruxolitinib (ctr: RPMI)':
                    ctr = '24 h RPMI'
                    treatment = '24 h RPMI + ruxolitinib'
                elif comparison == 'Ruxolitinib':
                    ctr = 'Dimethyl Sulfoxide'
                    treatment = 'Ruxolitinib'
                elif comparison == 'IL-10':
                    ctr = 'PBS'
                    treatment = 'IL-10'
                else:
                    print(f"    Warning: Unknown comparison {comparison}")
                    continue
                
                p_value_adj = stats_case['p_value_adj'].values[0]
                plot_donor_level_perturbation_effect(
                    case_tf, ctr, treatment, cell_type, p_value_adj, 
                    dataset=dataset, ax=ax, bbox_to_anchor=bbox_to_anchor
                )
                
                if i == 0:
                    ax.get_legend().remove()
                else:
                    ax.set_ylabel('', fontsize=10)
                    ax.spines[['left']].set_visible(False)
            
            output_path = os.path.join(
                output_dir, 
                f'donor_level_perturbation_effect_{comparison}_{cell_type}.png'
            )
            plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
            plt.close()
            print(f"    Saved: {output_path}")


def plot_pathway_analysis(stats, dataset, output_dir):
    """Save significant TFs for pathway enrichment analysis."""
    print("Saving significant TFs for pathway analysis...")
    
    pathway_data_store = []
    
    for ctr in stats['ctrl'].unique():
        for treatment in stats['condition'].unique():
            stats_sig = stats[
                (stats['ctrl'] == ctr) & 
                (stats['condition'] == treatment) & 
                (stats['p_value_adj'] < 0.05)
            ]
            
            if len(stats_sig) == 0:
                continue
            
            stats_sig_copy = stats_sig.copy()
            stats_sig_copy['ctrl'] = ctr
            stats_sig_copy['condition'] = treatment
            pathway_data_store.append(stats_sig_copy)
    
    if not pathway_data_store:
        print("  Warning: No significant TFs found for pathway analysis")
        return
    
    pathway_data_all = pd.concat(pathway_data_store)
    
    # Save significant TFs for pathway analysis
    output_path = os.path.join(output_dir, f'{dataset}_sig_tfs_for_pathway.csv')
    pathway_data_all.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    
    return pathway_data_all


def main():
    parser = argparse.ArgumentParser(
        description='Perturbation Analysis - Post-run visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (e.g., CXCL9, op, parsebioscience)'
    )
    parser.add_argument(
        '--data-type',
        type=str,
        default='sc',
        help='Data type: bulk or sc (default: sc)'
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
        default=['CD4T'],
        help='Cell types to analyze (default: CD4T)'
    )
    parser.add_argument(
        '--sig-threshold',
        type=float,
        default=0.05,
        help='Significance threshold for p-values (default: 0.05)'
    )
    parser.add_argument(
        '--top-aging-tfs',
        type=int,
        default=15,
        help='Number of top aging TFs to show (default: 15)'
    )
    parser.add_argument(
        '--skip-pathway',
        action='store_true',
        help='Skip pathway analysis'
    )
    parser.add_argument(
        '--skip-donor-level',
        action='store_true',
        help='Skip donor-level plots'
    )
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else PLOTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("Perturbation Analysis - Post-run Visualization")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Data type: {args.data_type}")
    print(f"Feature type: {args.feature_type}")
    print(f"Output directory: {output_dir}")
    print(f"Cell types: {', '.join(args.cell_types)}")
    print(f"Significance threshold: {args.sig_threshold}")
    print("="*60)
    
    # Load statistics
    print("\nLoading statistics...")
    stats, cfg = load_stats(args.dataset, args.data_type, args.feature_type)
    stats_sig = stats[stats['p_value_adj'] < args.sig_threshold].copy()
    
    print(f"Total significant TFs per cell type:")
    if len(stats_sig) > 0:
        print(stats_sig.groupby(['cell_type'])['tf'].nunique())
    else:
        print("  No significant TFs found!")
    print()
    
    # 1. Coverage/overlap plots
    if len(stats_sig) > 0:
        plot_coverage_overlap(stats_sig, args.dataset, output_dir)
    
    # 2. Overview heatmap
    stats['cell_type'] = pd.Categorical(stats['cell_type'], categories=cell_types, ordered=True)
    stats['trend'] = [
        'Increase after treatment' if x > 0 else 'Decrease after treatment' 
        for x in stats['slope_condition']
    ]
    plot_overview_heatmap(stats, output_dir)
    
    # 3. Aging vs perturbation comparison
    plot_aging_perturbation_comparison(
        stats, 
        args.dataset, 
        args.cell_types, 
        args.top_aging_tfs, 
        output_dir
    )
    
    # 4. Donor-level effects (optional)
    if not args.skip_donor_level:
        plot_donor_level_effects(stats, args.dataset, args.cell_types, output_dir)
    
    # 5. Pathway analysis (optional)
    if not args.skip_pathway:
        plot_pathway_analysis(stats, args.dataset, output_dir)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"All plots saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
