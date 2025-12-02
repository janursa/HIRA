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


def plot_aging_experiment_heatmap(stats_sig, dataset, output_dir):
    """
    Plot heatmap comparing TF activity/directions between aging and experimental conditions.
    
    Creates a heatmap where:
    - Rows: TFs (unlabeled due to high number)
    - First column: Natural aging slope values
    - Subsequent columns: Experimental condition slope values (one per dataset/condition)
    """
    import seaborn as sns
    from matplotlib.patches import Rectangle
        
    # Load aging stats
    aging_stats_sig = retrieve_sig_stats(type='bulk').drop_duplicates(subset=["cell_type", "tf"])
    
    cfg = get_config(dataset)
    target_treatments = cfg.target_treatments
    assert target_treatments is not None, "Target treatments configuration is missing."
    
    for cell_type in stats_sig['cell_type'].unique():
        print(f"  Processing cell type: {cell_type}")
        
        for treatment in target_treatments:
            # Get experiment data
            exp_df = stats_sig[
                (stats_sig['condition'] == treatment) & 
                (stats_sig['cell_type'] == cell_type)
            ][['tf', 'slope_condition']].copy()
            
            # Get aging data for same cell type
            aging_df = aging_stats_sig[
                aging_stats_sig['cell_type'] == cell_type
            ][['tf', 'slope']].copy()
            
            # Merge on TFs that are in the experiment (left join)
            merged = exp_df.merge(aging_df, on='tf', how='left')
            
            if len(merged) == 0:
                print(f"    Warning: No data for {treatment} in {cell_type}")
                continue
            
            # Sort by experiment slope for better visualization
            merged = merged.sort_values('slope_condition', ascending=False)
            
            # Prepare data matrix
            data_matrix = merged[['slope', 'slope_condition']].values
            
            # Create heatmap
            n_tfs = len(merged)
            figsize = (2.5, max(6, n_tfs * 0.03))  
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot heatmap with diverging colormap
            im = ax.imshow(
                data_matrix,
                aspect='auto',
                # cmap='RdBu_r',
                vmin=-max(abs(data_matrix.min()), abs(data_matrix.max())),
                vmax=max(abs(data_matrix.min()), abs(data_matrix.max())),
                interpolation='nearest'
            )
            
            # Add vertical line to separate aging from experiment
            ax.axvline(0.5, color='black', linewidth=2)
            
            # Set column labels
            # treatment_short = treatment.replace('Older (55-65y)', 'Older 55-65y').replace(' ', '\n')
            ax.set_xticks([0, 1])
            # ax.set_xticklabels(['Natural\naging', treatment_short], fontsize=9)
            ax.set_xlabel('')
            
            # Remove y-axis labels (too many TFs)
            ax.set_yticks([])
            ax.set_ylabel(f'TFs (n={n_tfs})', fontsize=9)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Slope (effect size)', rotation=270, labelpad=15, fontsize=9)
            cbar.ax.tick_params(labelsize=8)
            
            # Title
            ax.set_title(f'{cell_type}', 
                        fontsize=10, weight='bold', pad=10)
            
            # Add agreement statistics as text
            same_direction = (np.sign(merged['slope']) == np.sign(merged['slope_condition'])).sum()
            total_with_aging = merged['slope'].notna().sum()
            if total_with_aging > 0:
                agreement_pct = (same_direction / total_with_aging) * 100
                ax.text(
                    0.02, 0.98, 
                    f'Same direction:\n{same_direction}/{total_with_aging} ({agreement_pct:.0f}%)',
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                )
            
            # Save
            name = f'{dataset}_{treatment}_{cell_type}'
            output_path = os.path.join(output_dir, f'aging_experiment_heatmap_{name}.png')
            output_path = output_path.replace(' ', '_').replace('(', '_').replace(')', '_').replace(':', '_')
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
            plt.close()
            print(f"    Saved: {output_path}")


def plot_coverage_overlap(stats_sig, dataset, output_dir):
    """Plot overlap between perturbation and aging TFs."""
    print("Generating coverage/overlap plots...")
    
    aging_stats_sig = retrieve_sig_stats(type='bulk').drop_duplicates(subset=["cell_type", "tf"])
    aging_stats_sig = aging_stats_sig[["tf", "cell_type", "slope"]]
    
    cfg = get_config(dataset)
    target_treatments = cfg.target_treatments
    assert target_treatments is not None, "Target treatments configuration is missing."
    
    for cell_type in stats_sig['cell_type'].unique():
        print(f"  Processing cell type: {cell_type}")
        for treatment in target_treatments:
            
            df_sub = stats_sig[(stats_sig['condition'] == treatment) & (stats_sig['cell_type'] == cell_type)]
            aging_stats_sig_sub = aging_stats_sig[aging_stats_sig['cell_type'] == cell_type]
            
            if len(df_sub) == 0:
                raise ValueError(f"No data for condition {treatment}")
            
            # Calculate overlap statistics
            merged = df_sub[['tf', 'slope_condition']].merge(
                aging_stats_sig_sub[['tf', 'slope']], 
                on='tf', 
                how='inner'
            )
            
            if len(merged) > 0:
                same_direction = (np.sign(merged['slope_condition']) == np.sign(merged['slope'])).sum()
                opposite_direction = (np.sign(merged['slope_condition']) == -np.sign(merged['slope'])).sum()
                total_overlap = len(merged)
                
                print(f"    Overlap with aging TFs: {total_overlap}")
                print(f"      Same direction: {same_direction} ({same_direction/total_overlap*100:.1f}%)")
                print(f"      Opposite direction: {opposite_direction} ({opposite_direction/total_overlap*100:.1f}%)")
            
            plot_overlap(
                df_sub[['tf', 'cell_type', 'slope_condition']], 
                aging_stats_sig_sub[['tf', 'cell_type', 'slope']], 
                col='cell_type', 
                how='left', 
                agreement=cfg.comparison_mode, 
                legend=True, 
                figsize=(1.5, 2), 
                legend_loc=(1, 0.5)
            )
            plt.title('')
            plt.legend().remove()
            
            name = f'{dataset}_{treatment}_{cell_type}'
            output_path = os.path.join(output_dir, f'drug_aging_overlap_{name}.png')
            output_path = output_path.replace(' ', '_').replace('(', '_').replace(')', '_').replace(':', '_')
            plt.savefig(output_path, bbox_inches="tight", dpi=300, transparent=True)
            plt.close()
            print(f"  Saved: {output_path}")


def plot_overview_heatmap(stats, dataset, output_dir):
    """Generate overview heatmap of perturbation effects."""
    print("Generating overview heatmap...")
    
    # Get unique conditions
    conditions = stats['condition'].unique()
    print(f"  Found {len(conditions)} condition(s): {', '.join(conditions)}")
    
    # Plot one heatmap per condition (or just first if many)
    target_treatments = get_config(dataset).target_treatments
    assert target_treatments is not None, "Target treatments configuration is missing."
    
    for condition in target_treatments:
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
        condition_clean = condition.replace('(', '').replace(')', '').replace(':', '').replace(' ', '_').replace(':', '_')
        output_path = os.path.join(output_dir, f'overview_perturbation_{condition_clean}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
        plt.close()
        print(f"  Saved: {output_path}")


def plot_aging_perturbation_comparison(stats, dataset, target_cell_types, top_aging_tfs, output_dir):
    """Plot comparison between aging and perturbation TFs."""
    print("Generating aging vs perturbation comparison...")
    
    palette_all = {**palette_trend, **palette_treatment}
    stats_df = stats.copy()

    target_treatments = get_config(dataset).target_treatments
    assert target_treatments is not None, "Target treatments configuration is missing."
    
    for cell_type in target_cell_types:
        print(f"  Processing cell type: {cell_type}")
        
        # Aging TFs
        stats_aging = retrieve_sig_stats(type='bulk').drop_duplicates(subset=['tf', 'cell_type'])
        aging_df = stats_aging[stats_aging['cell_type'] == cell_type]
        aging_df['analysis'] = 'Age-associated'
        
        # Perturbation TFs
        # comparisons = stats_df['condition'].unique()
        comparisons = target_treatments
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
        output_path = output_path.replace(' ', '_').replace('(', '_').replace(')', '_').replace(':', '_')
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
            output_path = output_path.replace(' ', '_').replace('(', '_').replace(')', '_').replace(':', '_')
            plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
            plt.close()
            print(f"    Saved: {output_path}")


def plot_pathway_analysis(stats, dataset, output_dir):
    """Generate pathway enrichment analysis using GSEA and KDE plots."""
    print("Generating pathway analysis...")
    
    from ciim.src.pathway_analysis.util import gsea_func, pathway_kde_func
    from ciim.src.pathway_analysis.plots import plot_pathway_kde
    from ciim.src.feature_association.plots import dotplot_category_color
    
    # Prepare stats for pathway analysis
    stats_for_pathway = stats.copy()
    stats_for_pathway = stats_for_pathway[stats_for_pathway['p_value_adj'] < 0.05]
    
    if len(stats_for_pathway) == 0:
        print("  Warning: No significant TFs found for pathway analysis")
        return None
    
    # Save significant TFs
    output_csv = os.path.join(output_dir, f'{dataset}_sig_tfs_for_pathway.csv')
    stats_for_pathway.to_csv(output_csv, index=False)
    print(f"  Saved significant TFs: {output_csv}")
    
    # 1. GSEA enrichment analysis
    print("\n  Running GSEA enrichment analysis...")
    try:
        pathway_scores = gsea_func(
            stats_for_pathway,
            pvalue_col='p_value_adj',
            gene_sets=['MSigDB_Hallmark_2020'],
            feature_col='tf'
        )
        
        if pathway_scores is not None and len(pathway_scores) > 0:
            print(f"    Found {len(pathway_scores)} significant pathway enrichments")
            
            # Plot GSEA dotplot
            n_terms = pathway_scores['Term'].nunique()
            cell_types = pathway_scores['cell_type'].unique()
            
            fig, ax = plt.subplots(
                1, 1, 
                figsize=(len(cell_types) * 0.12 + 1, 1 + 0.15 * n_terms), 
                sharey=True, 
                sharex=True
            )
            
            pathway_scores['cell_type'] = pd.Categorical(
                pathway_scores['cell_type'], 
                categories=cell_types, 
                ordered=True
            )
            
            dotplot_category_color(
                pathway_scores,
                ax,
                color_col='trend',
                size_col='neg_log10_adj_pval',
                x='cell_type',
                y='Term',
                palette=palette_treatment,
                sizes=(50, 250),
                show_color_legend=True,
                show_size_legend=True,
                size_legend_title='Significance',
                color_legend_title='Pathway activity',
                size_legend_loc=(1.2, 0.35),
                color_legend_loc=(1.02, 0.75),
                y_label='',
                alpha=0.5
            )
            
            output_path = os.path.join(output_dir, f'{dataset}_pathway_gsea.png')
            plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
            plt.close()
            print(f"    Saved GSEA plot: {output_path}")
        else:
            print("    No significant pathway enrichments found")
            pathway_scores = None
    
    except Exception as e:
        print(f"    Warning: GSEA analysis failed: {e}")
        pathway_scores = None
    
    # 2. KDE pathway analysis
    print("\n  Running KDE pathway analysis...")
    try:
        # Add 'slope' column required by pathway_kde_func
        stats_for_pathway_kde = stats_for_pathway.copy()
        stats_for_pathway_kde['slope'] = stats_for_pathway_kde['slope_condition']
        
        res_pathway_kde = pathway_kde_func(
            stats_for_pathway_kde,
            pathway='hallmark',
            sets=None,
            test='wilcoxon',
            min_genes=10,
            fdr_method='fdr_bh',
            feature_col='tf'
        )
        
        if len(res_pathway_kde) > 0:
            print(f"    Found {len(res_pathway_kde)} significant pathways in KDE analysis")
            
            # Plot KDE - use the modified dataframe with 'slope' column
            fig, skipped = plot_pathway_kde(
                df_condition=stats_for_pathway_kde,
                res_cond=res_pathway_kde,
                df_aging=None,
                res_aging=None,
                sets=None,
                cell_types=None,
                max_height=0.2,
                row_spacing=0.4,
                cell_spacing=1,
                min_genes=10,
                feature_col='tf'
            )
            
            output_path = os.path.join(output_dir, f'{dataset}_pathway_kde.png')
            plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
            plt.close()
            print(f"    Saved KDE plot: {output_path}")
            
            if skipped:
                print(f"    Note: Some pathways were skipped due to insufficient genes")
        else:
            print("    No significant pathways found in KDE analysis")
    
    except Exception as e:
        print(f"    Warning: KDE analysis failed: {e}")
        res_pathway_kde = None
    
    return pathway_scores


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
    
    # 1b. Aging vs experiment heatmap
    if len(stats_sig) > 0:
        plot_aging_experiment_heatmap(stats_sig, args.dataset, output_dir)
    
    # 2. Overview heatmap
    stats['cell_type'] = pd.Categorical(stats['cell_type'], categories=cell_types, ordered=True)
    stats['trend'] = [
        'Increase after treatment' if x > 0 else 'Decrease after treatment' 
        for x in stats['slope_condition']
    ]
    plot_overview_heatmap(stats, args.dataset, output_dir)
    
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
