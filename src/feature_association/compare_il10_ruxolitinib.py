#!/usr/bin/env python
"""
Compare IL10 (parsebioscience) to Ruxolitinib (op) - Directional Plot
This script compares the treatment effects of IL10 from parsebioscience dataset
to Ruxolitinib from the op dataset by creating a directional consistency scatter plot.
"""
import argparse
import os
import sys
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

from hira import retrieve_stats
from hira.src.config import (
    COMPARISON_PLOTS_DIR as PLOTS_DIR,
    OUTPUT_DIR,
    MAJOR_CTS,
    surrogate_names,
    get_config_fa,
    get_available_fa_analyses
)
from hira.src.config import get_config

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "Arial"

# Define JAK-STAT pathway TFs
JAK_STAT_TFS = [
    'STAT1', 'STAT2', 'STAT3', 'STAT4', 'STAT5A', 'STAT5B', 'STAT6',
    'IRF1', 'IRF2', 'IRF3', 'IRF4', 'IRF5', 'IRF7', 'IRF8', 'IRF9',
    'JUN', 'JUNB', 'JUND', 'FOS', 'FOSB', 'FOSL1', 'FOSL2',
    'NFKB1', 'NFKB2', 'RELA', 'RELB', 'REL',
    'SOCS1', 'SOCS2', 'SOCS3', 'SOCS5',
    'BCL6', 'MYC', 'CEBPB', 'CEBPD'
]


def print_jakstat_details(merged_data, cell_type):
    """Print details of JAK-STAT TFs in the comparison"""
    jakstat_data = merged_data[merged_data['gene'].isin(JAK_STAT_TFS)].copy()
    
    if len(jakstat_data) == 0:
        print(f"\n  {cell_type}: No JAK-STAT TFs found in overlapping data")
        return jakstat_data
    
    print(f"\n  {'='*70}")
    print(f"  JAK-STAT TFs in {cell_type} ({len(jakstat_data)} TFs found)")
    print(f"  {'='*70}")
    
    # Sort by absolute IL10 effect
    jakstat_data = jakstat_data.sort_values('abs_log10_p_adj_il10', ascending=False)
    
    print(f"\n  {'Gene':<10} {'IL10_slope':<12} {'IL10_pAdj':<12} {'Ruxo_slope':<12} {'Ruxo_pAdj':<12} {'Direction':<12}")
    print(f"  {'-'*70}")
    
    for _, row in jakstat_data.iterrows():
        il10_slope = f"{row['slope_il10']:.3f}"
        il10_padj = f"{row['p_value_adj_il10']:.2e}"
        ruxo_slope = f"{row['slope_ruxo']:.3f}"
        ruxo_padj = f"{row['p_value_adj_ruxo']:.2e}"
        direction = "Consistent" if row['consistent'] else "Opposing"
        
        print(f"  {row['gene']:<10} {il10_slope:<12} {il10_padj:<12} {ruxo_slope:<12} {ruxo_padj:<12} {direction:<12}")
    
    # Summary statistics
    consistent_count = jakstat_data['consistent'].sum()
    opposing_count = len(jakstat_data) - consistent_count
    
    print(f"\n  Summary:")
    print(f"    Consistent direction: {consistent_count}/{len(jakstat_data)} TFs")
    print(f"    Opposing direction:   {opposing_count}/{len(jakstat_data)} TFs")
    
    # Significant in both
    sig_threshold = 0.05
    sig_both = jakstat_data[
        (jakstat_data['p_value_adj_il10'] < sig_threshold) & 
        (jakstat_data['p_value_adj_ruxo'] < sig_threshold)
    ]
    print(f"    Significant in both treatments: {len(sig_both)}/{len(jakstat_data)} TFs")
    
    if len(sig_both) > 0:
        print(f"\n  Key JAK-STAT TFs significant in both:")
        for _, row in sig_both.iterrows():
            direction = "same direction" if row['consistent'] else "opposite directions"
            print(f"    - {row['gene']}: {direction}")
    
    print(f"  {'='*70}\n")
    
    return jakstat_data


def plot_directional_comparison(stats_il10, stats_ruxo, cell_types, args):
    """
    Plot directional comparison between IL10 and Ruxolitinib effects.
    
    Parameters
    ----------
    stats_il10 : pd.DataFrame
        Statistics from parsebioscience (IL10) dataset
    stats_ruxo : pd.DataFrame
        Statistics from op (Ruxolitinib) dataset
    cell_types : list
        List of cell types to analyze
    args : argparse.Namespace
        Command-line arguments
    """
    output_dir = args.output_dir
    
    # Define colors
    opposing_color = 'indianred'
    consistent_color = 'darkseagreen'
    
    # label_consistent = 'Consistent \n ({} TFs)'
    # label_opposing = 'Opposing \n ({} TFs)'
    
    all_cell_data = []
    
    for cell_type in cell_types:
        # Filter data for current cell type
        if cell_type not in stats_il10['cell_type'].unique():
            print(f"  Warning: No IL10 data for {cell_type}")
            continue
        if cell_type not in stats_ruxo['cell_type'].unique():
            print(f"  Warning: No Ruxolitinib data for {cell_type}")
            continue
            
        il10_ct = stats_il10[stats_il10['cell_type'] == cell_type].copy()
        ruxo_ct = stats_ruxo[stats_ruxo['cell_type'] == cell_type].copy()
        
        # Merge on gene: keep only TFs present in both datasets
        merged = il10_ct[['gene', 'slope', 'p_value_adj']].merge(
            ruxo_ct[['gene', 'slope', 'p_value_adj']],
            on='gene',
            how='inner',
            suffixes=('_il10', '_ruxo')
        )
        
        if len(merged) == 0:
            print(f"  Warning: No overlapping genes for {cell_type}")
            continue
        
        print(f"\n  {cell_type}: {len(merged)} overlapping TFs")
        
        # Calculate signed -log10 p-values
        merged['-log10_p_adj_il10'] = -np.log10(merged['p_value_adj_il10'] + 1e-300) * np.sign(merged['slope_il10'])
        merged['-log10_p_adj_ruxo'] = -np.log10(merged['p_value_adj_ruxo'] + 1e-300) * np.sign(merged['slope_ruxo'])
        
        # Determine consistency: same direction = consistent
        merged['consistent'] = np.sign(merged['slope_il10']) == np.sign(merged['slope_ruxo'])
        merged['cell_type'] = cell_type
        
        # Print top TFs
        merged['abs_log10_p_adj_il10'] = merged['p_value_adj_il10'].apply(lambda x: -np.log10(x + 1e-300))
        
        print(f"  Top 5 TFs with positive IL10 slope:")
        top_5 = merged[merged['slope_il10'] > 0].nlargest(5, 'abs_log10_p_adj_il10')
        names = ', '.join(top_5['gene'].tolist())
        print(f"    {names}")
        
        print(f"  Top 5 TFs with negative IL10 slope:")
        top_5 = merged[merged['slope_il10'] < 0].nlargest(5, 'abs_log10_p_adj_il10')
        names = ', '.join(top_5['gene'].tolist())
        print(f"    {names}")
        
        # Print JAK-STAT TF details
        jakstat_data = print_jakstat_details(merged, cell_type)
        merged['is_jakstat'] = merged['gene'].isin(JAK_STAT_TFS)
        
        all_cell_data.append(merged)
    
    if not all_cell_data:
        print("  Warning: No data to plot")
        return
    
    # Combine all cell types
    combined_data = pd.concat(all_cell_data, ignore_index=True)
    
    # Create grouped plot
    fig, axes = plt.subplots(1, len(cell_types), figsize=(4 * len(cell_types) + 1, 4), sharey=False)
    
    if len(cell_types) == 1:
        axes = [axes]
    
    for idx, cell_type in enumerate(cell_types):
        ax = axes[idx]
        cell_data = combined_data[combined_data['cell_type'] == cell_type]
        
        if len(cell_data) == 0:
            continue
        
        # Separate JAK-STAT and non-JAK-STAT TFs
        jakstat_mask = cell_data['is_jakstat']
        non_jakstat = cell_data[~jakstat_mask]
        jakstat = cell_data[jakstat_mask]
        
        consistent = non_jakstat[non_jakstat['consistent']]
        inconsistent = non_jakstat[~non_jakstat['consistent']]
        
        jakstat_consistent = jakstat[jakstat['consistent']]
        jakstat_inconsistent = jakstat[~jakstat['consistent']]
        
        # Plot parameters
        s = 10
        linewidths = 0.1
        
        # Plot non-JAK-STAT TFs (background)
        # Plot inconsistent (opposing)
        if len(inconsistent) > 0:
            ax.scatter(
                inconsistent['-log10_p_adj_il10'],
                inconsistent['-log10_p_adj_ruxo'],
                c=opposing_color,
                s=s,
                alpha=0.4,
                edgecolors='darkred',
                linewidths=linewidths,
                zorder=1
            )
        
        # Plot consistent
        if len(consistent) > 0:
            ax.scatter(
                consistent['-log10_p_adj_il10'],
                consistent['-log10_p_adj_ruxo'],
                c=consistent_color,
                s=s,
                alpha=0.4,
                edgecolors='darkgreen',
                linewidths=linewidths,
                zorder=1
            )
        
        # Plot JAK-STAT TFs (foreground with larger markers and annotations)
        jakstat_s = 30
        jakstat_linewidths = 0.8
        
        # Plot JAK-STAT inconsistent
        if len(jakstat_inconsistent) > 0:
            ax.scatter(
                jakstat_inconsistent['-log10_p_adj_il10'],
                jakstat_inconsistent['-log10_p_adj_ruxo'],
                c=opposing_color,
                s=jakstat_s,
                alpha=0.9,
                edgecolors='darkred',
                linewidths=jakstat_linewidths,
                zorder=3,
                marker='D'
            )
            
        # Plot JAK-STAT consistent
        if len(jakstat_consistent) > 0:
            ax.scatter(
                jakstat_consistent['-log10_p_adj_il10'],
                jakstat_consistent['-log10_p_adj_ruxo'],
                c=consistent_color,
                s=jakstat_s,
                alpha=0.9,
                edgecolors='darkgreen',
                linewidths=jakstat_linewidths,
                zorder=3,
                marker='D'
            )
        
        # Annotate JAK-STAT TFs that are significant in both treatments
        if len(jakstat) > 0:
            sig_threshold = 0.05
            jakstat_sig_both = jakstat[
                (jakstat['p_value_adj_il10'] < sig_threshold) & 
                (jakstat['p_value_adj_ruxo'] < sig_threshold)
            ].copy()
            
            if len(jakstat_sig_both) > 0:
                # Sort by position to detect overlaps
                jakstat_sig_both = jakstat_sig_both.sort_values(['-log10_p_adj_ruxo', '-log10_p_adj_il10'])
                
                # Track label positions to avoid overlaps
                used_positions = []
                distance_threshold = 0.5  # Threshold for considering points too close
                
                for _, row in jakstat_sig_both.iterrows():
                    x_pos = row['-log10_p_adj_il10']
                    y_pos = row['-log10_p_adj_ruxo']
                    
                    # Check if this position is too close to any existing label
                    too_close = False
                    for used_x, used_y in used_positions:
                        distance = np.sqrt((x_pos - used_x)**2 + (y_pos - used_y)**2)
                        if distance < distance_threshold:
                            too_close = True
                            break
                    
                    # Determine direction away from center (0, 0)
                    # Calculate offset based on quadrant
                    x_offset = 15 if x_pos > 0 else -15
                    y_offset = 10 if y_pos > 0 else -10
                    ha_align = 'left' if x_pos > 0 else 'right'
                    va_align = 'bottom' if y_pos > 0 else 'top'
                    
                    if too_close:
                        # Place label further away with arrow, positioned away from center
                        ax.annotate(
                            row['gene'],
                            xy=(x_pos, y_pos),
                            xytext=(x_offset, y_offset),
                            textcoords='offset points',
                            fontsize=7,
                            alpha=0.9,
                            ha=ha_align,
                            va=va_align,
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='gray', lw=0.5, alpha=0.7),
                            zorder=4
                        )
                    else:
                        # Place label right beside the point, away from center
                        simple_x_offset = 3 if x_pos > 0 else -3
                        simple_ha = 'left' if x_pos > 0 else 'right'
                        ax.annotate(
                            row['gene'],
                            xy=(x_pos, y_pos),
                            xytext=(simple_x_offset, 0),
                            textcoords='offset points',
                            fontsize=7,
                            alpha=0.9,
                            ha=simple_ha,
                            va='center',
                            zorder=4
                        )
                    
                    used_positions.append((x_pos, y_pos))
        
        # Styling
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        
        # Set symmetric axis limits around zero
        x_max = max(abs(cell_data['-log10_p_adj_il10'].min()), abs(cell_data['-log10_p_adj_il10'].max()))
        y_max = max(abs(cell_data['-log10_p_adj_ruxo'].min()), abs(cell_data['-log10_p_adj_ruxo'].max()))
        
        # Add padding - more for x-axis
        x_max *= 1.45
        y_max *= 1.1
        
        ax.set_xlim(-x_max, x_max)
        ax.set_ylim(-y_max, y_max)
        
        ax.set_xlabel('IL10 treatment \n(significance)', fontsize=10)
        ax.set_ylabel('Ruxolitinib treatment \n(significance)', fontsize=10)
        
        ax.set_title(f'{cell_type}', fontsize=12, pad=30)
        ax.grid(False)
        
        # Create legend
        legend_elements = []
        markersize = 5
        
        # Total counts
        total_consistent = len(cell_data[cell_data['consistent']])
        total_inconsistent = len(cell_data[~cell_data['consistent']])
        jakstat_count = len(jakstat)
        
        if total_inconsistent > 0:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', 
                       markerfacecolor=opposing_color, markersize=markersize,
                       markeredgecolor='darkred', markeredgewidth=0.5,
                       label=f'Opposing ({total_inconsistent} TFs)')
            )
        if total_consistent > 0:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', 
                       markerfacecolor=consistent_color, markersize=markersize,
                       markeredgecolor='darkgreen', markeredgewidth=0.5,
                       label=f'Consistent ({total_consistent} TFs)')
            )
        
        # Add JAK-STAT TF indicator
        if jakstat_count > 0:
            legend_elements.append(
                Line2D([0], [0], marker='D', color='w', 
                       markerfacecolor='gray', markersize=6,
                       markeredgecolor='black', markeredgewidth=0.8,
                       label=f'JAK-STAT known TFs ({jakstat_count})')
            )
        
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), 
                 frameon=False, fontsize=8, ncol=1)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'comparison_il10_ruxolitinib_{args.analysis_name}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\n✓ Saved comparison plot: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare IL10 (parsebioscience) to Ruxolitinib (op) effects'
    )
    parser.add_argument(
        '--analysis-name',
        type=str,
        required=True,
        choices=get_available_fa_analyses(),
        help='Analysis configuration name'
    )


    parser.add_argument(
        '--output-dir',
        type=str,
        default=PLOTS_DIR,
        help='Output directory for plots (default: PLOTS_DIR from config)'
    )
    parser.add_argument(
        '--cell-types',
        type=str,
        nargs='+',
        default=['CD4T', 'CD8T'],
        help='Cell types to analyze (default: CD4T CD8T)'
    )
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("Comparing IL10 (parsebioscience) to Ruxolitinib (op)")
    print(f"Analysis name: {args.analysis_name}")

    print(f"Cell types: {', '.join(args.cell_types)}")
    print("="*60)
    
    # Load stats for both datasets
    print("\nLoading IL10 (parsebioscience) stats...")
    stats_il10 = retrieve_stats(
        dataset='parsebioscience',
        analysis_name=args.analysis_name,
        multi_cohort=False
    )
    print(f"  Loaded {len(stats_il10)} records")
    
    print("\nLoading Ruxolitinib (op) stats...")
    stats_ruxo = retrieve_stats(
        dataset='op',
        analysis_name=args.analysis_name,
        multi_cohort=False
    )
    print(f"  Loaded {len(stats_ruxo)} records")
    
    # Plot directional comparison
    print("\nCreating directional comparison plot...")
    plot_directional_comparison(stats_il10, stats_ruxo, args.cell_types, args)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == '__main__':
    main()
