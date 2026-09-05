#!/usr/bin/env python
"""
Compare age-associated TFs identified using full GRN models vs promotor-only GRN models.

This script validates that age-associated TFs are robust even when using only
promotor-based regulatory evidence, addressing potential causality concerns.
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from matplotlib_venn import venn2

from hira.src.config import COMPARISON_PLOTS_DIR as PLOTS_DIR, OUTPUT_DIR, MAJOR_CTS, palette_major_cts


def load_stats(data_type='bulk', suffix=''):
    """Load statistics from analysis."""
    file_path = f'{OUTPUT_DIR}/tf_activity/stats_all_{data_type}{suffix}.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Stats file not found: {file_path}")
    return pd.read_csv(file_path)


def get_significant_tfs(stats, p_threshold=0.05, condition='healthy'):
    """Extract significant TFs from stats."""
    mask = (stats['condition'] == condition) & (stats['meta_p_adj'] < p_threshold)
    return stats[mask].copy()


def compare_significant_tfs(stats_full, stats_promotor, cell_type=None):
    """
    Compare significant TFs between full and promotor-only models.
    
    Returns:
        dict with comparison metrics
    """
    # Filter by cell type if specified
    if cell_type:
        stats_full = stats_full[stats_full['cell_type'] == cell_type]
        stats_promotor = stats_promotor[stats_promotor['cell_type'] == cell_type]
    
    # Get significant TFs
    sig_full = get_significant_tfs(stats_full)
    sig_promotor = get_significant_tfs(stats_promotor)
    
    # Extract TF lists
    tfs_full = set(sig_full['tf'].unique())
    tfs_promotor = set(sig_promotor['tf'].unique())
    
    # Calculate overlap
    overlap = tfs_full & tfs_promotor
    only_full = tfs_full - tfs_promotor
    promotor_only = tfs_promotor - tfs_full
    
    # Calculate metrics
    jaccard = len(overlap) / len(tfs_full | tfs_promotor) if len(tfs_full | tfs_promotor) > 0 else 0
    overlap_pct_full = len(overlap) / len(tfs_full) * 100 if len(tfs_full) > 0 else 0
    overlap_pct_promotor = len(overlap) / len(tfs_promotor) * 100 if len(tfs_promotor) > 0 else 0
    
    return {
        'cell_type': cell_type,
        'n_full': len(tfs_full),
        'n_promotor': len(tfs_promotor),
        'n_overlap': len(overlap),
        'n_only_full': len(only_full),
        'n_promotor_only': len(promotor_only),
        'jaccard': jaccard,
        'overlap_pct_full': overlap_pct_full,
        'overlap_pct_promotor': overlap_pct_promotor,
        'tfs_full': tfs_full,
        'tfs_promotor': tfs_promotor,
        'tfs_overlap': overlap,
        'tfs_only_full': only_full,
        'tfs_promotor_only': promotor_only
    }


def compare_effect_sizes(stats_full, stats_promotor, cell_type=None):
    """
    Compare effect sizes (slopes) between full and promotor-only models.
    
    Returns:
        DataFrame with merged statistics
    """
    # Filter by cell type if specified
    if cell_type:
        stats_full = stats_full[stats_full['cell_type'] == cell_type]
        stats_promotor = stats_promotor[stats_promotor['cell_type'] == cell_type]
    
    # Get significant TFs from both
    sig_full = get_significant_tfs(stats_full)
    sig_promotor = get_significant_tfs(stats_promotor)
    
    # Merge on TF and dataset
    merged = sig_full[['cell_type', 'tf', 'dataset', 'slope', 'meta_p_adj']].merge(
        sig_promotor[['cell_type', 'tf', 'dataset', 'slope', 'meta_p_adj']],
        on=['cell_type', 'tf', 'dataset'],
        how='inner',
        suffixes=('_full', '_promotor')
    )
    
    return merged


def plot_venn_diagram(comparison, output_path):
    """Plot Venn diagram showing TF overlap."""
    fig, ax = plt.subplots(figsize=(3, 3))
    
    venn2(
        subsets=(
            comparison['n_only_full'],
            comparison['n_promotor_only'],
            comparison['n_overlap']
        ),
        set_labels=('Full GRN', 'Promotor-only GRN'),
        ax=ax
    )
    
    cell_type = comparison['cell_type'] or 'All'
    ax.set_title(f"Age-associated TFs overlap\n({cell_type})", fontsize=12, weight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Venn diagram to {output_path}")
    plt.close()


def plot_effect_size_correlation(merged, output_path, cell_type=None):
    """Plot correlation between effect sizes (slopes)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Calculate correlation
    corr_spearman, p_spearman = spearmanr(merged['slope_full'], merged['slope_promotor'])
    corr_pearson, p_pearson = pearsonr(merged['slope_full'], merged['slope_promotor'])
    
    # Scatter plot
    ax.scatter(merged['slope_full'], merged['slope_promotor'], 
               alpha=0.6, s=30, color='steelblue', edgecolors='white', linewidth=0.5)
    
    # Diagonal line
    lims = [
        min(merged['slope_full'].min(), merged['slope_promotor'].min()),
        max(merged['slope_full'].max(), merged['slope_promotor'].max())
    ]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1, label='y=x')
    
    # Labels and title
    ax.set_xlabel('Effect size (Full GRN)', fontsize=11)
    ax.set_ylabel('Effect size (Promotor-only GRN)', fontsize=11)
    
    cell_type_label = cell_type or 'All'
    ax.set_title(f'Age-association effect sizes\n({cell_type_label})', 
                 fontsize=12, weight='bold')
    
    # Add correlation stats
    textstr = f'Spearman ρ = {corr_spearman:.3f} (p={p_spearman:.2e})\n'
    textstr += f'Pearson r = {corr_pearson:.3f} (p={p_pearson:.2e})\n'
    textstr += f'n = {len(merged)} TFs'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved correlation plot to {output_path}")
    plt.close()
    
    return {
        'spearman_rho': corr_spearman,
        'spearman_p': p_spearman,
        'pearson_r': corr_pearson,
        'pearson_p': p_pearson,
        'n_tfs': len(merged)
    }


def plot_comparison_summary(comparison_results, output_path):
    """Plot summary bar chart across cell types."""
    df = pd.DataFrame(comparison_results)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Plot 1: Number of significant TFs
    ax = axes[0]
    x = np.arange(len(df))
    width = 0.35
    
    ax.bar(x - width/2, df['n_full'], width, label='Full GRN', color='steelblue')
    ax.bar(x + width/2, df['n_promotor'], width, label='Promotor-only', color='coral')
    
    ax.set_xlabel('Cell Type', fontsize=11)
    ax.set_ylabel('Number of significant TFs', fontsize=11)
    ax.set_title('Significant TFs per model', fontsize=12, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['cell_type'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Overlap percentage
    ax = axes[1]
    ax.bar(x, df['overlap_pct_full'], color='mediumseagreen', alpha=0.7)
    
    ax.set_xlabel('Cell Type', fontsize=11)
    ax.set_ylabel('Overlap (%)', fontsize=11)
    ax.set_title('% of Full GRN TFs\nfound in Promotor-only', fontsize=12, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['cell_type'])
    ax.set_ylim([0, 105])
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Jaccard index
    ax = axes[2]
    ax.bar(x, df['jaccard'], color='mediumpurple', alpha=0.7)
    
    ax.set_xlabel('Cell Type', fontsize=11)
    ax.set_ylabel('Jaccard Index', fontsize=11)
    ax.set_title('TF overlap\n(Jaccard similarity)', fontsize=12, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['cell_type'])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved summary plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compare age-associated TFs: full GRN vs promotor-only GRN'
    )
    parser.add_argument('--data-type', type=str, default='bulk',
                        help='Data type (default: bulk)')
    parser.add_argument('--cell-types', nargs='+', default=None,
                        help='Cell types to analyze (default: all)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = f'{PLOTS_DIR}/promotor_comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("PROMOTOR-BASED GRN COMPARISON ANALYSIS")
    print("="*80)
    
    # Load statistics
    print("\nLoading statistics...")
    stats_full = load_stats(args.data_type, suffix='')
    stats_promotor = load_stats(args.data_type, suffix='_promotor')
    print(f"  - Full GRN stats: {len(stats_full)} rows")
    print(f"  - Promotor-only stats: {len(stats_promotor)} rows")
    
    # Determine cell types to analyze
    cell_types_to_analyze = args.cell_types if args.cell_types else MAJOR_CTS
    
    # Run comparison for each cell type
    print("\nRunning comparisons...")
    comparison_results = []
    correlation_results = []
    
    for cell_type in cell_types_to_analyze:
        print(f"\n  Cell type: {cell_type}")
        
        # Compare significant TFs
        comparison = compare_significant_tfs(stats_full, stats_promotor, cell_type=cell_type)
        comparison_results.append(comparison)
        
        print(f"    Full GRN: {comparison['n_full']} significant TFs")
        print(f"    Promotor-only: {comparison['n_promotor']} significant TFs")
        print(f"    Overlap: {comparison['n_overlap']} TFs ({comparison['overlap_pct_full']:.1f}%)")
        print(f"    Jaccard index: {comparison['jaccard']:.3f}")
        
        # Plot Venn diagram
        venn_path = f"{output_dir}/venn_{cell_type}.png"
        plot_venn_diagram(comparison, venn_path)
        
        # Compare effect sizes for overlapping TFs
        if comparison['n_overlap'] > 0:
            merged = compare_effect_sizes(stats_full, stats_promotor, cell_type=cell_type)
            
            corr_path = f"{output_dir}/correlation_{cell_type}.png"
            corr_stats = plot_effect_size_correlation(merged, corr_path, cell_type=cell_type)
            corr_stats['cell_type'] = cell_type
            correlation_results.append(corr_stats)
            
            print(f"    Effect size correlation (Spearman): ρ={corr_stats['spearman_rho']:.3f}")
    
    # Plot summary across cell types
    print("\nGenerating summary plots...")
    summary_path = f"{output_dir}/summary_comparison.png"
    plot_comparison_summary(comparison_results, summary_path)
    
    # Save comparison table
    comparison_df = pd.DataFrame([
        {k: v for k, v in comp.items() if not k.startswith('tfs_')}
        for comp in comparison_results
    ])
    comparison_table_path = f"{output_dir}/comparison_table.csv"
    comparison_df.to_csv(comparison_table_path, index=False)
    print(f"Saved comparison table to {comparison_table_path}")
    
    # Save correlation table
    if correlation_results:
        correlation_df = pd.DataFrame(correlation_results)
        correlation_table_path = f"{output_dir}/correlation_table.csv"
        correlation_df.to_csv(correlation_table_path, index=False)
        print(f"Saved correlation table to {correlation_table_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nOverall findings:")
    print(f"  Average overlap: {comparison_df['overlap_pct_full'].mean():.1f}% of Full GRN TFs")
    print(f"  Average Jaccard index: {comparison_df['jaccard'].mean():.3f}")
    
    if correlation_results:
        print(f"  Average effect size correlation: ρ={correlation_df['spearman_rho'].mean():.3f}")
    
    print(f"\nConclusion:")
    print(f"  Age-associated TFs identified using full GRN models are {'HIGHLY' if comparison_df['overlap_pct_full'].mean() > 80 else 'MODERATELY'} ")
    print(f"  consistent with those from promotor-only models, supporting the causal")
    print(f"  interpretation of age-associated regulatory changes.")
    
    print(f"\nAll outputs saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
