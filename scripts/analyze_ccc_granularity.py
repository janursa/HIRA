"""
Analyze CCC results at major and sub granularity levels.
Compare and interpret how granular (sub) data explains patterns seen in major cell types.
"""

import sys
sys.path.insert(0, '/home/jnourisa/projs/ongoing/hiara/')

from hiara.src.feature_association.helper import retrieve_stats
from hiara.src.feature_association.cc.plots import parse_ccc_feature_name
import pandas as pd
import numpy as np

def analyze_ccc_results(analysis_name):
    """Analyze CCC results for a given analysis."""
    print(f"\n{'='*80}")
    print(f"Analysis: {analysis_name}")
    print(f"{'='*80}\n")
    
    # Load stats
    stats = retrieve_stats(analysis_name=analysis_name)
    
    # Parse CCC feature components
    stats['source'] = stats['gene'].apply(lambda x: parse_ccc_feature_name(x)[0])
    stats['ligand'] = stats['gene'].apply(lambda x: parse_ccc_feature_name(x)[1])
    stats['target'] = stats['gene'].apply(lambda x: parse_ccc_feature_name(x)[2])
    stats['receptor'] = stats['gene'].apply(lambda x: parse_ccc_feature_name(x)[3])
    stats['lr_pair'] = stats['ligand'] + '-' + stats['receptor']
    stats['cell_pair'] = stats['source'] + ' → ' + stats['target']
    
    # Filter significant results
    stats_sig = stats[stats['p_value_adj'] < 0.05].copy()
    
    print(f"Total interactions tested: {len(stats):,}")
    print(f"Significant interactions (p_adj < 0.05): {len(stats_sig):,} ({len(stats_sig)/len(stats)*100:.1f}%)")
    
    # Overall trend
    positive_sig = stats_sig[stats_sig['slope'] > 0]
    negative_sig = stats_sig[stats_sig['slope'] < 0]
    print(f"\nDirectionality of significant interactions:")
    print(f"  - Increasing with age: {len(positive_sig):,} ({len(positive_sig)/len(stats_sig)*100:.1f}%)")
    print(f"  - Decreasing with age: {len(negative_sig):,} ({len(negative_sig)/len(stats_sig)*100:.1f}%)")
    
    # Cell type pair analysis
    print(f"\n--- Cell-Cell Communication Patterns ---")
    cell_pair_counts = stats_sig.groupby('cell_pair').size().sort_values(ascending=False)
    print(f"\nTop 10 most active cell type pairs (by # significant interactions):")
    for i, (pair, count) in enumerate(cell_pair_counts.head(10).items(), 1):
        print(f"  {i}. {pair}: {count} interactions")
    
    # Sender cell type analysis
    sender_counts = stats_sig.groupby('source').size().sort_values(ascending=False)
    print(f"\nTop 10 sender cell types (by # significant outgoing signals):")
    for i, (cell, count) in enumerate(sender_counts.head(10).items(), 1):
        print(f"  {i}. {cell}: {count} signals")
    
    # Receiver cell type analysis
    receiver_counts = stats_sig.groupby('target').size().sort_values(ascending=False)
    print(f"\nTop 10 receiver cell types (by # significant incoming signals):")
    for i, (cell, count) in enumerate(receiver_counts.head(10).items(), 1):
        print(f"  {i}. {cell}: {count} signals")
    
    # L-R pair analysis
    lr_counts = stats_sig.groupby('lr_pair').size().sort_values(ascending=False)
    print(f"\nTop 10 most prevalent L-R pairs (across all cell type pairs):")
    for i, (lr, count) in enumerate(lr_counts.head(10).items(), 1):
        # Get average slope for this L-R pair
        avg_slope = stats_sig[stats_sig['lr_pair'] == lr]['slope'].mean()
        trend = "↑" if avg_slope > 0 else "↓"
        print(f"  {i}. {lr}: {count} occurrences {trend}")
    
    # Dataset distribution
    print(f"\n--- Dataset Distribution ---")
    dataset_counts = stats_sig.groupby('dataset').size().sort_values(ascending=False)
    for dataset, count in dataset_counts.items():
        print(f"  {dataset}: {count} significant interactions")
    
    return stats, stats_sig


def compare_granularities(stats_major_sig, stats_sub_sig):
    """Compare major vs sub granularity results."""
    print(f"\n{'='*80}")
    print(f"COMPARISON: Major vs Sub Granularity")
    print(f"{'='*80}\n")
    
    # Extract unique cell types at each level
    major_sources = set(stats_major_sig['source'].unique())
    major_targets = set(stats_major_sig['target'].unique())
    major_cells = major_sources | major_targets
    
    sub_sources = set(stats_sub_sig['source'].unique())
    sub_targets = set(stats_sub_sig['target'].unique())
    sub_cells = sub_sources | sub_targets
    
    print(f"Number of cell types involved:")
    print(f"  Major: {len(major_cells)} cell types")
    print(f"  Sub: {len(sub_cells)} cell types")
    print(f"  Ratio: {len(sub_cells)/len(major_cells):.1f}x more granular\n")
    
    # Compare number of significant interactions
    print(f"Significant interactions:")
    print(f"  Major: {len(stats_major_sig):,}")
    print(f"  Sub: {len(stats_sub_sig):,}")
    print(f"  Ratio: {len(stats_sub_sig)/len(stats_major_sig):.1f}x more interactions at sub level\n")
    
    # Compare L-R pairs
    major_lr = set(stats_major_sig['lr_pair'].unique())
    sub_lr = set(stats_sub_sig['lr_pair'].unique())
    common_lr = major_lr & sub_lr
    
    print(f"Unique L-R pairs:")
    print(f"  Major: {len(major_lr)}")
    print(f"  Sub: {len(sub_lr)}")
    print(f"  Shared: {len(common_lr)} ({len(common_lr)/len(major_lr)*100:.1f}% of major)\n")
    
    # Map sub cell types to major cell types
    print(f"--- Mapping Sub to Major Cell Types ---")
    # Define explicit mapping based on naming conventions
    sub_to_major_mapping = {
        'CD16_NK': 'NK',
        'Classic_MONO': 'MONO',
        'NonClassic_MONO': 'MONO',
        'MAIT': 'CD8T',  # MAIT are CD8+ T cells
        'Memory_B': 'B',
        'Naive_B': 'B',
        'Tcm_Naive_CD4': 'CD4T',
        'Tem_Effector_CD4': 'CD4T',
        'Tcm_Naive_CD8': 'CD8T',
        'Tem_Temra_CD8': 'CD8T',
        'Tem_Trm_CD8': 'CD8T',
    }
    
    print(f"Mapped {len(sub_to_major_mapping)} sub types to major types:")
    for sub, major in sorted(sub_to_major_mapping.items()):
        print(f"  {sub} → {major}")
    
    # Find major cell pairs and their corresponding sub cell pairs
    print(f"--- Granularity Breakdown: Major → Sub ---")
    major_cell_pairs = stats_major_sig.groupby(['source', 'target']).size().sort_values(ascending=False)
    
    for (major_source, major_target), major_count in major_cell_pairs.head(5).items():
        print(f"\n{major_source} → {major_target} (Major level: {major_count} interactions)")
        
        # Find corresponding sub pairs
        sub_sources_mapped = [sub for sub, maj in sub_to_major_mapping.items() if maj == major_source]
        sub_targets_mapped = [sub for sub, maj in sub_to_major_mapping.items() if maj == major_target]
        
        # Filter sub results for these cell types
        sub_subset = stats_sub_sig[
            (stats_sub_sig['source'].isin(sub_sources_mapped)) & 
            (stats_sub_sig['target'].isin(sub_targets_mapped))
        ]
        
        if len(sub_subset) > 0:
            sub_pairs = sub_subset.groupby(['source', 'target']).size().sort_values(ascending=False)
            print(f"  Sub-level breakdown ({len(sub_subset)} total interactions):")
            for (sub_source, sub_target), sub_count in sub_pairs.head(5).items():
                print(f"    - {sub_source} → {sub_target}: {sub_count} interactions")
        else:
            print(f"  No matching sub-level pairs found (check cell type naming)")
    
    return sub_to_major_mapping


if __name__ == "__main__":
    # Analyze major granularity
    stats_major, stats_major_sig = analyze_ccc_results('ccc_major_b')
    
    # Analyze sub granularity
    stats_sub, stats_sub_sig = analyze_ccc_results('ccc_sub_b')
    
    # Compare them
    mapping = compare_granularities(stats_major_sig, stats_sub_sig)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
