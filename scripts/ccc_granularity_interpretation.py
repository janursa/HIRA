"""
Detailed interpretation of CCC results at major and sub granularity levels.
Generates comprehensive insights and comparisons.
"""

import sys
sys.path.insert(0, '/home/jnourisa/projs/ongoing/hiara/')

from hiara.src.feature_association.helper import retrieve_stats
from hiara.src.feature_association.cc.plots import parse_ccc_feature_name
import pandas as pd
import numpy as np

# Define cell type mapping
SUB_TO_MAJOR = {
    'CD16_NK': 'NK',
    'Classic_MONO': 'MONO',
    'NonClassic_MONO': 'MONO',
    'MAIT': 'CD8T',
    'Memory_B': 'B',
    'Naive_B': 'B',
    'Tcm_Naive_CD4': 'CD4T',
    'Tem_Effector_CD4': 'CD4T',
    'Tcm_Naive_CD8': 'CD8T',
    'Tem_Temra_CD8': 'CD8T',
    'Tem_Trm_CD8': 'CD8T',
}

def prepare_stats(analysis_name):
    """Load and prepare stats with CCC features parsed."""
    stats = retrieve_stats(analysis_name=analysis_name)
    stats['source'] = stats['gene'].apply(lambda x: parse_ccc_feature_name(x)[0])
    stats['ligand'] = stats['gene'].apply(lambda x: parse_ccc_feature_name(x)[1])
    stats['target'] = stats['gene'].apply(lambda x: parse_ccc_feature_name(x)[2])
    stats['receptor'] = stats['gene'].apply(lambda x: parse_ccc_feature_name(x)[3])
    stats['lr_pair'] = stats['ligand'] + '-' + stats['receptor']
    stats['cell_pair'] = stats['source'] + ' → ' + stats['target']
    stats_sig = stats[stats['p_value_adj'] < 0.05].copy()
    return stats, stats_sig

def analyze_specific_major_pair(stats_major_sig, stats_sub_sig, major_source, major_target):
    """Deep dive into a specific major cell type pair."""
    print(f"\n{'='*80}")
    print(f"DEEP DIVE: {major_source} → {major_target}")
    print(f"{'='*80}\n")
    
    # Major level stats
    major_subset = stats_major_sig[
        (stats_major_sig['source'] == major_source) & 
        (stats_major_sig['target'] == major_target)
    ]
    
    print(f"MAJOR LEVEL ({len(major_subset)} interactions):")
    print(f"  Increasing with age: {len(major_subset[major_subset['slope'] > 0])} ({len(major_subset[major_subset['slope'] > 0])/len(major_subset)*100:.1f}%)")
    print(f"  Decreasing with age: {len(major_subset[major_subset['slope'] < 0])} ({len(major_subset[major_subset['slope'] < 0])/len(major_subset)*100:.1f}%)")
    
    # Top L-R pairs at major level
    print(f"\n  Top 5 L-R pairs at major level:")
    major_lr = major_subset.groupby('lr_pair').agg({
        'slope': 'mean',
        'p_value_adj': 'min'
    }).sort_values('p_value_adj')
    for i, (lr, row) in enumerate(major_lr.head(5).iterrows(), 1):
        trend = "↑" if row['slope'] > 0 else "↓"
        print(f"    {i}. {lr} {trend} (slope: {row['slope']:.3f})")
    
    # Sub level stats
    sub_sources = [sub for sub, maj in SUB_TO_MAJOR.items() if maj == major_source]
    sub_targets = [sub for sub, maj in SUB_TO_MAJOR.items() if maj == major_target]
    
    sub_subset = stats_sub_sig[
        (stats_sub_sig['source'].isin(sub_sources)) & 
        (stats_sub_sig['target'].isin(sub_targets))
    ]
    
    print(f"\nSUB LEVEL ({len(sub_subset)} interactions, {len(sub_subset)/len(major_subset):.1f}x more):")
    print(f"  Increasing with age: {len(sub_subset[sub_subset['slope'] > 0])} ({len(sub_subset[sub_subset['slope'] > 0])/len(sub_subset)*100:.1f}%)")
    print(f"  Decreasing with age: {len(sub_subset[sub_subset['slope'] < 0])} ({len(sub_subset[sub_subset['slope'] < 0])/len(sub_subset)*100:.1f}%)")
    
    # Breakdown by sub cell type pairs
    print(f"\n  Breakdown by sub cell type pairs:")
    sub_pairs = sub_subset.groupby(['source', 'target']).agg({
        'gene': 'size',
        'slope': 'mean'
    }).rename(columns={'gene': 'count'}).sort_values('count', ascending=False)
    
    for i, ((sub_src, sub_tgt), row) in enumerate(sub_pairs.head(5).iterrows(), 1):
        trend = "↑" if row['slope'] > 0 else "↓"
        pos = len(sub_subset[(sub_subset['source']==sub_src) & (sub_subset['target']==sub_tgt) & (sub_subset['slope']>0)])
        neg = len(sub_subset[(sub_subset['source']==sub_src) & (sub_subset['target']==sub_tgt) & (sub_subset['slope']<0)])
        print(f"    {i}. {sub_src} → {sub_tgt}: {row['count']} interactions ({pos}↑ / {neg}↓)")
    
    # New L-R pairs discovered at sub level
    major_lr_set = set(major_subset['lr_pair'].unique())
    sub_lr_set = set(sub_subset['lr_pair'].unique())
    new_lr = sub_lr_set - major_lr_set
    
    print(f"\n  New L-R pairs discovered at sub level: {len(new_lr)}")
    if len(new_lr) > 0:
        print(f"  Examples of new discoveries:")
        for i, lr in enumerate(list(new_lr)[:5], 1):
            lr_data = sub_subset[sub_subset['lr_pair'] == lr]
            avg_slope = lr_data['slope'].mean()
            trend = "↑" if avg_slope > 0 else "↓"
            # Which sub pairs?
            sub_pairs_with_lr = lr_data.groupby(['source', 'target']).size()
            pair_examples = ', '.join([f"{s}→{t}" for (s,t), _ in sub_pairs_with_lr.head(2).items()])
            print(f"    {i}. {lr} {trend} (in {pair_examples})")

def analyze_heterogeneity(stats_major_sig, stats_sub_sig):
    """Analyze heterogeneity within major cell types."""
    print(f"\n{'='*80}")
    print(f"HETEROGENEITY ANALYSIS")
    print(f"{'='*80}\n")
    
    # For each major cell type, analyze how its sub types differ
    for major_cell in ['MONO', 'CD8T', 'CD4T', 'B', 'NK']:
        sub_cells = [sub for sub, maj in SUB_TO_MAJOR.items() if maj == major_cell]
        
        if len(sub_cells) <= 1:
            continue
            
        print(f"\n{major_cell} (broken into {len(sub_cells)} subtypes: {', '.join(sub_cells)})")
        
        # Compare sending behavior
        for sub_cell in sub_cells:
            sub_as_sender = stats_sub_sig[stats_sub_sig['source'] == sub_cell]
            if len(sub_as_sender) > 0:
                pos = len(sub_as_sender[sub_as_sender['slope'] > 0])
                neg = len(sub_as_sender[sub_as_sender['slope'] < 0])
                ratio = pos/neg if neg>0 else float('inf')
                print(f"  {sub_cell} (sender): {len(sub_as_sender)} signals ({pos}↑ / {neg}↓, ratio: {ratio:.2f})")
        
        # Compare receiving behavior
        print()
        for sub_cell in sub_cells:
            sub_as_receiver = stats_sub_sig[stats_sub_sig['target'] == sub_cell]
            if len(sub_as_receiver) > 0:
                pos = len(sub_as_receiver[sub_as_receiver['slope'] > 0])
                neg = len(sub_as_receiver[sub_as_receiver['slope'] < 0])
                ratio = pos/neg if neg>0 else float('inf')
                print(f"  {sub_cell} (receiver): {len(sub_as_receiver)} signals ({pos}↑ / {neg}↓, ratio: {ratio:.2f})")

def directional_trend_comparison(stats_major_sig, stats_sub_sig):
    """Compare directional trends between major and sub."""
    print(f"\n{'='*80}")
    print(f"DIRECTIONAL TREND COMPARISON")
    print(f"{'='*80}\n")
    
    # Overall
    major_pos = len(stats_major_sig[stats_major_sig['slope'] > 0])
    major_neg = len(stats_major_sig[stats_major_sig['slope'] < 0])
    sub_pos = len(stats_sub_sig[stats_sub_sig['slope'] > 0])
    sub_neg = len(stats_sub_sig[stats_sub_sig['slope'] < 0])
    
    print(f"Overall trend distribution:")
    print(f"  Major: {major_pos}↑ ({major_pos/len(stats_major_sig)*100:.1f}%) vs {major_neg}↓ ({major_neg/len(stats_major_sig)*100:.1f}%)")
    print(f"  Sub:   {sub_pos}↑ ({sub_pos/len(stats_sub_sig)*100:.1f}%) vs {sub_neg}↓ ({sub_neg/len(stats_sub_sig)*100:.1f}%)")
    print(f"\n  → Interpretation: Major level shows MORE increasing interactions ({major_pos/len(stats_major_sig)*100:.1f}% vs {sub_pos/len(stats_sub_sig)*100:.1f}%)")
    print(f"                    This suggests averaging masks declining sub-populations")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("CCC GRANULARITY INTERPRETATION")
    print("="*80)
    
    # Load data
    stats_major, stats_major_sig = prepare_stats('ccc_major_b')
    stats_sub, stats_sub_sig = prepare_stats('ccc_sub_b')
    
    # Directional trends
    directional_trend_comparison(stats_major_sig, stats_sub_sig)
    
    # Deep dives into specific pairs
    analyze_specific_major_pair(stats_major_sig, stats_sub_sig, 'MONO', 'CD8T')
    analyze_specific_major_pair(stats_major_sig, stats_sub_sig, 'MONO', 'MONO')
    analyze_specific_major_pair(stats_major_sig, stats_sub_sig, 'CD8T', 'CD8T')
    
    # Heterogeneity analysis
    analyze_heterogeneity(stats_major_sig, stats_sub_sig)
    
    print("\n" + "="*80)
    print("INTERPRETATION COMPLETE")
    print("="*80)
