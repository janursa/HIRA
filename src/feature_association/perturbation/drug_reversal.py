"""
Drug reversal analysis: Identify rejuvenating drugs based on TF activity reversal.

This module analyzes drugs by their ability to reverse age-associated TF activity changes
using Fisher's exact test, independent of aging clock predictions.
"""

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import os
from typing import Tuple, Dict, List
from ciim.src.common import SAVE_DIR


def compute_tf_overlap(
    aging_stats: pd.DataFrame,
    drug_stats: pd.DataFrame,
    cell_type: str,
    drug_name: str,
    sig_threshold: float = 0.05
) -> Dict[str, int]:
    """
    Compute overlap between age-associated and drug-associated TF changes.
    
    Parameters
    ----------
    aging_stats : pd.DataFrame
        Aging statistics with columns: tf, cell_type, slope, meta_p_adj
    drug_stats : pd.DataFrame
        Drug statistics with columns: tf, cell_type, slope_condition, p_value_adj, comparision
    cell_type : str
        Cell type to analyze
    drug_name : str
        Drug/comparison name
    sig_threshold : float
        Significance threshold (default: 0.05)
    
    Returns
    -------
    Dict[str, int]
        Contingency table with keys: a, b, c, d
        a: drug_increase & age_increase (ACCELERATION)
        b: drug_decrease & age_increase (REVERSAL - decreasing increasing TFs)
        c: drug_increase & age_decrease (REVERSAL - increasing decreasing TFs)
        d: drug_decrease & age_decrease (ACCELERATION)
    """
    # Filter for cell type
    aging_ct = aging_stats[aging_stats['cell_type'] == cell_type].copy()
    drug_ct = drug_stats[
        (drug_stats['cell_type'] == cell_type) & 
        (drug_stats['comparision'] == drug_name)
    ].copy()
    
    # Filter for significant changes
    aging_sig = aging_ct[aging_ct['meta_p_adj'] < sig_threshold]
    drug_sig = drug_ct[drug_ct['p_value_adj'] < sig_threshold]
    
    # Categorize by direction
    aging_sig['direction'] = aging_sig['slope'].apply(lambda x: 'increase' if x > 0 else 'decrease')
    drug_sig['direction'] = drug_sig['slope_condition'].apply(lambda x: 'increase' if x > 0 else 'decrease')
    
    # Find overlap
    common_tfs = set(aging_sig['tf']) & set(drug_sig['tf'])
    
    if len(common_tfs) == 0:
        return {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'n_common': 0}
    
    # Create merged dataframe for common TFs
    merged = pd.merge(
        aging_sig[['tf', 'direction']],
        drug_sig[['tf', 'direction']],
        on='tf',
        suffixes=('_age', '_drug')
    )
    
    # Count combinations
    a = len(merged[(merged['direction_age'] == 'increase') & (merged['direction_drug'] == 'increase')])
    b = len(merged[(merged['direction_age'] == 'increase') & (merged['direction_drug'] == 'decrease')])
    c = len(merged[(merged['direction_age'] == 'decrease') & (merged['direction_drug'] == 'increase')])
    d = len(merged[(merged['direction_age'] == 'decrease') & (merged['direction_drug'] == 'decrease')])
    
    return {
        'a': a,  # Both increase (acceleration)
        'b': b,  # Age increase, drug decrease (reversal)
        'c': c,  # Age decrease, drug increase (reversal)
        'd': d,  # Both decrease (acceleration)
        'n_common': len(common_tfs)
    }


def fisher_exact_test_reversal(contingency: Dict[str, int]) -> Tuple[float, float, str]:
    """
    Run Fisher's exact test for reversal vs acceleration.
    
    Tests whether the drug shows significant reversal (b+c) vs acceleration (a+d).
    
    Parameters
    ----------
    contingency : Dict[str, int]
        Contingency table from compute_tf_overlap
    
    Returns
    -------
    Tuple[float, float, str]
        (p_value, odds_ratio, effect_type)
        effect_type: 'reversal', 'acceleration', or 'neutral'
    """
    a, b, c, d = contingency['a'], contingency['b'], contingency['c'], contingency['d']
    
    # No overlapping TFs
    if a + b + c + d == 0:
        return 1.0, 1.0, 'neutral'
    
    # Create 2x2 table: [reversal, acceleration]
    # Row 1: Drug increases TFs
    # Row 2: Drug decreases TFs
    # Col 1: Age increases TFs
    # Col 2: Age decreases TFs
    table = [[a, c],  # Drug increases: with age-increase (a), with age-decrease (c)
             [b, d]]  # Drug decreases: with age-increase (b), with age-decrease (d)
    
    # Run Fisher's exact test (two-sided)
    odds_ratio, p_value = fisher_exact(table, alternative='two-sided')
    
    # Determine effect type based on counts
    reversal_count = b + c  # Opposite directions
    acceleration_count = a + d  # Same directions
    
    if reversal_count > acceleration_count:
        effect_type = 'reversal'
    elif acceleration_count > reversal_count:
        effect_type = 'acceleration'
    else:
        effect_type = 'neutral'
    
    return p_value, odds_ratio, effect_type


def calculate_reversal_score(contingency: Dict[str, int]) -> float:
    """
    Calculate a reversal score: ratio of reversal to total effects.
    
    Score ranges from -1 (complete acceleration) to +1 (complete reversal).
    
    Parameters
    ----------
    contingency : Dict[str, int]
        Contingency table from compute_tf_overlap
    
    Returns
    -------
    float
        Reversal score
    """
    a, b, c, d = contingency['a'], contingency['b'], contingency['c'], contingency['d']
    
    reversal = b + c
    acceleration = a + d
    total = reversal + acceleration
    
    if total == 0:
        return 0.0
    
    # Score: (reversal - acceleration) / total
    # +1 = complete reversal, -1 = complete acceleration, 0 = neutral
    score = (reversal - acceleration) / total
    
    return score


def classify_drug_effect(
    p_value: float,
    effect_type: str,
    reversal_score: float,
    p_threshold: float = 0.05,
    score_threshold: float = 0.2
) -> str:
    """
    Classify drug effect as rejuvenating, accelerating, or neutral.
    
    Parameters
    ----------
    p_value : float
        Fisher's exact test p-value (FDR-adjusted)
    effect_type : str
        Effect type from fisher_exact_test_reversal
    reversal_score : float
        Reversal score from calculate_reversal_score
    p_threshold : float
        Significance threshold for p-value
    score_threshold : float
        Minimum absolute reversal score to be non-neutral
    
    Returns
    -------
    str
        'rejuvenating', 'accelerating', or 'neutral'
    """
    if p_value >= p_threshold:
        return 'neutral'
    
    if abs(reversal_score) < score_threshold:
        return 'neutral'
    
    if effect_type == 'reversal' and reversal_score > 0:
        return 'rejuvenating'
    elif effect_type == 'acceleration' and reversal_score < 0:
        return 'accelerating'
    else:
        return 'neutral'


def analyze_drug_reversal(
    aging_stats: pd.DataFrame,
    drug_stats: pd.DataFrame,
    cell_types: List[str],
    dataset: str,
    sig_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Run complete reversal analysis for all drugs and cell types.
    
    Parameters
    ----------
    aging_stats : pd.DataFrame
        Aging statistics (from retrieve_sig_stats)
    drug_stats : pd.DataFrame
        Drug statistics (from stats_drugs files)
    cell_types : List[str]
        List of cell types to analyze
    dataset : str
        Dataset name (e.g., 'op', 'CXCL9')
    sig_threshold : float
        Significance threshold for individual TF changes
    
    Returns
    -------
    pd.DataFrame
        Results with columns:
        - drug
        - cell_type
        - n_reversal (b+c)
        - n_acceleration (a+d)
        - n_common (total common TFs)
        - fisher_pvalue
        - fisher_pvalue_adj
        - odds_ratio
        - reversal_score
        - effect_type
        - classification
    """
    results = []
    
    # Get unique drugs
    drugs = drug_stats['comparision'].unique()
    
    for cell_type in cell_types:
        for drug in drugs:
            # Skip if no data for this combination
            drug_ct_data = drug_stats[
                (drug_stats['cell_type'] == cell_type) & 
                (drug_stats['comparision'] == drug)
            ]
            
            if len(drug_ct_data) == 0:
                continue
            
            # Compute contingency table
            contingency = compute_tf_overlap(
                aging_stats, drug_stats, cell_type, drug, sig_threshold
            )
            
            # Run Fisher's exact test
            p_value, odds_ratio, effect_type = fisher_exact_test_reversal(contingency)
            
            # Calculate reversal score
            rev_score = calculate_reversal_score(contingency)
            
            results.append({
                'drug': drug,
                'cell_type': cell_type,
                'dataset': dataset,
                'n_reversal': contingency['b'] + contingency['c'],
                'n_decrease_increasing': contingency['b'],
                'n_increase_decreasing': contingency['c'],
                'n_acceleration': contingency['a'] + contingency['d'],
                'n_common': contingency['n_common'],
                'fisher_pvalue': p_value,
                'odds_ratio': odds_ratio,
                'reversal_score': rev_score,
                'effect_type': effect_type,
                'a': contingency['a'],
                'b': contingency['b'],
                'c': contingency['c'],
                'd': contingency['d']
            })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        return results_df
    
    # Apply FDR correction
    results_df['fisher_pvalue_adj'] = multipletests(
        results_df['fisher_pvalue'], 
        method='fdr_bh'
    )[1]
    
    # Classify drugs
    results_df['classification'] = results_df.apply(
        lambda row: classify_drug_effect(
            row['fisher_pvalue_adj'],
            row['effect_type'],
            row['reversal_score']
        ),
        axis=1
    )
    
    # Sort by reversal score (descending)
    results_df = results_df.sort_values('reversal_score', ascending=False)
    
    return results_df


def filter_rejuvenating_drugs(
    results_df: pd.DataFrame,
    p_threshold: float = 0.05,
    min_reversal_score: float = 0.2,
    min_common_tfs: int = 3
) -> pd.DataFrame:
    """
    Filter for rejuvenating drugs only.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from analyze_drug_reversal
    p_threshold : float
        Maximum adjusted p-value
    min_reversal_score : float
        Minimum reversal score
    min_common_tfs : int
        Minimum number of common TFs
    
    Returns
    -------
    pd.DataFrame
        Filtered dataframe with rejuvenating drugs only
    """
    filtered = results_df[
        (results_df['fisher_pvalue_adj'] < p_threshold) &
        (results_df['reversal_score'] > min_reversal_score) &
        (results_df['n_common'] >= min_common_tfs) &
        (results_df['classification'] == 'rejuvenating')
    ].copy()
    
    return filtered


def save_reversal_results(
    results_df: pd.DataFrame,
    dataset: str,
    output_dir: str = None
) -> str:
    """
    Save reversal analysis results to CSV.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from analyze_drug_reversal
    dataset : str
        Dataset name
    output_dir : str, optional
        Output directory (default: SAVE_DIR/perturbations)
    
    Returns
    -------
    str
        Path to saved file
    """
    if output_dir is None:
        output_dir = f'{SAVE_DIR}/perturbations'
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = f'{output_dir}/reversal_stats_{dataset}.csv'
    results_df.to_csv(output_path, index=False)
    
    print(f"Saved reversal results to: {output_path}")
    
    return output_path


def print_reversal_summary(results_df: pd.DataFrame, dataset: str):
    """
    Print a summary of reversal analysis results.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from analyze_drug_reversal
    dataset : str
        Dataset name
    """
    print(f"\n{'='*80}")
    print(f"DRUG REVERSAL ANALYSIS SUMMARY - {dataset.upper()}")
    print(f"{'='*80}\n")
    
    total_combinations = len(results_df)
    print(f"Total drug-cell type combinations analyzed: {total_combinations}")
    
    if total_combinations == 0:
        print("No results to display.")
        return
    
    # Summary by classification
    classification_counts = results_df['classification'].value_counts()
    print(f"\nClassification summary:")
    for classification, count in classification_counts.items():
        print(f"  {classification.capitalize()}: {count}")
    
    # Top rejuvenating drugs
    rejuvenating = results_df[results_df['classification'] == 'rejuvenating']
    if len(rejuvenating) > 0:
        print(f"\n{'-'*80}")
        print(f"TOP REJUVENATING DRUGS (sorted by reversal score):")
        print(f"{'-'*80}")
        for idx, row in rejuvenating.head(10).iterrows():
            print(f"\n  Drug: {row['drug']}")
            print(f"  Cell type: {row['cell_type']}")
            print(f"  Reversal score: {row['reversal_score']:.3f}")
            print(f"  P-value (adj): {row['fisher_pvalue_adj']:.4e}")
            print(f"  Common TFs: {row['n_common']}")
            print(f"  Reversal TFs: {row['n_reversal']} (↓{row['n_decrease_increasing']} increasing, ↑{row['n_increase_decreasing']} decreasing)")
            print(f"  Acceleration TFs: {row['n_acceleration']}")
    else:
        print("\nNo rejuvenating drugs found with current thresholds.")
    
    # Statistics
    if len(results_df) > 0:
        print(f"\n{'-'*80}")
        print(f"STATISTICS:")
        print(f"{'-'*80}")
        print(f"  Mean reversal score: {results_df['reversal_score'].mean():.3f}")
        print(f"  Median reversal score: {results_df['reversal_score'].median():.3f}")
        print(f"  Mean common TFs: {results_df['n_common'].mean():.1f}")
        print(f"  Significant (p<0.05): {(results_df['fisher_pvalue_adj'] < 0.05).sum()}")
    
    print(f"\n{'='*80}\n")
