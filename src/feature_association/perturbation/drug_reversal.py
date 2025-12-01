"""
Drug reversal analysis: Identify rejuvenating drugs based on TF activity reversal.

This module analyzes drugs by their ability to reverse age-associated TF activity changes.

TWO MODES OF OPERATION:
----------------------
1. Standard Fisher's exact test (use_weighting=False):
   - Uses simple overlap counts (a, b, c, d) from contingency table
   - Standard Fisher's exact test for statistical significance
   - Fast and straightforward

2. Centrality and effect-size weighted analysis (use_weighting=True):
   - Combined Weight = TF_centrality × normalized_|aging_slope| × normalized_|drug_slope|
   - Weights each TF by:
     * Network centrality (out-degree, pagerank, betweenness) - hub importance [0-1]
     * Normalized magnitude of aging effect - how much aging changes this TF [0-1]
     * Normalized magnitude of drug effect - how much the drug changes this TF [0-1]
   - All three components are normalized to [0, 1] to prevent any single factor from dominating
   - Uses permutation test for statistical significance

Key functions:
- compute_network_centrality(): Calculate centrality metrics for TFs
- compute_weighted_tf_overlap(): Weight contingency table by centrality × slopes (or standard counts)
- calculate_weighted_reversal_score(): Reversal score using weights (or counts)
- weighted_permutation_test(): Statistical test for weighted scores
- analyze_drug_reversal(): Main analysis function with use_weighting parameter

The weighted approach prioritizes drugs that strongly reverse changes in important hub TFs.
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import os
from typing import Tuple, Dict, List
from ciim.src.common import SAVE_DIR
from ciim.src.utils.util import retrieve_net_consensus


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
    score_threshold: float = 0
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
    sig_threshold: float = 0.05,
    centrality_weight: str = 'composite_centrality',
    datasets_for_network: List[str] = None,
    min_network_degree: int = 3,
    use_weighting: bool = True
) -> pd.DataFrame:
    """
    Run drug reversal analysis for all drugs and cell types.
    
    This uses network centrality and effect sizes to weight TFs, prioritizing drugs 
    that reverse changes in hub regulators over peripheral TFs.
    
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
    centrality_weight : str
        Centrality metric to use for weighting (default: 'composite_centrality')
        Ignored if use_weighting=False
    datasets_for_network : List[str], optional
        Datasets to use for consensus network
        Ignored if use_weighting=False
    min_network_degree : int
        Minimum degree for network consensus
        Ignored if use_weighting=False
    use_weighting : bool
        If True, uses centrality and effect size weighting with permutation test
        If False, uses standard Fisher's exact test without any weighting
    
    Returns
    -------
    pd.DataFrame
        Results with weighted or standard metrics depending on use_weighting
    """
    from ciim.src.common import datasets_all
    if datasets_for_network is None:
        datasets_for_network = datasets_all
    
    results = []
    
    # Compute centrality for each cell type (only if using weighting)
    centrality_cache = {}
    if use_weighting:
        print("Computing network centrality metrics...")
        for cell_type in cell_types:
            print(f"  {cell_type}...")
            centrality_cache[cell_type] = compute_network_centrality(
                cell_type=cell_type,
                datasets=datasets_for_network,
                min_degree=min_network_degree
            )
    else:
        print("Using standard Fisher's exact test (no weighting)...")
        # Create dummy centrality dataframes (won't be used)
        for cell_type in cell_types:
            centrality_cache[cell_type] = pd.DataFrame()
    
    # Get unique drugs
    drugs = drug_stats['comparision'].unique()
    
    print(f"\nAnalyzing {len(drugs)} drugs across {len(cell_types)} cell types...")
    
    for cell_type in cell_types:
        centrality_df = centrality_cache[cell_type]
        
        for drug in drugs:
            # Skip if no data for this combination
            drug_ct_data = drug_stats[
                (drug_stats['cell_type'] == cell_type) & 
                (drug_stats['comparision'] == drug)
            ]
            
            if len(drug_ct_data) == 0:
                continue
            
            # Compute contingency (weighted or standard)
            contingency = compute_weighted_tf_overlap(
                aging_stats=aging_stats,
                drug_stats=drug_stats,
                centrality_df=centrality_df,
                cell_type=cell_type,
                drug_name=drug,
                sig_threshold=sig_threshold,
                weight_column=centrality_weight,
                use_weighting=use_weighting
            )
            
            # Calculate reversal score
            weighted_score = calculate_weighted_reversal_score(contingency)
            
            # Perform statistical test
            if use_weighting:
                # Weighted chi-square test (proper test for weighted contingency tables)
                p_value, effect_size = weighted_chi_square_test(contingency)
            else:
                # Standard Fisher's exact test
                try:
                    # fisher_exact returns (odds_ratio, p_value) - NOTE THE ORDER!
                    odds_ratio, p_value = fisher_exact(
                        [[int(contingency['a_w']), int(contingency['c_w'])],
                         [int(contingency['b_w']), int(contingency['d_w'])]],
                        alternative='two-sided'
                    )
                    effect_size = odds_ratio
                except (ValueError, ZeroDivisionError):
                    # Handle edge cases where Fisher's test fails
                    p_value = 1.0
                    effect_size = 1.0
            
            # Store results with both weighted and standard columns for compatibility
            result_dict = {
                'drug': drug,
                'cell_type': cell_type,
                'dataset': dataset,
                'n_common': contingency['n_common'],
                'total_centrality_weight': contingency['total_weight'],
                'avg_tf_centrality': contingency['avg_centrality'],
                'n_reversal': int(contingency['b_w'] + contingency['c_w']),
                'reversal_weight': contingency['b_w'] + contingency['c_w'],
                'acceleration_weight': contingency['a_w'] + contingency['d_w'],
                'reversal_score': weighted_score,
                'fisher_pvalue': p_value,
                'effect_size': effect_size,
                'a_weighted': contingency['a_w'],
                'b_weighted': contingency['b_w'],
                'c_weighted': contingency['c_w'],
                'd_weighted': contingency['d_w']
            }
            
            # Add standard counts (a, b, c, d) for plotting compatibility
            if not use_weighting:
                result_dict.update({
                    'a': int(contingency['a_w']),
                    'b': int(contingency['b_w']),
                    'c': int(contingency['c_w']),
                    'd': int(contingency['d_w'])
                })
            
            results.append(result_dict)
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        return results_df
    
    # Clean p-values: replace inf and NaN with 1.0 (non-significant)
    results_df['fisher_pvalue'] = results_df['fisher_pvalue'].replace([np.inf, -np.inf], 1.0)
    results_df['fisher_pvalue'] = results_df['fisher_pvalue'].fillna(1.0)
    
    # Apply FDR correction
    results_df['fisher_pvalue_adj'] = multipletests(
        results_df['fisher_pvalue'],
        method='fdr_bh'
    )[1]
    
    # Classify drugs based on score and p-value
    # For standard analysis: any positive score = reversal, negative = acceleration
    # For weighted analysis: can use stricter thresholds if desired
    def classify_drug(row, p_thresh=0.05, score_thresh=0.0):
        if row['fisher_pvalue_adj'] >= p_thresh:
            return 'neutral'
        if row['reversal_score'] > score_thresh:
            return 'rejuvenating'
        elif row['reversal_score'] < -score_thresh:
            return 'accelerating'
        else:
            return 'neutral'
    
    results_df['classification'] = results_df.apply(classify_drug, axis=1)
    
    # Add effect_type for compatibility
    results_df['effect_type'] = results_df['reversal_score'].apply(
        lambda x: 'reversal' if x > 0 else ('acceleration' if x < 0 else 'neutral')
    )
    
    # Sort by weighted reversal score
    results_df = results_df.sort_values('reversal_score', ascending=False)
    
    return results_df


def filter_rejuvenating_drugs(
    results_df: pd.DataFrame,
    p_threshold: float = 0.05,
    min_reversal_score: float = 0.1,
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
    
    return filtered


# ============================================================================
# Network Centrality-Based Functions
# ============================================================================

def compute_network_centrality(
    cell_type: str,
    datasets: List[str] = None,
    min_degree: int = 3,
    centrality_metrics: List[str] = None
) -> pd.DataFrame:
    """
    Compute network centrality metrics for TFs in the consensus network.
    
    Parameters
    ----------
    cell_type : str
        Cell type to analyze
    datasets : List[str], optional
        Datasets to use for consensus network (default: all)
    min_degree : int
        Minimum degree for consensus network filtering
    centrality_metrics : List[str], optional
        Centrality metrics to compute. Options:
        - 'out_degree': number of outgoing edges (TF targets)
        - 'betweenness': betweenness centrality
        - 'pagerank': PageRank centrality
        Default: ['out_degree', 'pagerank', 'betweenness']
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: tf, out_degree, pagerank, betweenness
        Plus normalized versions and composite_centrality
    """
    if centrality_metrics is None:
        centrality_metrics = ['out_degree', 'pagerank', 'betweenness']
    
    # Retrieve consensus network
    from ciim.src.common import datasets_all
    if datasets is None:
        datasets = datasets_all
    
    net = retrieve_net_consensus(
        datasets=datasets,
        cell_type=cell_type,
        min_degree=min_degree
    )
    
    # Create directed graph
    G = nx.DiGraph()
    for _, row in net.iterrows():
        G.add_edge(row['source'], row['target'], weight=abs(row['weight']))
    
    # Get all TFs (sources in the network)
    all_tfs = list(set(net['source']))
    
    # Initialize results
    centrality_data = {'tf': all_tfs}
    
    # Compute centrality metrics
    for metric in centrality_metrics:
        if metric == 'out_degree':
            # Number of targets regulated by each TF
            out_deg = dict(G.out_degree())
            centrality_data['out_degree'] = [out_deg.get(tf, 0) for tf in all_tfs]
        
        elif metric == 'betweenness':
            # Betweenness centrality
            between = nx.betweenness_centrality(G, weight='weight')
            centrality_data['betweenness'] = [between.get(tf, 0) for tf in all_tfs]
        
        elif metric == 'pagerank':
            # PageRank centrality
            pr = nx.pagerank(G, weight='weight')
            centrality_data['pagerank'] = [pr.get(tf, 0) for tf in all_tfs]
    
    # Create dataframe
    df = pd.DataFrame(centrality_data)
    
    # Normalize each metric to [0, 1]
    for metric in centrality_metrics:
        if metric in df.columns:
            max_val = df[metric].max()
            if max_val > 0:
                df[f'{metric}_norm'] = df[metric] / max_val
            else:
                df[f'{metric}_norm'] = 0
    
    # Compute composite centrality (average of normalized metrics)
    norm_cols = [f'{m}_norm' for m in centrality_metrics if m in df.columns]
    if norm_cols:
        df['composite_centrality'] = df[norm_cols].mean(axis=1)
    else:
        df['composite_centrality'] = 0
    
    return df


def compute_weighted_tf_overlap(
    aging_stats: pd.DataFrame,
    drug_stats: pd.DataFrame,
    centrality_df: pd.DataFrame,
    cell_type: str,
    drug_name: str,
    sig_threshold: float = 0.05,
    weight_column: str = 'composite_centrality',
    use_weighting: bool = False
) -> Dict[str, float]:
    """
    Compute weighted overlap using network centrality AND effect sizes.
    
    Weight = centrality × normalized_|aging_slope| × normalized_|drug_slope|
    
    This weights TFs by:
    1. Network importance (centrality) [0-1]
    2. Normalized magnitude of aging effect [0-1]
    3. Normalized magnitude of drug effect [0-1]
    
    All components are normalized to prevent any single factor from dominating.
    
    If use_weighting=False, returns standard counts (a, b, c, d) without any weighting.
    
    Parameters
    ----------
    aging_stats : pd.DataFrame
        Aging statistics (must have 'slope' column)
    drug_stats : pd.DataFrame
        Drug statistics (must have 'slope_condition' column)
    centrality_df : pd.DataFrame
        TF centrality metrics (ignored if use_weighting=False)
    cell_type : str
        Cell type
    drug_name : str
        Drug/comparison name
    sig_threshold : float
        Significance threshold
    weight_column : str
        Column from centrality_df to use as weights (ignored if use_weighting=False)
    use_weighting : bool
        If False, returns standard unweighted counts for Fisher's exact test
    
    Returns
    -------
    Dict[str, float]
        Weighted contingency table with keys: a_w, b_w, c_w, d_w, n_common
        If use_weighting=False, a_w=a, b_w=b, c_w=c, d_w=d (integer counts)
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
        return {'a_w': 0, 'b_w': 0, 'c_w': 0, 'd_w': 0, 'n_common': 0, 'total_weight': 0, 'avg_centrality': 0}
    
    # Merge with slopes
    merged = pd.merge(
        aging_sig[['tf', 'direction', 'slope']],
        drug_sig[['tf', 'direction', 'slope_condition']],
        on='tf',
        suffixes=('_age', '_drug')
    )
    
    # Calculate standard counts (always needed)
    a = len(merged[(merged['direction_age'] == 'increase') & (merged['direction_drug'] == 'increase')])
    b = len(merged[(merged['direction_age'] == 'increase') & (merged['direction_drug'] == 'decrease')])
    c = len(merged[(merged['direction_age'] == 'decrease') & (merged['direction_drug'] == 'increase')])
    d = len(merged[(merged['direction_age'] == 'decrease') & (merged['direction_drug'] == 'decrease')])
    
    # If not using weighting, return standard counts
    if not use_weighting:
        return {
            'a_w': float(a),
            'b_w': float(b),
            'c_w': float(c),
            'd_w': float(d),
            'n_common': len(common_tfs),
            'total_weight': float(a + b + c + d),
            'avg_centrality': 0.0
        }
    
    # Merge with centrality for weighted analysis
    merged = pd.merge(merged, centrality_df[['tf', weight_column]], on='tf', how='left')
    
    # Fill missing centrality with minimum value
    merged[weight_column] = merged[weight_column].fillna(merged[weight_column].min())
    
    # Normalize slopes to [0, 1] to put them on same scale as centrality
    aging_slope_abs = merged['slope'].abs()
    drug_slope_abs = merged['slope_condition'].abs()
    
    aging_slope_norm = aging_slope_abs / aging_slope_abs.max() if aging_slope_abs.max() > 0 else aging_slope_abs
    drug_slope_norm = drug_slope_abs / drug_slope_abs.max() if drug_slope_abs.max() > 0 else drug_slope_abs
    
    # Compute combined weight: centrality × normalized_aging_slope × normalized_drug_slope
    # All three components are now in [0, 1] range
    merged['combined_weight'] = (
        merged[weight_column] * 
        aging_slope_norm * 
        drug_slope_norm
    )
    
    # Compute weighted counts using combined weight
    a_w = merged[
        (merged['direction_age'] == 'increase') & 
        (merged['direction_drug'] == 'increase')
    ]['combined_weight'].sum()
    
    b_w = merged[
        (merged['direction_age'] == 'increase') & 
        (merged['direction_drug'] == 'decrease')
    ]['combined_weight'].sum()
    
    c_w = merged[
        (merged['direction_age'] == 'decrease') & 
        (merged['direction_drug'] == 'increase')
    ]['combined_weight'].sum()
    
    d_w = merged[
        (merged['direction_age'] == 'decrease') & 
        (merged['direction_drug'] == 'decrease')
    ]['combined_weight'].sum()
    
    total_weight = a_w + b_w + c_w + d_w
    
    # Average centrality (not combined weight, for interpretability)
    avg_centrality = merged[weight_column].mean()
    
    return {
        'a_w': a_w,
        'b_w': b_w,
        'c_w': c_w,
        'd_w': d_w,
        'n_common': len(common_tfs),
        'total_weight': total_weight,
        'avg_centrality': avg_centrality
    }


def calculate_weighted_reversal_score(contingency: Dict[str, float]) -> float:
    """
    Calculate weighted reversal score using centrality weights.
    
    Score = (reversal_weight - acceleration_weight) / total_weight
    Ranges from -1 (complete acceleration) to +1 (complete reversal)
    
    Parameters
    ----------
    contingency : Dict[str, float]
        Weighted contingency from compute_weighted_tf_overlap
    
    Returns
    -------
    float
        Weighted reversal score
    """
    a_w = contingency['a_w']
    b_w = contingency['b_w']
    c_w = contingency['c_w']
    d_w = contingency['d_w']
    
    reversal_weight = b_w + c_w
    acceleration_weight = a_w + d_w
    total_weight = contingency['total_weight']
    
    if total_weight == 0:
        return 0.0
    
    score = (reversal_weight - acceleration_weight) / total_weight
    
    return score


def weighted_chi_square_test(
    contingency: Dict[str, float]
) -> Tuple[float, float]:
    """
    Perform chi-square test for weighted contingency tables.
    
    This is the proper statistical test for weighted 2x2 tables, analogous to
    Fisher's exact test for unweighted tables. It tests independence while
    accounting for the marginal totals and sample size.
    
    Parameters
    ----------
    contingency : Dict[str, float]
        Weighted contingency table with keys: a_w, b_w, c_w, d_w
    
    Returns
    -------
    Tuple[float, float]
        (p_value, chi_square_statistic)
    """
    from scipy.stats import chi2
    
    a_w = contingency['a_w']
    b_w = contingency['b_w']
    c_w = contingency['c_w']
    d_w = contingency['d_w']
    
    # Total weight
    total = a_w + b_w + c_w + d_w
    
    if total == 0:
        return 1.0, 0.0
    
    # Marginal totals
    row1_total = a_w + c_w  # Drug increases
    row2_total = b_w + d_w  # Drug decreases
    col1_total = a_w + b_w  # Age increases
    col2_total = c_w + d_w  # Age decreases
    
    # Expected values under independence
    e_a = (row1_total * col1_total) / total if total > 0 else 0
    e_b = (row2_total * col1_total) / total if total > 0 else 0
    e_c = (row1_total * col2_total) / total if total > 0 else 0
    e_d = (row2_total * col2_total) / total if total > 0 else 0
    
    # Handle edge cases
    if e_a == 0 or e_b == 0 or e_c == 0 or e_d == 0:
        # If any expected value is 0, use a small epsilon to avoid division by zero
        # This typically means the table is very sparse
        epsilon = 1e-10
        e_a = max(e_a, epsilon)
        e_b = max(e_b, epsilon)
        e_c = max(e_c, epsilon)
        e_d = max(e_d, epsilon)
    
    # Chi-square statistic
    chi2_stat = (
        ((a_w - e_a) ** 2) / e_a +
        ((b_w - e_b) ** 2) / e_b +
        ((c_w - e_c) ** 2) / e_c +
        ((d_w - e_d) ** 2) / e_d
    )
    
    # P-value from chi-square distribution with 1 degree of freedom
    p_value = 1 - chi2.cdf(chi2_stat, df=1)
    
    return p_value, chi2_stat


def analyze_drug_reversal_with_centrality(
    aging_stats: pd.DataFrame,
    drug_stats: pd.DataFrame,
    cell_types: List[str],
    dataset: str,
    sig_threshold: float = 0.05,
    centrality_weight: str = 'composite_centrality',
    datasets_for_network: List[str] = None,
    min_network_degree: int = 3,
    include_standard: bool = True
) -> pd.DataFrame:
    """
    Run drug reversal analysis with network centrality weighting.
    
    This extends the standard analysis by weighting TFs by their network centrality,
    giving more importance to hub regulators in the reversal score.
    
    Parameters
    ----------
    aging_stats : pd.DataFrame
        Aging statistics
    drug_stats : pd.DataFrame
        Drug statistics
    cell_types : List[str]
        Cell types to analyze
    dataset : str
        Dataset name
    sig_threshold : float
        Significance threshold for individual TF changes
    centrality_weight : str
        Centrality metric to use for weighting
    datasets_for_network : List[str], optional
        Datasets to use for consensus network
    min_network_degree : int
        Minimum degree for network consensus
    include_standard : bool
        If True, also compute standard (unweighted) metrics
    
    Returns
    -------
    pd.DataFrame
        Results with both standard and weighted metrics
    """
    from ciim.src.common import datasets_all
    if datasets_for_network is None:
        datasets_for_network = datasets_all
    
    results = []
    
    # Compute centrality for each cell type
    print("Computing network centrality metrics...")
    centrality_cache = {}
    for cell_type in cell_types:
        print(f"  {cell_type}...")
        centrality_cache[cell_type] = compute_network_centrality(
            cell_type=cell_type,
            datasets=datasets_for_network,
            min_degree=min_network_degree
        )
    
    # Get unique drugs
    drugs = drug_stats['comparision'].unique()
    
    print(f"\nAnalyzing {len(drugs)} drugs across {len(cell_types)} cell types...")
    
    for cell_type in cell_types:
        centrality_df = centrality_cache[cell_type]
        
        for drug in drugs:
            # Skip if no data for this combination
            drug_ct_data = drug_stats[
                (drug_stats['cell_type'] == cell_type) & 
                (drug_stats['comparision'] == drug)
            ]
            
            if len(drug_ct_data) == 0:
                continue
            
            result = {
                'drug': drug,
                'cell_type': cell_type,
                'dataset': dataset
            }
            
            # Standard analysis
            if include_standard:
                contingency_std = compute_tf_overlap(
                    aging_stats, drug_stats, cell_type, drug, sig_threshold
                )
                p_val_std, odds_std, effect_std = fisher_exact_test_reversal(contingency_std)
                score_std = calculate_reversal_score(contingency_std)
                
                result.update({
                    'n_common': contingency_std['n_common'],
                    'n_reversal': contingency_std['b'] + contingency_std['c'],
                    'n_acceleration': contingency_std['a'] + contingency_std['d'],
                    'reversal_score': score_std,
                    'fisher_pvalue': p_val_std,
                    'odds_ratio': odds_std,
                    'effect_type': effect_std,
                    'a': contingency_std['a'],
                    'b': contingency_std['b'],
                    'c': contingency_std['c'],
                    'd': contingency_std['d']
                })
            
            # Weighted analysis
            contingency_w = compute_weighted_tf_overlap(
                aging_stats=aging_stats,
                drug_stats=drug_stats,
                centrality_df=centrality_df,
                cell_type=cell_type,
                drug_name=drug,
                sig_threshold=sig_threshold,
                weight_column=centrality_weight
            )
            
            weighted_score = calculate_weighted_reversal_score(contingency_w)
            p_val_w, effect_w = weighted_chi_square_test(contingency_w)
            
            result.update({
                'total_centrality_weight': contingency_w['total_weight'],
                'avg_tf_centrality': contingency_w['avg_centrality'],
                'reversal_weight': contingency_w['b_w'] + contingency_w['c_w'],
                'acceleration_weight': contingency_w['a_w'] + contingency_w['d_w'],
                'weighted_reversal_score': weighted_score,
                'weighted_pvalue': p_val_w,
                'weighted_effect_size': effect_w,
                'a_weighted': contingency_w['a_w'],
                'b_weighted': contingency_w['b_w'],
                'c_weighted': contingency_w['c_w'],
                'd_weighted': contingency_w['d_w']
            })
            
            results.append(result)
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        return results_df
    
    # Apply FDR correction
    if include_standard:
        results_df['fisher_pvalue_adj'] = multipletests(
            results_df['fisher_pvalue'],
            method='fdr_bh'
        )[1]
        
        results_df['classification'] = results_df.apply(
            lambda row: classify_drug_effect(
                row['fisher_pvalue_adj'],
                row['effect_type'],
                row['reversal_score']
            ),
            axis=1
        )
    
    results_df['weighted_pvalue_adj'] = multipletests(
        results_df['weighted_pvalue'],
        method='fdr_bh'
    )[1]
    
    def classify_weighted(row, p_thresh=0.05, score_thresh=0.1):
        if row['weighted_pvalue_adj'] >= p_thresh:
            return 'neutral'
        if row['weighted_reversal_score'] > score_thresh:
            return 'rejuvenating'
        elif row['weighted_reversal_score'] < -score_thresh:
            return 'accelerating'
        else:
            return 'neutral'
    
    results_df['weighted_classification'] = results_df.apply(classify_weighted, axis=1)
    
    # Sort by weighted reversal score
    results_df = results_df.sort_values('weighted_reversal_score', ascending=False)
    
    return results_df


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
    Print a summary of drug reversal analysis results.
    
    Detects whether results are weighted or standard based on column presence.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from analyze_drug_reversal
    dataset : str
        Dataset name
    """
    # Detect if this is weighted analysis
    is_weighted = 'total_centrality_weight' in results_df.columns and results_df['total_centrality_weight'].sum() > 0
    
    print(f"\n{'='*80}")
    if is_weighted:
        print(f"CENTRALITY-WEIGHTED DRUG REVERSAL ANALYSIS - {dataset.upper()}")
    else:
        print(f"STANDARD DRUG REVERSAL ANALYSIS (Fisher's Exact Test) - {dataset.upper()}")
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
        if is_weighted:
            print(f"TOP REJUVENATING DRUGS (by centrality-weighted reversal score):")
        else:
            print(f"TOP REJUVENATING DRUGS (by reversal score):")
        print(f"{'-'*80}")
        for idx, row in rejuvenating.head(10).iterrows():
            print(f"\n  Drug: {row['drug']}")
            print(f"  Cell type: {row['cell_type']}")
            print(f"  Reversal score: {row['reversal_score']:.3f}")
            print(f"  P-value (adj): {row['fisher_pvalue_adj']:.4e}")
            print(f"  Common TFs: {row['n_common']}")
            if is_weighted:
                print(f"  Total centrality weight: {row['total_centrality_weight']:.2f}")
                print(f"  Avg TF centrality: {row['avg_tf_centrality']:.3f}")
                print(f"  Reversal weight: {row['reversal_weight']:.2f}")
    else:
        print("\nNo rejuvenating drugs found with current thresholds.")
    
    # Statistics
    if len(results_df) > 0:
        print(f"\n{'-'*80}")
        print(f"STATISTICS:")
        print(f"{'-'*80}")
        print(f"  Mean reversal score: {results_df['reversal_score'].mean():.3f}")
        print(f"  Median reversal score: {results_df['reversal_score'].median():.3f}")
        if is_weighted:
            print(f"  Mean total centrality weight: {results_df['total_centrality_weight'].mean():.2f}")
        print(f"  Mean common TFs: {results_df['n_common'].mean():.1f}")
        print(f"  Significant (p<0.05): {(results_df['fisher_pvalue_adj'] < 0.05).sum()}")
    
    print(f"\n{'='*80}\n")
