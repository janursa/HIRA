"""
Unified analysis runner for condition-based studies (disease and perturbation).

This module replaces separate disease and perturbation analysis scripts.
"""

import argparse
import os
import sys
from typing import List

from ciim.src.common import SAVE_DIR
from ciim.src.feature_association.config import get_config, list_datasets
from ciim.src.feature_association.helper import (
    wrapper_tf_activity,
    wrapper_gene_expression,
    wrapper_aging_hallmarks,
    wrapper_association_with_age_condition,
    retrieve_sig_stats
)


def run_condition_analysis(
    dataset: str,
    cell_types: List[str],
    feature_type: str = 'tf_activity',
    data_type: str = 'bulk',
    skip_features: bool = False,
    association_type: str = 'spearman'
):
    """
    Run complete condition analysis pipeline.
    
    Handles datasets with multiple configurations (e.g., different data subsets).
    Results from multiple configs are concatenated.
    
    Parameters
    ----------
    dataset : str
        Dataset name (e.g., 'SLE_European', 'op', 'CXCL9', 'soundlife')
    cell_types : List[str]
        Cell types to analyze
    feature_type : str
        'tf_activity' or 'gene_expression'
    data_type : str
        'bulk' or 'sc'
    skip_features : bool
        If True, skip feature calculation (use cached)
    association_type : str
        'spearman' or 'pearson'
    
    Returns
    -------
    dict
        Results dictionary with 'condition_stats'
    """
    # Get configuration(s) - may be single or multiple
    configs = get_config(dataset)
    
    print("\n" + "=" * 80)
    print(f"CONDITION ANALYSIS: {configs[0].display_name}")
    print(f"Type: {configs[0].analysis_type.upper()}")
    print(f"Dataset: {dataset}")
    print(f"Feature: {feature_type}")
    print(f"Cell types: {', '.join(cell_types)}")
    if len(configs) > 1:
        print(f"Number of configs: {len(configs)}")
        for i, cfg in enumerate(configs, 1):
            print(f"  [{i}] {cfg.config_label or f'config_{i}'}: {cfg.display_name}")
    print("=" * 80 + "\n")
    
    # Prepare parameters
    par = {
        'feature_type': feature_type,
        'datasets': [dataset],
        'cell_types': cell_types,
        'type': data_type,
        'cell_type_resolution': 'cell_type',
        'association_type': association_type
    }
    
    # Step 1: Calculate features (if needed) - only once for all configs
    if not skip_features:
        print("\n[1/3] Calculating features...")
        if feature_type == 'tf_activity':
            wrapper_tf_activity(par)
        elif feature_type == 'gene_expression':
            wrapper_gene_expression(par)
        elif feature_type == 'aging_hallmarks':
            wrapper_aging_hallmarks(par)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        print("✓ Features calculated")
    else:
        print("\n[1/3] Skipping feature calculation (using cached data)")
    
    # Step 2: Compute condition statistics for each config
    print(f"\n[2/3] Computing condition statistics ({len(configs)} config(s))...")
    
    all_condition_stats = []
    
    for i, config in enumerate(configs, 1):
        config_label = config.config_label or f"config_{i}"
        
        if len(configs) > 1:
            print(f"\n  [{i}/{len(configs)}] Running: {config_label} ({config.display_name})")
        
        condition_stats = wrapper_association_with_age_condition(
            par=par,
            features=None,
            test_type=config.test_type,
            condition=None,
            config=config
        )
        
        # Add config label to results for tracking
        if len(configs) > 1:
            condition_stats['config_label'] = config_label
        
        all_condition_stats.append(condition_stats)
    
    # Concatenate results from all configs
    import pandas as pd
    if len(all_condition_stats) > 1:
        print(f"\n  Concatenating results from {len(all_condition_stats)} configs...")
        condition_stats_combined = pd.concat(all_condition_stats, ignore_index=True)
    else:
        condition_stats_combined = all_condition_stats[0]
    
    # Save condition stats
    os.makedirs(f'{SAVE_DIR}/stats', exist_ok=True)
    stats_file = f'{SAVE_DIR}/stats/stats_{dataset}_{data_type}_{feature_type}_{configs[0].test_type}.csv'
    condition_stats_combined.to_csv(stats_file, index=False)
    print(f"✓ Condition stats saved: {stats_file}")



def main():
    parser = argparse.ArgumentParser(
        description='Run unified condition analysis (disease or perturbation)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Disease analysis
  python -m ciim.src.feature_association.run_analysis \\
      --dataset SLE_European --cell-types CD4T CD8T --feature-type tf_activity
  
  # Perturbation analysis
  python -m ciim.src.feature_association.run_analysis \\
      --dataset op --cell-types CD4T CD8T --feature-type tf_activity
  
  # Skip feature calculation (use cached)
  python -m ciim.src.feature_association.run_analysis \\
      --dataset SLE_European --cell-types CD4T CD8T --skip-features

Available datasets:
  Disease:      """ + ', '.join(list_datasets('disease')) + """
  Perturbation: """ + ', '.join(list_datasets('perturbation')) + """
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (e.g., SLE_European, op, CXCL9)'
    )
    
    parser.add_argument(
        '--cell-types',
        type=str,
        nargs='+',
        default=['CD4T', 'CD8T'],
        help='Cell types to analyze (default: CD4T CD8T)'
    )
    
    parser.add_argument(
        '--feature-type',
        type=str,
        default='tf_activity',
        help='Feature type to analyze (default: tf_activity)'
    )
    
    parser.add_argument(
        '--data-type',
        type=str,
        choices=['bulk', 'sc', 'metacell'],
        default='bulk',
        help='Data type (default: bulk)'
    )
    
    parser.add_argument(
        '--skip-features',
        action='store_true',
        help='Skip feature calculation (use cached data)'
    )
    
    parser.add_argument(
        '--association-type',
        type=str,
        choices=['spearman', 'pearson'],
        default='spearman',
        help='Association type (default: spearman)'
    )
    
    args = parser.parse_args()
    
    try:
        run_condition_analysis(
            dataset=args.dataset,
            cell_types=args.cell_types,
            feature_type=args.feature_type,
            data_type=args.data_type,
            skip_features=args.skip_features,
            association_type=args.association_type
        )
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
