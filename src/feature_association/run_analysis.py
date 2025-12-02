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
    
    Parameters
    ----------
    dataset : str
        Dataset name (e.g., 'SLE_European', 'op', 'CXCL9')
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
        Results dictionary with 'condition_stats' and 'aging_stats'
    """
    # Get configuration
    config = get_config(dataset)
    
    print("\n" + "=" * 80)
    print(f"CONDITION ANALYSIS: {config.display_name}")
    print(f"Type: {config.analysis_type.upper()}")
    print(f"Dataset: {dataset}")
    print(f"Feature: {feature_type}")
    print(f"Cell types: {', '.join(cell_types)}")
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
    
    # Step 1: Calculate features (if needed)
    if not skip_features:
        print("\n[1/3] Calculating features...")
        if feature_type == 'tf_activity':
            wrapper_tf_activity(par)
        elif feature_type == 'gene_expression':
            wrapper_gene_expression(par)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        print("✓ Features calculated")
    else:
        print("\n[1/3] Skipping feature calculation (using cached data)")
    
    # Step 2: Compute condition statistics
    print("\n[2/3] Computing condition statistics...")
    condition_stats = wrapper_association_with_age_condition(
        par=par,
        features=None,
        test_type=config.test_type,
        condition=None,
        config=config
    )
    
    # Save condition stats
    os.makedirs(f'{SAVE_DIR}/stats', exist_ok=True)
    stats_file = f'{SAVE_DIR}/stats/stats_{dataset}_{data_type}_{feature_type}_{config.test_type}.csv'
    condition_stats.to_csv(stats_file, index=False)
    print(f"✓ Condition stats saved: {stats_file}")
    
    # Step 3: Load aging reference
    print("\n[3/3] Loading aging reference...")
    aging_stats = retrieve_sig_stats(
        type='bulk',
        race='both',
        feature_type=feature_type
    ).drop_duplicates(subset=['cell_type', 'tf' if feature_type == 'tf_activity' else 'target'])
    print(f"✓ Loaded {len(aging_stats)} significant aging features")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80 + "\n")
    
    return {
        'config': config,
        'condition_stats': condition_stats,
        'aging_stats': aging_stats,
        'stats_file': stats_file
    }


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
        choices=['tf_activity', 'gene_expression'],
        default='tf_activity',
        help='Feature type to analyze (default: tf_activity)'
    )
    
    parser.add_argument(
        '--data-type',
        type=str,
        choices=['bulk', 'sc'],
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
        results = run_condition_analysis(
            dataset=args.dataset,
            cell_types=args.cell_types,
            feature_type=args.feature_type,
            data_type=args.data_type,
            skip_features=args.skip_features,
            association_type=args.association_type
        )
        
        print("\nResults:")
        print(f"  Condition stats: {results['stats_file']}")
        print(f"  Significant features: {(results['condition_stats']['p_value_adj'] < 0.05).sum()}")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
