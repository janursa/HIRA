"""
Unified analysis runner for all feature association analyses.

This module handles:
1. Single-cohort condition analysis (disease/perturbation)
2. Multi-cohort aging analysis with meta-analysis

Replaces: separate disease, perturbation, and script.py files.
"""

import argparse
import os
import sys
from typing import List
import pandas as pd
import warnings

from ciim.src.common import FEATURES_DIR, CELL_TYPES
from ongoing.ciim.src.config import get_config
from ciim.src.feature_association.helper import (
    wrapper_tf_activity,
    wrapper_gene_expression,
    wrapper_aging_hallmarks,
    wrapper_gene_score,
    wrapper_association_with_age_condition,
    wrapper_meta_analysis,
    retrieve_sig_stats
)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore")


def run_single_cohort_analysis(
    args
):

    # Get configuration(s) - may be single or multiple
    configs = get_config(args.dataset)
    
    print("\n" + "=" * 80)
    print(f"CONDITION ANALYSIS: {configs[0].display_name}")
    print(f"Type: {configs[0].analysis_type.upper()}")
    print(f"Dataset: {args.dataset}")
    print(f"Feature: {feature_type}")
    print(f"Cell types: {', '.join(cell_types)}")

    print("=" * 80 + "\n")
    
    # Prepare parameters
    par = {
        'feature_type': feature_type,
        'datasets': [dataset],
        'cell_types': cell_types,
        'type': data_type,
        'cell_type_resolution': 'cell_type'
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
    if len(all_condition_stats) > 1:
        print(f"\n  Concatenating results from {len(all_condition_stats)} configs...")
        condition_stats_combined = pd.concat(all_condition_stats, ignore_index=True)
    else:
        condition_stats_combined = all_condition_stats[0]
    
    # Save condition stats
    os.makedirs(f'{FEATURES_DIR}/stats', exist_ok=True)
    stats_file = f'{FEATURES_DIR}/stats/stats_{dataset}_{data_type}_{feature_type}_{configs[0].test_type}.csv'
    condition_stats_combined.to_csv(stats_file, index=False)
    print(f"✓ Condition stats saved: {stats_file}")


def run_multi_cohort_analysis(
    args
):
    print("\n" + "=" * 80)
    print(f"MULTI-COHORT AGING ANALYSIS")
    print(f"Feature: {args.feature_type}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Cell types: {', '.join(args.cell_types)}")
    print(f"Data type: {args.data_type}")
    print(f"Promotor-based only: {args.promotor_only}")
    print("=" * 80 + "\n")
    
    suffix = '_promotor' if args.promotor_only else ''
    
    # Prepare parameters
    par = {
        'type': args.data_type,
        'feature_type': args.feature_type,
        'cell_types': args.cell_types,
        'datasets': args.datasets,
        'association_type': 'spearman',
        'stats_features': f'{FEATURES_DIR}/{args.feature_type}/stats_features_{args.data_type}{suffix}.csv',
        'stats_all': f'{FEATURES_DIR}/{args.feature_type}/stats_all_{args.data_type}{suffix}.csv',
        'temp_dir': f'{FEATURES_DIR}/tmp/',
        'only_promotor_based': args.promotor_only,
    }
    
    # Step 1: Calculate features
    if not args.skip_features:
        print("\n[1/3] Calculating features...")
        if args.feature_type == 'tf_activity':
            wrapper_tf_activity(par)
        elif args.feature_type == 'gene_expression':
            wrapper_gene_expression(par)
        elif args.feature_type == 'gene_score':
            wrapper_gene_score(par)
        elif args.feature_type == 'aging_hallmarks':
            wrapper_aging_hallmarks(par)
        else:
            raise ValueError(f"Unknown feature type: {args.feature_type}")
    print("✓ Features calculated")
    
    # Step 2: Calculate association with age
    print("\n[2/3] Computing associations with age...")
    stats_features = wrapper_association_with_age_condition(par)
    stats_features.to_csv(par['stats_features'], index=False)
    print(f"✓ Stats saved: {par['stats_features']}")
    
    # Step 3: Meta-analysis (discovery/validation)
    print("\n[3/3] Running meta-analysis...")
    wrapper_meta_analysis(par)
    print(f"✓ Meta-analysis complete: {par['stats_all']}")
    
    print("\n" + "=" * 80)
    print("MULTI-COHORT ANALYSIS COMPLETE")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--analysis-mode',
        type=str,
        choices=['single-cohort', 'multi-cohort'],
        default='single-cohort',
        help='Analysis mode: single dataset or multiple cohorts (default: single-cohort)'
    )
    
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=
        help='List of datasets for multi-cohort mode (e.g., data1 data2 data3)'
    )
    
    parser.add_argument(
        '--cell-types',
        type=str,
        nargs='+',
        default=CELL_TYPES,
        help='Cell types to analyze (default: CD4T CD8T)'
    )
    
    parser.add_argument(
        '--feature-type',
        type=str,
        default='tf_activity',
        choices=['tf_activity', 'gene_expression', 'gene_score', 'aging_hallmarks'],
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
        '--promotor-only',
        action='store_true',
        help='Use promotor-based GRN only (multi-cohort tf_activity only)'
    )
    
    args = parser.parse_args()

    if args.analysis_mode == 'single-cohort':
        assert len(args.datasets) == 1, "--datasets must contain exactly one dataset for single-cohort mode"
        run_single_cohort_analysis(
            args
        )
    
    else:  # multi-cohort
        run_multi_cohort_analysis(
            args
            
        )
    
    return 0
    


if __name__ == '__main__':
    sys.exit(main())
