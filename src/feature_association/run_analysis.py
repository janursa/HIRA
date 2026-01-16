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

from hiara.src.config import FEATURES_DIR, CELL_TYPES, DISCOVERY_COHORTS, get_config, meta_analysis_min_cohorts
from hiara.src.feature_association.helper import (
    wrapper_tf_activity,
    wrapper_aging_hallmarks,
    wrapper_gene_score,
    wrapper_association_with_age_condition,
    wrapper_meta_analysis,
    retrieve_sig_stats, 
    write_features_stats
)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore")


def run_single_cohort_analysis(args):

    # Get configuration(s) - may be single or multiple
    dataset = args.datasets[0]
    cell_types = args.cell_types
    feature_type = args.feature_type
    data_type = args.data_type
    skip_features = args.skip_features
    config = get_config(dataset)
    
    print("\n" + "=" * 80)
    print(f"Analysis type: {args.association_type}")
    print(f"Dataset: {dataset}")
    print(f"Feature: {feature_type}")
    print(f"Cell types: {', '.join(cell_types)}")

    print("=" * 80 + "\n")
    
    # Prepare parameters
    par = {
        'data_type': data_type,
        'feature_type': feature_type,
        'datasets': [dataset],
        'cell_types': cell_types,
        'association_type': args.association_type,
        'use_consensus_net': True,
    }
    
    # Step 1: Calculate features (if needed) - only once for all configs
    if not skip_features:
        print("\n[1/3] Calculating features...")
        if feature_type == 'tf_activity':
            wrapper_tf_activity(par)
        # elif feature_type == 'gene_expression':
        #     wrapper_gene_expression(par)
        elif feature_type == 'aging_hallmarks':
            wrapper_aging_hallmarks(par)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        print("✓ Features calculated")
    else:
        print("\n[1/3] Skipping feature calculation (using cached data)")
    
    # Step 2: Compute condition statistics for each config
    print(f"\n[2/3] Computing condition statistics...")    
    condition_stats = wrapper_association_with_age_condition(
        par=par,
        features=None,
        test_type=config.test_type,
        condition=None,
        config=config,
        association_type=args.association_type
    )
    
    write_features_stats(
        stats=condition_stats,
        data_type=data_type,
        feature_type=feature_type,
        multi_cohort=False,
        dataset=dataset
    )


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
        'data_type': args.data_type,
        'feature_type': args.feature_type,
        'cell_types': args.cell_types,
        'datasets': args.datasets,
        'temp_dir': f'{FEATURES_DIR}/tmp/',
        'only_promotor': args.promotor_only,
        'meta_analysis_min_cohorts': meta_analysis_min_cohorts,
        'condition': 'healthy',
        'use_consensus_net': True
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
    stats_features = wrapper_association_with_age_condition(par, association_type=args.association_type)
    
    # Step 3: Meta-analysis (discovery/validation)
    print("\n[3/3] Running meta-analysis...")
    stats = wrapper_meta_analysis(stats_features, par)
    write_features_stats(
        stats=stats,
        data_type=args.data_type,
        feature_type=args.feature_type,
        multi_cohort=True,
        suffix=suffix
    )
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
        default=DISCOVERY_COHORTS,
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
    parser.add_argument(
        '--association-type',
        type=str,   
        required=True,
        choices=['continous', 'grouped'],
        help='Type of analysis to perform: condition (disease/perturbation) or aging'
    )
    
    args = parser.parse_args()
    datasets = args.datasets
    assert len(datasets) >= 1, "At least one dataset must be specified."
    if len(datasets) == 1:
        run_single_cohort_analysis(
            args
        )
        multi_cohort = False
        dataset = args.datasets[0]
    
    else:  # multi-cohort
        run_multi_cohort_analysis(
            args
            
        )
        multi_cohort = True

    stats_sig = retrieve_sig_stats(dataset=None if multi_cohort else dataset,
                                   feature_type=args.feature_type,
                                   data_type=args.data_type,
                                   multi_cohort=multi_cohort).drop_duplicates(subset=['gene', 'cell_type', 'condition'])
    print(f"\nSignificant features (FDR < 0.05) in meta-analysis:")
    print(stats_sig.groupby(['cell_type', 'condition'])['gene'].nunique())
    
    return 0
    
if __name__ == '__main__':
    sys.exit(main())
