"""
Unified analysis runner for all feature association analyses.

This module handles:
1. Single-cohort condition analysis (disease/perturbation)
2. Multi-cohort aging analysis with meta-analysis

Replaces: separate disease, perturbation, and script.py files.
"""

import argparse
import sys
import warnings

from hiara.src.config import FEATURES_DIR, MAJOR_CTS, DISCOVERY_COHORTS, DATA_TYPES, FEATURE_TYPES, get_config, META_MIN_COHORT, CONFIG_FA
from hiara.src.feature_association.helper import (
    wrapper_tf_activity,
    wrapper_tfa_peg,
    wrapper_aging_hallmarks,
    wrapper_genesets_scores,
    wrapper_association_with_age_condition,
    wrapper_meta_analysis,
    retrieve_sig_stats, 
    write_features_stats,
    wrapper_ct_freq,
    wrapper_ct_pol_dist,
    wrapper_cc_communication
)

warnings.filterwarnings("ignore")


def calculate_features(analysis_name, par):
    """Calculate features based on type."""
    if analysis_name in ['tfa_major_b', 'tfa_sub_b']:
        wrapper_tf_activity(analysis_name, par)
    elif analysis_name in ['ge_sub_b', 'ge_major_b']:
        pass
    elif analysis_name == 'gene_score':
        wrapper_genesets_scores(par)
    elif analysis_name == 'aging_hallmarks':
        wrapper_aging_hallmarks(par)
    elif analysis_name == 'tfa_peg':
        wrapper_tfa_peg(par)
    elif analysis_name == 'ct_freq':
        wrapper_ct_freq(par)
    elif analysis_name == 'ct_pol_dist':
        wrapper_ct_pol_dist(par)
    elif analysis_name == 'ccc_sub_b':
        wrapper_cc_communication(analysis_name, par, n_jobs=1)
    else:
        raise ValueError(f"Unknown analysis_name: {analysis_name}")


def run_single_cohort_analysis(args):

    # Get configuration(s) - may be single or multiple
    dataset = args.datasets[0]
    
    skip_features = args.skip_features
    config = get_config(dataset)
    
    # Get analysis configuration from CONFIG_FA
    if args.analysis_name not in CONFIG_FA:
        raise ValueError(f"Unknown analysis name: {args.analysis_name}. Available: {list(CONFIG_FA.keys())}")    
    print("\n" + "=" * 80)
    print(f"Analysis type: {args.association_type}")
    print(f"Analysis name: {args.analysis_name}")


    print("=" * 80 + "\n")
    
    # Prepare parameters
    par = {
        **args.__dict__,
        'cell_types': args.cell_types if args.cell_types != ['all'] else MAJOR_CTS,
        'association_type': args.association_type,
        'use_consensus_net': True,
        'promotor_only': args.promotor_only,
        'condition': config.treatment_groups if hasattr(config, 'treatment_groups') else None
    }
    
    # Step 1: Calculate features (if needed) - only once for all configs
    if not skip_features:  # ct_freq is fast to compute, no need to skip
        print("\n[1/3] Calculating features...")
        calculate_features(args.analysis_name, par)
        print("✓ Features calculated")
    else:
        print("\n[1/3] Skipping feature calculation (using cached data)")
    
    # Step 2: Compute condition statistics for each config
    print(f"\n[2/3] Computing condition statistics...")    
    condition_stats = wrapper_association_with_age_condition(
        analysis_name=args.analysis_name,
        par=par,
        features=None,
        test_type=config.test_type,
        condition=None,
        config=config,
        association_type=args.association_type
    )
    
    write_features_stats(
        stats=condition_stats,
        analysis_name=args.analysis_name,
        multi_cohort=False,
        dataset=dataset
    )


def run_multi_cohort_analysis(
    args
):
    # Get analysis configuration from CONFIG_FA
    if args.analysis_name not in CONFIG_FA:
        raise ValueError(f"Unknown analysis name: {args.analysis_name}. Available: {list(CONFIG_FA.keys())}")
    
    analysis_config = CONFIG_FA[args.analysis_name]
    data_type = analysis_config['data_type']
    granularity = analysis_config['granularity']
    
    print("\n" + "=" * 80)
    print(f"MULTI-COHORT AGING ANALYSIS")
    print(f"Analysis name: {args.analysis_name}")
    print(f"Data type: {data_type}")
    print(f"Granularity: {granularity}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Cell types: {', '.join(args.cell_types)}")
    print(f"Promotor-based only: {args.promotor_only}")
    print("=" * 80 + "\n")
    
    suffix = '_promotor' if args.promotor_only else ''
    
    # Prepare parameters
    par = {
        **args.__dict__,
        'temp_dir': f'{FEATURES_DIR}/tmp/',
        'META_MIN_COHORT': META_MIN_COHORT,
        'condition': 'healthy',
        'use_consensus_net': True
    }
   
    # Step 1: Calculate features
    if not args.skip_features and args.analysis_name not in ['sub_tf_markers']:
        print("\n[1/3] Calculating features...")
        calculate_features(args.analysis_name, par)
        print("✓ Features calculated")
    else:
        print("\n[1/3] Skipping feature calculation (using cached data)")
    
    # Step 2: Calculate association with age or identify markers
    if args.analysis_name == 'sub_tf_markers':
        print("\n[2/3] Identifying sub cell type markers...")
        from hiara.src.feature_association.helper import wrapper_sub_celltype_markers
        stats_features = wrapper_sub_celltype_markers(args.analysis_name, par)
    else:
        print("\n[2/3] Computing associations with age...")
        stats_features = wrapper_association_with_age_condition(args.analysis_name, par, association_type=args.association_type)
    
    # Step 3: Meta-analysis (discovery/validation)
    print("\n[3/3] Running meta-analysis...")
    stats = wrapper_meta_analysis(analysis_name=args.analysis_name, stats_features=stats_features, par=par)
    write_features_stats(
        stats=stats,
        analysis_name=args.analysis_name,
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
        '--analysis-name',
        type=str,
        required=True,
        choices=list(CONFIG_FA.keys()),
        help=f'Analysis configuration name from CONFIG_FA. Available: {list(CONFIG_FA.keys())}'
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
        default=MAJOR_CTS,
        help='Cell types to analyze (default: CD4T CD8T)'
    )
    
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run in test mode with reduced data for quick testing'
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
                                   analysis_name=args.analysis_name).drop_duplicates(subset=['gene', 'cell_type', 'condition'])
    print(f"\nSignificant features (FDR < 0.05) in meta-analysis:")
    print(stats_sig.groupby(['cell_type', 'condition'])['gene'].nunique())
    
    return 0
    
if __name__ == '__main__':
    sys.exit(main())
