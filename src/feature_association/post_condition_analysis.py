#!/usr/bin/env python
"""
Unified Condition Analysis - Post-run visualization script
This script performs post-run analysis for both disease and perturbation experiments,
generating various visualizations including:
- Overview heatmap of cell types
- Overlap analysis with aging genes
- Age-stratified analysis (disease) / Aging-perturbation comparison
- Case TF trends / Donor-level effects (perturbations)
- Pathway analysis
"""
import argparse
import os
import warnings
import matplotlib.pyplot as plt

from hira import retrieve_stats
from hira.src.feature_association.plots_groups import (
    wrapper_plots_tfa_major_b_condition,
    wrapper_plots_tfa_sub_b_condition,
    wrapper_plots_gene_expression_condition,
    wrapper_plots_ct_tf_markers_condition,
    wrapper_plots_tfa_peg_condition,
    wrapper_plots_ct_freq_condition,
    wrapper_plots_ct_pol_dist_condition,
    wrapper_plots_ccc_sub_b_condition
)
# Import common utilities and configuration
from hira.src.config import (
    get_config_fa,
    get_available_fa_analyses,
    PLOTS_DIR, 
    MAJOR_CTS
)
from hira.src.config import get_config

warnings.filterwarnings("ignore")
# Set matplotlib defaults
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "Arial"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (e.g., SLE_European, CXCL9, op, parsebioscience, soundlife)'
    )
    parser.add_argument(
        '--analysis-type',
        type=str,
        choices=['disease', 'perturbation', 'aging'],
        required=True,
        help='Type of analysis: disease, perturbation, or aging'
    )
    parser.add_argument(
        '--analysis-name',
        type=str,
        required=True,
        choices=get_available_fa_analyses(),
        help=f'Analysis configuration name. Available: {get_available_fa_analyses()}'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=PLOTS_DIR,
        help='Output directory for plots (default: PLOTS_DIR from common.py)'
    )
    parser.add_argument(
        '--cell-types',
        type=str,
        nargs='+',
        default=['CD4T', 'CD8T'],
        help='Cell types to analyze (default: CD4T CD8T)'
    )
    parser.add_argument(
        '--case-tfs',
        type=str,
        nargs='+',
        default=['LEF1', 'ZEB2'],
        help='Case genes for trend plotting (disease only, default: LEF1 ZEB2)'
    )
    parser.add_argument(
        '--case-cell-type',
        type=str,
        default='CD8T',
        help='Cell type for case TF analysis (disease only, default: CD8T)'
    )
    parser.add_argument(
        '--top-aging-tfs',
        type=int,
        default=15,
        help='Number of top aging genes to show (default: 15)'
    )
    parser.add_argument(
        '--agreement',
        type=str,
        default='same',
        choices=['same', 'opposite'],
        help='Agreement type for overlap analysis (default: same)'
    )
    parser.add_argument(
        '--sig-threshold',
        type=float,
        default=0.05,
        help='Significance threshold for p-values (default: 0.05)'
    )
    parser.add_argument(
        '--skip-pathway',
        action='store_true',
        help='Skip pathway analysis'
    )
    parser.add_argument(
        '--skip-case-studies',
        action='store_true',
        help='Skip case studies (TF trends for disease or donor-level for perturbations)'
    )
    parser.add_argument(
        '--skip-heatmap',
        action='store_true',
        help='Skip aging-experiment heatmap (perturbation only)'
    )
    parser.add_argument(
        '--skip-overview',
        action='store_true',
        help='Skip overview heatmap (perturbation only)'
    )
    parser.add_argument(
        '--association-type',
        type=str,
        default='grouped',
        choices=['grouped', 'continous'],
        help='Association type used in run_analysis: grouped (uses slope_condition) or continous (uses slope). Default: grouped'
    )
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get configuration
    analysis_config = get_config_fa(args.analysis_name)
    feature_type = analysis_config['feature_type']
    data_type = analysis_config['data_type']
    granularity = analysis_config['granularity']
    
    # Add to args for backward compatibility
    args.feature_type = feature_type
    args.data_type = data_type
    args.granularity = granularity
    
    if False:
        print("="*60)
        print(f"Dataset: {args.dataset}")
        print(f"Analysis type: {args.analysis_type}")
        print(f"Analysis name: {args.analysis_name}")
        print(f"Data type: {data_type}")
        print(f"Feature type: {feature_type}")
        print(f"Granularity: {granularity}")
        print("="*60)

    args.case_tfs = ['KLF6', 'PRDM1' ,'LEF1', 'ZEB2']
    stats = retrieve_stats(
        dataset=args.dataset,
        analysis_name=args.analysis_name,
        multi_cohort=False
    )

    stats_sig = stats[stats['is_significant']]

    args.cell_types = MAJOR_CTS if args.cell_types == ['all'] else args.cell_types
    
    config = get_config(dataset=args.dataset)

    # Call the appropriate wrapper based on analysis_name
    if args.analysis_name == 'tfa_major_b':
        wrapper_plots_tfa_major_b_condition(args, stats, stats_sig)
    elif args.analysis_name == 'tfa_sub_b':
        wrapper_plots_tfa_sub_b_condition(args, stats, stats_sig)
    elif args.analysis_name in ('ge_major_b', 'ge_sub_b'):
        wrapper_plots_gene_expression_condition(args, stats, stats_sig)
    elif args.analysis_name == 'ct_tf_markers':
        wrapper_plots_ct_tf_markers_condition(args, stats, stats_sig)
    elif args.analysis_name == 'tfa_peg':
        wrapper_plots_tfa_peg_condition(args, stats, stats_sig)
    elif args.analysis_name == 'ct_freq':
        wrapper_plots_ct_freq_condition(args, stats, stats_sig)
    elif args.analysis_name == 'ct_pol_dist':
        wrapper_plots_ct_pol_dist_condition(args, stats, stats_sig)
    elif args.analysis_name == 'ccc_sub_b':
        wrapper_plots_ccc_sub_b_condition(args, stats, stats_sig)
    else:
        raise ValueError(f'Unknown analysis_name: {args.analysis_name}')
 
  
    
if __name__ == '__main__':
    main()
