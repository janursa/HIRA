#!/usr/bin/env python
"""

This script performs post-run analysis for age-associated TFs.
generating various visualizations including:

"""

import argparse
import os
import sys
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
pd.set_option("display.max_columns", None)

# Import common utilities and configuration
from hira.src.config import (
    PLOTS_DIR, 
    MAJOR_CTS, 
    DISCOVERY_COHORTS,
    palette_trend,
    surrogate_names,
    get_config_fa,
    get_available_fa_analyses
)
from hira import retrieve_net_consensus

from hira import retrieve_sig_stats, retrieve_stats, mapping_minor_2_major
from hira.src.feature_association.plots_groups import (
    wrapper_plots_tfa_major_aging,
    wrapper_plots_tfa_sub_aging,
    wrapper_plots_gene_expression_aging,
    wrapper_plots_ct_tf_markers_aging,
    wrapper_plots_tfa_peg_aging,
    wrapper_plots_ct_freq_aging,
    wrapper_plots_ct_pol_dist_aging,
    wrapper_plots_ccc_aging
)


warnings.filterwarnings("ignore")

# Set matplotlib defaults
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "Arial"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-name', type=str, required=True, 
                       choices=get_available_fa_analyses(),
                       help='Analysis configuration name')
    parser.add_argument('--skip-pathway', action='store_true', help='Skip pathway analysis')

    args = parser.parse_args()
    analysis_name = args.analysis_name
    skip_pathway = args.skip_pathway
    print(f"\n{'='*80}")
    print(f"POST-AGING ANALYSIS")
    print(f"Analysis name: {args.analysis_name}")
    print(f"{'='*80}\n")

    stats_features = retrieve_stats(analysis_name=args.analysis_name)
    stats_features_sig = retrieve_sig_stats(analysis_name=args.analysis_name)

    if analysis_name in ['tfa_major_b', 'tfa_major_mc', 'tfa_major_sc']:
        wrapper_plots_tfa_major_aging(args, stats_features, stats_features_sig, skip_pathway)
    elif analysis_name in ['tfa_sub_b']:
        wrapper_plots_tfa_sub_aging(args, stats_features, stats_features_sig, skip_pathway)
    elif analysis_name in ['ge_major_b']:
        wrapper_plots_gene_expression_aging(args, stats_features, stats_features_sig, skip_pathway)
    elif analysis_name in ['ct_tf_markers']:
        wrapper_plots_ct_tf_markers_aging(args, stats_features, stats_features_sig, skip_pathway)
    elif analysis_name in ['tfa_peg']:
        wrapper_plots_tfa_peg_aging(args, stats_features, stats_features_sig, skip_pathway)
    elif analysis_name in ['ct_freq']:
        wrapper_plots_ct_freq_aging(args, stats_features, stats_features_sig, skip_pathway)
    elif analysis_name == 'ct_pol_dist':
        wrapper_plots_ct_pol_dist_aging(args, stats_features, stats_features_sig, skip_pathway)
    elif analysis_name in ['ccc_major_b', 'ccc_sub_b']:
        wrapper_plots_ccc_aging(args, stats_features, stats_features_sig, skip_pathway)
    else:
        raise ValueError('Unknown feature type')