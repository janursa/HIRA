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
from hiara.src.config import (
    PLOTS_DIR, 
    CELL_TYPES, 
    DISCOVERY_COHORTS,
    palette_trend,
    surrogate_names
)
from hiara import retrieve_net_consensus

from hiara import retrieve_sig_stats, retrieve_stats
from hiara.src.feature_association.plots import wrapper_sig_features_counts, plot_heatmap_overal, plot_central_features, \
    plot_interaction_of_features_between_cell_types, plot_case_tf, gsea_analysis, plot_sig_genes_counts_hallmarks, plot_scatter_feature_vs_age, \
        plot_features_vs_datasets


warnings.filterwarnings("ignore")

# Set matplotlib defaults
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "Arial"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-type', type=str, required=True, help='Feature type to analyze')
    parser.add_argument('--data-type', type=str, default='bulk', required=False, help='Data type to analyze')
    parser.add_argument('--skip-pathway', action='store_true', help='Skip pathway analysis')

    args = parser.parse_args()
    feature_type = args.feature_type
    data_type = args.data_type
    skip_pathway = args.skip_pathway

    stats_aging = retrieve_stats(data_type=data_type, feature_type=feature_type)
    stats_aging_sig = retrieve_sig_stats(data_type=data_type, feature_type=feature_type)

    # - add sig signs
    stats_sig = retrieve_sig_stats(data_type=data_type, feature_type=feature_type)
    tuple_index = stats_sig.set_index(['cell_type', 'gene', 'dataset']).index

    stats_aging = stats_aging.set_index(['cell_type', 'gene', 'dataset'])
    stats_aging['is_significant'] = False
    stats_aging.loc[tuple_index, 'is_significant'] = True

    stats_aging = stats_aging.reset_index()[['gene', 'cell_type', 'dataset', 'slope', 'is_significant']].drop_duplicates(subset=['gene', 'cell_type', 'dataset'])

    if feature_type == 'tf_activity':
        plot_heatmap_overal(stats_aging)
        wrapper_sig_features_counts(args)
        plot_central_features(stats_aging_sig, cell_types=['CD4T', 'CD8T', 'NK', 'MONO'])
        plot_interaction_of_features_between_cell_types(args)
        plot_case_tf(args)
        if not skip_pathway:
            gsea_analysis(stats_sig=stats_aging_sig)
        # plot_scatter_feature_vs_age(args, cell_types=['CD8T'], features=None, feature_selection_mode='top_sig', top_genes=5)
        # plot_sig_networks()
    elif feature_type == 'aging_hallmarks':
        plot_sig_genes_counts_hallmarks()
    elif feature_type == 'gene_expression':
        wrapper_sig_features_counts(args)
        plot_heatmap_overal(stats_aging, feature_type=feature_type)

    elif feature_type in [ 'tfa_traj']:
        # wrapper_sig_features_counts(args)
        # plot_heatmap_overal(stats_aging, feature_type=feature_type) features=['TCF7', 'LEF1', 'GATA3', 'KLF6'], 
        plot_scatter_feature_vs_age(args, cell_types=['CD8T'], features=['TCF7', 'LEF1', 'GATA3', 'KLF6'])
        plot_scatter_feature_vs_age(args, cell_types=['CD8T'], feature_selection_mode='top_central')
        plot_scatter_feature_vs_age(args, cell_types=['CD8T'], feature_selection_mode='top_sig')
        # plot_scatter_feature_vs_age(args, cell_types=['CD8T'], feature_selection_mode='top_sig', top_genes=10)
        plot_features_vs_datasets(cell_type='CD8T', data_type=data_type, feature_type=feature_type, top_features=20)
    else:
        raise ValueError('Unknown feature type')