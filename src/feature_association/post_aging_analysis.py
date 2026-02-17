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
    MAJOR_CTS, 
    DISCOVERY_COHORTS,
    palette_trend,
    surrogate_names,
    CONFIG_FA
)
from hiara import retrieve_net_consensus

from hiara import retrieve_sig_stats, retrieve_stats, mapping_minor_2_major
from hiara.src.feature_association.plots import wrapper_sig_features_counts, plot_heatmap_overal, plot_central_features, \
    plot_interaction_of_features_between_cell_types, plot_case_tf, gsea_analysis, plot_scatter_feature_vs_age, \
        plot_features_vs_datasets, plot_directional_consistency_scatter, \
        plot_young_vs_aging


warnings.filterwarnings("ignore")

# Set matplotlib defaults
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "Arial"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-name', type=str, required=True, 
                       choices=list(CONFIG_FA.keys()),
                       help='Analysis configuration name from CONFIG_FA')
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

    if analysis_name in ['tfa_major_b']:
        plot_heatmap_overal(stats_features, analysis_name=args.analysis_name)
        wrapper_sig_features_counts(args)
        plot_central_features(stats_features_sig, cell_types=['CD4T', 'CD8T', 'NK', 'MONO'])
        plot_interaction_of_features_between_cell_types(args)
        plot_case_tf(args)
        if not skip_pathway:
            gsea_analysis(stats_sig=stats_features_sig)
        # plot_scatter_feature_vs_age(analysis_name, cell_types=['CD8T'], features=None, feature_selection_mode='top_sig', top_genes=5)
        # plot_sig_networks()
    elif analysis_name in ['tfa_sub_b']:
        wrapper_sig_features_counts(args)
        plot_heatmap_overal(stats_features, analysis_name=args.analysis_name)
        # plot_features_vs_datasets(cell_type='Tcm_Naive_CD8', analysis_name=analysis_name)


        stats_features_ref = retrieve_stats(analysis_name='tfa_major_b', cell_type='CD8T')
        stats_features_c = stats_features[stats_features['cell_type']=='Tcm_Naive_CD8'].copy()
        stats_features_c['cell_type'] = stats_features_c['cell_type'].map(mapping_minor_2_major)
        
        plot_directional_consistency_scatter(
                                            stats_features_c, 
                                            stats_features_ref,
                                            x_label = 'Tcm/Naive CD8 \n(significance)',
                                            y_label = 'CD8T \n(significance)',
                                            association_col = '-log10_p_adj',
                                            agreement='same',
                                            label_consistent = 'Consistent',
                                            label_opposing = 'Opposing',
                                            save_suffix = 'sub_vs_major_aging',
                                            pvalue_col='meta_p_adj',
                                            all_sig=True
                                        )
        
    elif analysis_name == 'gene_expression':
        wrapper_sig_features_counts(args)
        plot_heatmap_overal(stats_features, analysis_name=args.analysis_name)
    elif analysis_name == 'sub_tf_markers':
        plot_heatmap_overal(stats_features, analysis_name=args.analysis_name)
        stats_features_ref = retrieve_sig_stats(analysis_name='tfa_major_b', cell_type='CD8T')
        # print(stats_features[stats_features['cell_type']=='Tcm_Naive_CD8'][['gene', 'meta_p_adj', 'slope']])
        stats_features_s = stats_features[stats_features['cell_type']=='Tcm_Naive_CD8']
        stats_features_s['cell_type'] = stats_features_s['cell_type'].map(mapping_minor_2_major)

        plot_directional_consistency_scatter(stats_features_s, 
                                             stats_features_ref,
                                            x_label = 'TCM/Naive CD8 markers \n(significance)',
                                            y_label = 'TF act vs age in CD8T \n(significance)',
                                            association_col = '-log10_p_adj',
                                            agreement='same',
                                            label_consistent = 'Consistent',
                                            label_opposing = 'Opposing',
                                            save_suffix = '',
                                            pvalue_col='meta_p_adj'
                                        )

        
        
    elif analysis_name in ['tfa_traj']:
        plot_young_vs_aging(analysis_name, cell_type='CD8T',  
                        young_age_threshold=30, annotate_top_n=10)
        wrapper_sig_features_counts(args)
        # plot_heatmap_overal(stats_features, analysis_name=args.analysis_name) features=['TCF7', 'LEF1', 'GATA3', 'KLF6'], 
        plot_scatter_feature_vs_age(analysis_name, cell_types=['CD8T'], features=['TCF7', 'LEF1', 'GATA3', 'KLF6'])
        plot_scatter_feature_vs_age(analysis_name, cell_types=['CD8T'], feature_selection_mode='top_central')
        plot_scatter_feature_vs_age(analysis_name, cell_types=['CD8T'], feature_selection_mode='top_sig')
        plot_features_vs_datasets(cell_type='CD8T',  analysis_name=analysis_name, top_features=20)

        # - consistency with tf activity in major CD8T cell types
        stats_features_ref = retrieve_sig_stats(analysis_name='tfa_major_b', cell_type='CD8T')
        
        plot_directional_consistency_scatter(stats_features[stats_features['cell_type']=='CD8T'], 
                                             stats_features_ref,
                                            x_label = 'TF act vs traj. \n(significance)',
                                            y_label = 'TF act. \n(significance)',
                                            association_col = '-log10_p_adj',
                                            agreement='same',
                                            label_consistent = 'Consistent',
                                            label_opposing = 'Opposing',
                                            save_suffix = '',
                                            pvalue_col='meta_p_adj'
                                        )
    elif analysis_name in ['ct_freq']:
        plot_scatter_feature_vs_age(analysis_name, cell_types=['CD8T'], feature_selection_mode='top_sig', top_features=10)
        plot_features_vs_datasets(cell_type='CD8T', analysis_name=args.analysis_name, top_features=20)
    elif analysis_name == 'ct_pol_dist':
        wrapper_sig_features_counts(args)
        plot_heatmap_overal(stats_features, analysis_name=args.analysis_name)
        plot_scatter_feature_vs_age(analysis_name, cell_types=['CD8T', 'CD4T'], feature_selection_mode='top_sig', top_features=10)
        plot_features_vs_datasets(cell_type='CD8T', analysis_name=args.analysis_name, top_features=20)
        plot_features_vs_datasets(cell_type='CD4T', analysis_name=args.analysis_name, top_features=20)
    else:
        
        raise ValueError('Unknown feature type')