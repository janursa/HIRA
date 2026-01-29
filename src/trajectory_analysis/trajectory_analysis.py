#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trajectory Analysis Script
Identifies TFs whose activity within the differentiation trajectory changes with aging.

Usage:
    python trajectory_analysis.py association --dataset abf300 --cell_type CD8T
    python trajectory_analysis.py verify-dpt --dataset abf300 --cell_type CD8T
    python trajectory_analysis.py plot-association --dataset abf300 --cell_type CD8T --results_file path/to/results.csv
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import scanpy as sc
import warnings
import anndata as ad
import os
from scipy.sparse import issparse
from scipy.stats import spearmanr, pearsonr
import decoupler as dc
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import tqdm
import logging
import argparse

warnings.filterwarnings('ignore')
logging.getLogger('anndata').setLevel(logging.ERROR)
pd.set_option('display.max_columns', 100)
plt.rcParams["font.family"] = "Arial"

# Local imports
from hiara import retrieve_feature_data, retrieve_sig_stats
from hiara import OUTPUT_DIR, PRIOR_DIR, PLOTS_DIR, CLOCKS_DIR, CELL_TYPES, surrogate_names, colors_blind, palette_cell_types, palette_datasets, palette_datasets_pretty, palette_genders, AGING_COHORTS
from hiara import retrieve_adata, retrieve_net, retrieve_net_consensus
from hiara import get_config
from grn_benchmark.src.helper import load_env
from task_grn_inference import normalize_func, bulkify_func
from hiara.src.trajectory_analysis.helper import (
    process_donors, cluster, annotate, 
    association_analysis_dpt, association_analysis_marker,
    plot_association_results,
    write_traj_stats,
    retrieve_traj_stats,
)

def plot_association(args):
    print(f"\n{'='*60}")
    print(f"MODE: Plot Association Results")
    print(f"{'='*60}\n")
    
    adata_combined = process_donors(args.dataset, args.cell_type, test_mode=args.test)
    traj_stats = retrieve_traj_stats(args.dataset, args.cell_type, mode='dpt')
    top_tfs = plot_association_results(adata_combined, traj_stats, 
                                        args.dataset, args.cell_type,
                                        top_n=args.top_n, age_cutoff=args.age_cutoff)

def run_association_analysis(args):
    if args.mode == 'dpt':
        print(f"\n{'='*60}")
        print(f"MODE: DPT-based Trajectory Analysis")
        print(f"{'='*60}\n")
        
        # Process donors with single-cell DPT analysis
        adata_combined = process_donors(args.dataset, args.cell_type, test_mode=args.test)
        
        # Run association: TF_activity ~ dpt_pseudotime * age
        
        results = association_analysis_dpt(adata_combined)

        write_traj_stats(results, args.dataset, args.cell_type, mode='dpt')
        
    elif args.mode == 'marker':
        print(f"\n{'='*60}")
        print(f"MODE: Marker-based Analysis")
        print(f"{'='*60}\n")
        
        # Load data
        print(f"Loading data for dataset={args.dataset}, cell_type={args.cell_type}")
        net = retrieve_net_consensus(cell_type=args.cell_type)
        adata_o = retrieve_adata(dataset=args.dataset, data_type='sc', cell_type=args.cell_type)
        
        print(f"Total cells: {adata_o.n_obs}")
        print(f"Unique donor_age samples: {adata_o.obs['donor_age'].nunique()}")
        
        # Filter donors if in test mode
        if args.test:
            donor_age_counts = adata_o.obs['donor_age'].value_counts()
            test_donors = donor_age_counts.head(10).index
            adata_o = adata_o[adata_o.obs['donor_age'].isin(test_donors)].copy()
            print(f"TEST MODE: Using {len(test_donors)} donors")
        
        # Cluster and bulkify
        print("\n=== Clustering and creating pseudobulk ===")
        adata_bulk = cluster(adata_o, dataset=args.dataset)
        
        # Annotate with marker scores
        print("\n=== Annotating with marker scores ===")
        adata_bulk = annotate(adata_bulk)
        
        # Compute TF activities
        print("\n=== Computing TF activities ===")
        dc.mt.ulm(adata_bulk, net=net, tmin=5)
        
        # Run association: TF_activity ~ naive_score * age and TF_activity ~ effector_score * age
        
        association_analysis_marker(adata_bulk, output_path)
        
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
        

if __name__ == '__main__':
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str,
                        help="Mode of operation: 'dpt' for DPT association analysis, 'marker' for naive/effector marker analysis, 'post_association' for plotting association results.")
    parser.add_argument('--dataset', type=str, required=True,
                        help="Dataset name (e.g., 'abf300').")
    parser.add_argument('--cell_type', type=str, required=True,
                        help="Cell type (e.g., 'CD8T').")
    parser.add_argument('--test', action='store_true',
                        help="Run in test mode with a subset of data.")
    args = parser.parse_args()
    
    if args.mode in ['dpt', 'marker']:
        run_association_analysis(args)
    elif args.mode == 'post_association':
        plot_association(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    