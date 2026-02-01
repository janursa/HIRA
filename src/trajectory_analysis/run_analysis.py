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
    compute_tf_activity_per_donor, cluster, annotate, 
    association_analysis_dpt, association_analysis_marker,
    write_traj_stats,
    retrieve_traj_stats,
)

 

if __name__ == '__main__':
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,                         
                        help="Analysis mode: 'dpt-with-marker' or 'tfact-with-dpt'")
    parser.add_argument('--dataset', type=str, required=True,
                        help="Dataset name (e.g., 'abf300').")
    parser.add_argument('--cell_type', type=str, required=True,
                        help="Cell type (e.g., 'CD8T').")
    parser.add_argument('--test', action='store_true',
                        help="Run in test mode with a subset of data.")
    args = parser.parse_args()
    
    run_association_analysis(args)
    
    