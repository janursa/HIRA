#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --- Settings ---
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import anndata as ad
import scanpy as sc
from tqdm import tqdm
from sklearn.metrics import r2_score
from scipy.stats import linregress, pearsonr, spearmanr
from pandas.api.types import CategoricalDtype
from scipy.cluster.hierarchy import linkage
from matplotlib.patches import Patch
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = 4, 4

# --- Directories and Imports ---
from ciim.src.common import (
    aging_clock_train_datasets, palette_genders, palette_treatment,
    mapping_major_2_minor, mapping_minor_2_major, cell_types,
    datasets_e, datasets_all, datasets_a, palette_datasets,
    datasets_disease, datasets_drug_perturbation, save_dir,
    palette_datasets_pretty, surrogate_names, colors_blind,
    palette_trend, palette_trend_2, palette_regulation, palette_cell_types
)
from ciim.src.feature_association.helper import retrieve_sig_stats
from ciim.src.insilico_perturbation.helper import wrapper_in_silico_perturbation

def run_single_aging_tf_perturbation():
    print('Running single aging TF perturbation...')
    # --- Single TF Perturbation ---
    perturbation_mode = 'overexpression'
    save_file = f"{save_dir}/perturbation/single_aging_tf_perturbation.csv"

    par_single = {
        'simulation_iteration': simulation_iteration,
        'n_donors': n_donors,
        'data_type': data_type,
        'version': version,
        'reg_type': reg_type,
        'feature_type': 'gene_expression',
        'perturbation_mode': perturbation_mode,
        'perturbation_type': 'single',
        'tfs': None,
    }
    perturb_rr = wrapper_in_silico_perturbation(par_single, n_jobs=n_jobs, cell_types=cell_types, datasets=datasets)
    perturb_rr.to_csv(save_file)



if __name__ == '__main__':
   # --- Parameters ---
    reg_type = 'ridge'
    version = 'v1.0'
    data_type = 'bulk'
    simulation_iteration = 3
    n_donors = 20
    n_jobs = 1
    cell_types = ['CD8T']
    datasets = ['data12']

    run_single_aging_tf_perturbation()
    # run_all_aging_tf_perturbation()
    # run_sle_tf_perturbation()
    # run_perturbation_tf_perturbation()

