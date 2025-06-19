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
from ciim.src.tf_activity.helper import retrieve_sig_stats
from ciim.src.insilico_perturbation.helper import wrapper_run_tf_screening, plot_age_acceleration

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
    perturb_rr = wrapper_run_tf_screening(par_single, n_jobs=n_jobs, cell_types=cell_types, datasets=datasets)
    perturb_rr.to_csv(save_file)

def run_all_aging_tf_perturbation():
    print('Running all aging TF perturbation...')
    save_file = f"{save_dir}/perturbation/all_aging_tf_perturbation.csv"
    par_aging = {
        'simulation_iteration': simulation_iteration,
        'n_donors': n_donors,
        'data_type': data_type,
        'version': version,
        'reg_type': reg_type,
        'feature_type': 'gene_expression',
        'perturbation_mode': 'natural_aging',
        'perturbation_type': 'multi',
        'tfs': 'aging',
    }
    df_all = wrapper_run_tf_screening(par_aging, n_jobs=n_jobs, cell_types=cell_types, datasets=datasets)
    df_all.to_csv(save_file)

# --- SLE TF Perturbation ---
def retrieve_sig_stats_disease(dataset='', feature_type='tf_activity', test_type='unpaired', type='bulk'):
    stats_drug = pd.read_csv(f'{save_dir}/stats/stats_{dataset}_{type}_{feature_type}_{test_type}.csv', index_col=0)
    return stats_drug[(stats_drug['p_value_adj'] < 0.05) & (stats_drug['age_group'] == 'Both age groups')].copy()

def get_overlapping_tf_slopes_in_disease_and_aging(stats_disease_sig):
    stats_aging_sig = retrieve_sig_stats(type='bulk').drop_duplicates(subset=['cell_type', 'tf'])[['cell_type', 'tf']]
    stats_disease_sig = stats_disease_sig[['cell_type', 'tf', 'slope_condition']].drop_duplicates()
    stats_both = stats_aging_sig.merge(stats_disease_sig, on=['cell_type', 'tf'], how='inner')
    return stats_both.set_index('tf').rename(columns={'slope_condition': 'slope'})[['slope', 'cell_type']]
def run_sle_tf_perturbation():
    print('Running SLE TF perturbation...')
    save_file = f"{save_dir}/perturbation/sle_tf_perturbation.csv"
    dataset = 'SLE_European'
    stats_disease_sig = retrieve_sig_stats_disease(dataset=dataset)
    if True: # overlap with aging TFs
        slope_sle = get_overlapping_tf_slopes_in_disease_and_aging(stats_disease_sig)
    else: # SLE TFs
        slope_sle = stats_disease_sig[['tf','cell_type','slope_condition']].set_index('tf').rename(columns={'slope_condition': 'slope'})[['slope', 'cell_type']]
        print(slope_sle.groupby('cell_type').size())
    slope_sle['slope'] = np.sign(slope_sle['slope']) 
    par_sle = {
        'simulation_iteration': simulation_iteration,
        'n_donors': n_donors,
        'data_type': data_type,
        'version': version,
        'reg_type': reg_type,
        'feature_type': 'gene_expression',
        'perturbation_mode': None,
        'perturbation_type': 'multi',
        'slope_df': slope_sle,
    }
    df_sle = wrapper_run_tf_screening(par_sle, n_jobs=1, cell_types=cell_types, datasets=datasets_all)
    df_sle.to_csv(save_file)

# --- Drug Perturbation ---
def retrieve_sig_stats_drug(feature_type='tf_activity', dataset='CXCL9', test_type='paired', type='bulk'):
    stats_drug = pd.read_csv(f'{save_dir}/stats/stats_{dataset}_{type}_{feature_type}_{test_type}.csv', index_col=0)
    return stats_drug[(stats_drug['p_value_adj'] < 0.05)].copy()

def get_overlapping_tf_slopes_in_drug_and_aging(stats_drug_sig):
    stats_aging_sig = retrieve_sig_stats(type='bulk').drop_duplicates(subset=['cell_type', 'tf'])[['cell_type', 'tf']]
    stats_drug_sig = stats_drug_sig[['cell_type', 'tf', 'slope_condition', 'comparision']].drop_duplicates()
    stats_both = stats_aging_sig.merge(stats_drug_sig, on=['cell_type', 'tf'], how='inner')
    return stats_both.set_index('tf').rename(columns={'slope_condition': 'slope'})[['slope', 'cell_type', 'comparision']]

from ciim.src.insilico_perturbation.helper import wrapper_run_tf_screening, plot_age_acceleration
def run_perturbation_tf_perturbation():
    print('Running drug perturbation TF perturbation...')
    stats_drug_sig = retrieve_sig_stats_drug(dataset='CXCL9')
    if True: # overlapping TFs
        df_slope = get_overlapping_tf_slopes_in_drug_and_aging(stats_drug_sig)
    else:
        df_slope = stats_drug_sig.set_index('tf').rename(columns={'slope_condition': 'slope'})[['slope', 'cell_type', 'comparision']]
    print(df_slope.groupby(['cell_type', 'comparision']).size())
    df_slope['slope'] = np.sign(df_slope['slope'])
    df_slope.head()
    df_store = []
    for comparision in tqdm(df_slope['comparision'].unique()):
    # for comparision in ['Ruxolitinib (ctr: LPS)']:
        df_slope_c = df_slope[df_slope['comparision'] == comparision].copy()

        par = {
            'simulation_iteration': simulation_iteration,
            'n_donors': n_donors,
            'data_type': data_type,
            'version': version,
            'reg_type': reg_type,
            'feature_type': 'gene_expression',
            'perturbation_mode': None,
            'perturbation_type': 'multi',
            'slope_df': df_slope_c,
        }
        df_all_c = wrapper_run_tf_screening(par, 
                                    n_jobs=n_jobs,
                                    cell_types=cell_types,
                                    datasets=datasets# aging_clock_train_datasets, #['data12'] # datasets_all
                                    )
        df_all_c['comparision'] = comparision
        df_store.append(df_all_c)
    df_all = pd.concat(df_store)
    df_all.to_csv(f"{save_dir}/perturbation/tf_screen_results_drug_perturbation.csv")



if __name__ == '__main__':
   # --- Parameters ---
    reg_type = 'ridge'
    version = 'v1.0'
    data_type = 'bulk'
    simulation_iteration = 3
    n_donors = 50
    n_jobs = 10
    cell_types = ['CD8T']
    datasets = datasets_all

    run_single_aging_tf_perturbation()
    run_all_aging_tf_perturbation()
    run_sle_tf_perturbation()
    run_perturbation_tf_perturbation()

