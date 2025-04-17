import anndata as ad
import pandas as pd
import sys
import numpy as np
import scanpy as sc 
from scipy import stats
from scipy.stats import pearsonr
import os
import argparse
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from pandas.api.types import CategoricalDtype
from ciim.src.tf_activity.helper import  wrapper_association_with_age, wrapper_run_meta_analysis, \
        wrapper_tf_activity, wrapper_run_meta_analysis

from ciim.src.common import cell_types, datasets_e, datasets_a, datasets_all, mapping_minor_2_major



def run_workflow_tf_activity(par):
    print(par)
    # - step 1: calculate TF activity
    if False: 
        print('Calculating TF activity...')
        wrapper_tf_activity(cell_types, datasets_all, type=par['type'])
    # - step 2: calculate TF association with age
    if False:
        stats_tfs_all = wrapper_association_with_age(par, cell_types, datasets=datasets_all)
        stats_tfs_all.to_csv(par['stats_tfs'], index=False)
    if True:
        print('TF discovery/validation...')
        wrapper_run_meta_analysis(par)

    # if True:
    #     print('Target association with age...')
    #     wrapper_target_association_age(par)
    
    

if __name__ == '__main__':
    if True:
        # ----- bulk TF activity: non linear association
        par = {
            'type': 'bulk',
            'association_type': 'spearman',
            'cell_type_resolution': 'Major_CT',
            'predictor_tfs_df': 'output/tf_activation/predictor_tfs_bulk.csv',
            'stats_tfs': 'output/tf_activation/stats_tfs_bulk.csv',
            'stats_targets': 'output/tf_activation/stats_targets_bulk.csv',
            'stats_all': 'output/tf_activation/stats_all_bulk.csv', 
            'temp_dir': 'output/tmp/',
        }
        run_workflow_tf_activity(par)
    if False:
        # ----- bulk TF activity: minor
        par = {
            'type': 'bulk_minor',
            'association_type': 'spearman',
            'cell_type_resolution': 'Sub_CT',
            'predictor_tfs_df': 'output/tf_activation/predictor_tfs_bulk_minor.csv',
            'stats_tfs': 'output/tf_activation/stats_tfs_bulk_minor.csv',
            'stats_targets': 'output/tf_activation/stats_targets_bulk_minor.csv',
            'stats_all': 'output/tf_activation/stats_all_bulk_minor.csv', 
            'temp_dir': 'output/tmp/',
        }
        run_workflow(par)
    if False:
        # ----- bulk TF activity: gender
        for gender in ['F', 'M']:
            par = {
                'type': f'bulk_{gender}',
                'cell_type_resolution': 'Major_CT',
                'association_type': 'spearman',
                'predictor_tfs_df': f'output/tf_activation/predictor_tfs_bulk_{gender}.csv',
                'stats_tfs': f'output/tf_activation/stats_tfs_bulk_{gender}.csv',
                'stats_targets': f'output/tf_activation/stats_targets_bulk_{gender}.csv',
                'stats_all': f'output/tf_activation/stats_all_bulk_{gender}.csv', 
                'temp_dir': 'output/tmp/',
            }
            run_workflow(par)
    if False:
        # ----- bulk TF activity: gender and minor
        for gender in ['M', 'F']:
            
            par = {
                'type': 'bulk_minor',
                'cell_type_resolution': 'Sub_CT',
                'association_type': 'spearman',
                'tf_acts_dir': 'output/tf_activation/tf_acts/', 
                'predictor_tfs_df': f'output/tf_activation/predictor_tfs_bulk_minor_{gender}.csv',
                'stats_tfs': f'output/tf_activation/stats_tfs_bulk_minor_{gender}.csv',
                'stats_targets': f'output/tf_activation/stats_targets_bulk_minor_{gender}.csv',
                'stats_all': f'output/tf_activation/stats_all_bulk_minor_{gender}.csv', 
                'temp_dir': 'output/tmp/',
            }
            os.makedirs(par['tf_acts_dir'], exist_ok=True)
            run_workflow(par)
    if False:
        # ----- sc TF activity
        par = {
            'type': 'sc_std',
            'cell_type_resolution': 'Major_CT',
            'association_type': 'spearman',
            'tf_acts_dir': 'output/tf_activation/tf_acts/', 
            'stats_tfs': 'output/tf_activation/stats_tfs_sc.csv',
            'stats_targets': 'output/tf_activation/stats_targets_sc.csv',
            'stats_all': 'output/tf_activation/stats_all_sc.csv', 
            'temp_dir': 'output/tmp/',
            'std_dir': 'output/std/',
        }
        os.makedirs(par['tf_acts_dir'], exist_ok=True)
        os.makedirs(par['std_dir'], exist_ok=True)
        run_workflow(par)
    
        
    