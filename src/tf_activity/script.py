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
from ciim.src.tf_activity.helper import  wrapper_association_with_age_condition, wrapper_meta_analysis, \
        wrapper_tf_activity, wrapper_gene_expression, wrapper_meta_analysis

from ciim.src.common import cell_types, datasets_e, datasets_a, datasets_all, mapping_minor_2_major

def run_workflow_tf_activity(par):
    print(par)
    # - step 1: calculate TF activity
    if False: 
        print('Calculating TF activity...')
        wrapper_tf_activity(cell_types, datasets_all, type=par['type'])
    # - step 2: calculate TF association with age
    if False:
        stats_features_all = wrapper_association_with_age_condition(par, cell_types, datasets=datasets_all)
        stats_features_all.to_csv(par['stats_features'], index=False)
    if True:
        print('TF discovery/validation...')
        wrapper_meta_analysis(par)

def run_workflow_gene_expression(par):
    print(par)
    if True: 
        print('Calculating gene expression...')
        wrapper_gene_expression(cell_types, datasets_all, type=par['type'])
    # - step 2: calculate association with age
    if True:
        stats_features = wrapper_association_with_age_condition(par, cell_types, datasets=datasets_all, feature_type='gene_expression')
        stats_features.rename(columns={'tf': 'gene'}, inplace=True)
        stats_features.to_csv(par['stats_features'], index=False)
    if True:
        print('Discovery/validation...')
        wrapper_meta_analysis(par)

    # if True:
    #     print('Target association with age...')
    #     wrapper_target_association_age(par)
    
    

if __name__ == '__main__':
    os.makedirs('output/tmp/', exist_ok=True)
    os.makedirs('output/tf_activity/', exist_ok=True)
    os.makedirs('output/tf_activity/tf_acts', exist_ok=True)
    os.makedirs('output/gene_expression/', exist_ok=True)
    os.makedirs('output/gene_expression/gene_expression', exist_ok=True)

    run_bulk = True
    run_bulk_targets = True


    run_bulk_minor = True
    run_bulk_gender = True
    
    
    if run_bulk:
        # ----- bulk TF activity: non linear association
        par = {
            'type': 'bulk',
            'association_type': 'spearman',
            'cell_type_resolution': 'Major_CT',
            'stats_features': 'output/tf_activity/stats_features_bulk.csv',
            'stats_all': 'output/tf_activity/stats_all_bulk.csv', 
            'temp_dir': 'output/tmp/',
        }
        
        run_workflow_tf_activity(par)
        
    if run_bulk_targets:
        par = {
            'type': 'bulk',
            'association_type': 'spearman',
            'cell_type_resolution': 'Major_CT',
            'stats_features': 'output/gene_expression/stats_features_bulk.csv',
            'stats_all': 'output/gene_expression/stats_all_bulk.csv', #TODO: this for now only includes genes that are in the net. 
            # 'stats_targets': 'output/tf_activity/stats_targets_bulk.csv',
            'temp_dir': 'output/tmp/',
        }
        run_workflow_gene_expression(par)
    
    if run_bulk_minor:
        # ----- bulk TF activity: minor
        par = {
            'type': 'bulk_minor',
            'association_type': 'spearman',
            'cell_type_resolution': 'Sub_CT',
            'stats_features': 'output/tf_activity/stats_features_bulk_minor.csv',
            'stats_all': 'output/tf_activity/stats_all_bulk_minor.csv', 
            'temp_dir': 'output/tmp/',
        }
        run_workflow_tf_activity(par)

    if run_bulk_gender:
        # ----- bulk TF activity: gender
        for gender in ['M','F']:
            par = {
                'type': f'bulk_{gender}',
                'cell_type_resolution': 'Major_CT',
                'association_type': 'spearman',
                'stats_features': f'output/tf_activity/stats_features_bulk_{gender}.csv',
                'stats_all': f'output/tf_activity/stats_all_bulk_{gender}.csv', 
                'temp_dir': 'output/tmp/',
            }
            run_workflow_tf_activity(par)
    if False:
        # ----- sc TF activity
        par = {
            'type': 'sc_std',
            'cell_type_resolution': 'Major_CT',
            'association_type': 'spearman',
            'stats_features': 'output/tf_activity/stats_features_std.csv',
            'stats_targets': 'output/tf_activity/stats_targets_std.csv',
            'stats_all': 'output/tf_activity/stats_all_std.csv', 
            'temp_dir': 'output/tmp/',
            'std_dir': 'output/std/',
        }
        os.makedirs(par['tf_acts_dir'], exist_ok=True)
        os.makedirs(par['std_dir'], exist_ok=True)
        run_workflow_tf_activity(par)
    
        
    