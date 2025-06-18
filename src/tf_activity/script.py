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

from ciim.src.common import cell_types, datasets_e, datasets_a, datasets_all, mapping_minor_2_major, save_dir

def run_workflow_tf_activity(par):
    print(par)
    # - step 1: calculate TF activity
    if True: 
        print('Calculating TF activity...')
        wrapper_tf_activity(par)
    # - step 2: calculate TF association with age
    if True:
        stats_features_all = wrapper_association_with_age_condition(par)
        stats_features_all.to_csv(par['stats_features'], index=False)
    if True:
        print('TF discovery/validation...')
        wrapper_meta_analysis(par)

def run_workflow_gene_expression(par):
    print(par)
    if True: 
        print('Calculating gene expression...')
        wrapper_gene_expression(par)
    # - step 2: calculate association with age
    if True:
        stats_features = wrapper_association_with_age_condition(par)
        stats_features.rename(columns={'tf': 'gene'}, inplace=True)
        stats_features.to_csv(par['stats_features'], index=False)
    if True:
        print('Discovery/validation...')
        wrapper_meta_analysis(par)

    # if True:
    #     print('Target association with age...')
    #     wrapper_target_association_age(par)
    
    

if __name__ == '__main__':
    os.makedirs(f'{save_dir}/tmp/', exist_ok=True)
    os.makedirs(f'{save_dir}/tf_activity/', exist_ok=True)
    os.makedirs(f'{save_dir}/tf_activity/tf_acts', exist_ok=True)
    os.makedirs(f'{save_dir}/gene_expression/', exist_ok=True)
    os.makedirs(f'{save_dir}/gene_expression/gene_expression', exist_ok=True)
    os.makedirs(f'{save_dir}/gene_expression/gene_expression', exist_ok=True)

    run_flag = True
    run_flag_gender = False
    
    if run_flag:
        for data_type in ['bulk']:
            for feature_type in ['tf_activity']:
                par = {
                    'type': data_type,
                    'feature_type': feature_type,
                    'datasets': datasets_all,
                    'association_type': 'spearman',
                    'cell_type_resolution': 'Major_CT' if data_type == 'bulk' else 'Sub_CT',
                    'stats_features': f'{save_dir}/{feature_type}/stats_features_{data_type}.csv',
                    'stats_all': f'{save_dir}/{feature_type}/stats_all_{data_type}.csv', 
                    'temp_dir': f'{save_dir}/tmp/',
                }
                if feature_type == 'tf_activity':
                    run_workflow_tf_activity(par)
                elif feature_type == 'gene_expression':
                    run_workflow_gene_expression(par)


    if run_flag_gender:
        for data_type in ['bulk', 'bulk_minor']:
            for feature_type in ['tf_activity', 'gene_expression']:
                for gender in ['M', 'F']:
                    par = {
                        'type': f'{data_type}_{gender}',
                        'cell_type_resolution': 'Major_CT' if data_type == 'bulk' else 'Sub_CT',
                        'feature_type': feature_type,
                        'association_type': 'spearman',
                        'datasets': datasets_all,
                        'stats_features': f'{save_dir}/{feature_type}/stats_features_{data_type}_{gender}.csv',
                        'stats_all': f'{save_dir}/{feature_type}/stats_all_{data_type}_{gender}.csv', 
                        'temp_dir': f'{save_dir}/tmp/',
                    }
                    if feature_type == 'tf_activity':
                        run_workflow_tf_activity(par)
                    elif feature_type == 'gene_expression':
                        run_workflow_gene_expression(par)
