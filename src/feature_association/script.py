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
from ciim.src.feature_association.helper import  wrapper_association_with_age_condition, wrapper_meta_analysis, \
        wrapper_tf_activity, wrapper_gene_expression, wrapper_meta_analysis, wrapper_gene_score, wrapper_aging_hallmarks

from ciim.src.common import cell_types, datasets_e, datasets_a, datasets_all, mapping_minor_2_major, SAVE_DIR

def run_workflow_tf_activity(par):
    print(par)
    if True: 
        print('Calculating TF activity...')
        wrapper_tf_activity(par)
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
        stats_features.to_csv(par['stats_features'], index=False)
    if True:
        print('Meta analysis...')
        wrapper_meta_analysis(par)


def run_workflow_gene_score(par):
    print(par)
    if True: 
        print('Calculating gene scores...')
        wrapper_gene_score(par)
    # - step 2: calculate association with age
    if True:
        stats_features = wrapper_association_with_age_condition(par)
        stats_features.to_csv(par['stats_features'], index=False)
    if True:
        print('Meta analysis...')
        wrapper_meta_analysis(par)


def run_workflow_aging_hallmarks(par):
    print(par)
    if True: 
        print('Calculating aging hallmarks...')
        wrapper_aging_hallmarks(par)
    # - step 2: calculate association with age
    if True:
        stats_features = wrapper_association_with_age_condition(par)
        stats_features.to_csv(par['stats_features'], index=False)
    if True:
        print('Meta analysis...')
        wrapper_meta_analysis(par)


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run feature association analysis')
    parser.add_argument('--promotor-only', action='store_true', 
                        help='Use promotor-based GRN only (default: use full GRN)')
    parser.add_argument('--data-type', required=False, default='bulk',
                        help='Data type to process')
    parser.add_argument('--feature-type', required=False, default='tf_activity',
                        help='Feature type to process')
                                      
    args = parser.parse_args()
    
    os.makedirs(f'{SAVE_DIR}/tmp/', exist_ok=True)
    os.makedirs(f'{SAVE_DIR}/tf_activity/', exist_ok=True)
    os.makedirs(f'{SAVE_DIR}/tf_activity/tf_acts', exist_ok=True)
    os.makedirs(f'{SAVE_DIR}/gene_expression/', exist_ok=True)
    os.makedirs(f'{SAVE_DIR}/gene_expression/gene_expression', exist_ok=True)
    os.makedirs(f'{SAVE_DIR}/gene_score/', exist_ok=True)
    os.makedirs(f'{SAVE_DIR}/gene_score/gene_score', exist_ok=True)
    os.makedirs(f'{SAVE_DIR}/aging_hallmarks/', exist_ok=True)
    os.makedirs(f'{SAVE_DIR}/aging_hallmarks/aging_hallmarks', exist_ok=True)

    only_promotor_based = args.promotor_only  # From command-line flag
    data_type = args.data_type
    feature_type = args.feature_type
    suffix = '_promotor' if only_promotor_based else ''
    

    par = {
        'type': data_type,
        'feature_type': feature_type,
        'cell_types': cell_types,
        'datasets': datasets_all,
        'association_type': 'spearman',
        'cell_type_resolution': 'Major_CT' if data_type == 'bulk' else 'Sub_CT',
        'stats_features': f'{SAVE_DIR}/{feature_type}/stats_features_{data_type}{suffix}.csv',
        'stats_all': f'{SAVE_DIR}/{feature_type}/stats_all_{data_type}{suffix}.csv', 
        'temp_dir': f'{SAVE_DIR}/tmp/',
        'only_promotor_based': only_promotor_based,
        # 'pathway': 'canonical' if feature_type == 'gene_score' else None, #'canonical' #opengenes
        'gene_coverage': 'target_genes' if feature_type == 'gene_score' else None, # 'target_genes' all_genes
    }
    if feature_type == 'tf_activity':
        run_workflow_tf_activity(par)
    elif feature_type == 'gene_expression':
        run_workflow_gene_expression(par)
    elif feature_type == 'gene_score':
        run_workflow_gene_score(par)
    elif feature_type == 'aging_hallmarks':
        run_workflow_aging_hallmarks(par)


    # if run_flag_gender:
    #     for data_type in ['bulk', 'bulk_minor']:
    #         for feature_type in ['tf_activity', 'gene_expression']:
    #             for gender in ['M', 'F']:
    #                 par = {
    #                     'type': f'{data_type}_{gender}',
    #                     'cell_type_resolution': 'Major_CT' if data_type == 'bulk' else 'Sub_CT',
    #                     'feature_type': feature_type,
    #                     'cell_types': cell_types,
    #                     'association_type': 'spearman',
    #                     'datasets': datasets_all,
    #                     'stats_features': f'{SAVE_DIR}/{feature_type}/stats_features_{data_type}_{gender}.csv',
    #                     'stats_all': f'{SAVE_DIR}/{feature_type}/stats_all_{data_type}_{gender}.csv', 
    #                     'temp_dir': f'{SAVE_DIR}/tmp/',
    #                 }
    #                 if feature_type == 'tf_activity':
    #                     run_workflow_tf_activity(par)
    #                 elif feature_type == 'gene_expression':
    #                     run_workflow_gene_expression(par)
