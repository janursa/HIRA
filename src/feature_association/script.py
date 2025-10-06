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
        wrapper_tf_activity, wrapper_gene_expression, wrapper_meta_analysis, wrapper_gene_score

from ciim.src.common import cell_types, datasets_e, datasets_a, datasets_all, mapping_minor_2_major, save_dir

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
    if False: 
        print('Calculating gene expression...')
        wrapper_gene_expression(par)
    # - step 2: calculate association with age
    if True:
        stats_features = wrapper_association_with_age_condition(par)
        stats_features.rename(columns={'tf': 'gene'}, inplace=True)
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
        stats_features.rename(columns={'tf': 'pathway'}, inplace=True)
        stats_features.to_csv(par['stats_features'], index=False)
    if True:
        print('Meta analysis...')
        wrapper_meta_analysis(par)


if __name__ == '__main__':
    os.makedirs(f'{save_dir}/tmp/', exist_ok=True)
    os.makedirs(f'{save_dir}/tf_activity/', exist_ok=True)
    os.makedirs(f'{save_dir}/tf_activity/tf_acts', exist_ok=True)
    os.makedirs(f'{save_dir}/gene_expression/', exist_ok=True)
    os.makedirs(f'{save_dir}/gene_expression/gene_expression', exist_ok=True)
    os.makedirs(f'{save_dir}/gene_score/', exist_ok=True)
    os.makedirs(f'{save_dir}/gene_score/gene_score', exist_ok=True)

    run_flag = True
    run_flag_gender = False
    
    if run_flag:
        for data_type in ['bulk']: # 'bulk_minor', 'bulk
            for feature_type in ['gene_expression']: # 'gene_expression', 'tf_activity', 'gene_score'
                par = {
                    'type': data_type,
                    'feature_type': feature_type,
                    'cell_types': cell_types,
                    'datasets': datasets_all,
                    'association_type': 'spearman',
                    'cell_type_resolution': 'Major_CT' if data_type == 'bulk' else 'Sub_CT',
                    'stats_features': f'{save_dir}/{feature_type}/stats_features_{data_type}.csv',
                    'stats_all': f'{save_dir}/{feature_type}/stats_all_{data_type}.csv', 
                    'temp_dir': f'{save_dir}/tmp/',
                    # 'pathway': 'canonical' if feature_type == 'gene_score' else None, #'canonical' #opengenes
                    'gene_coverage': 'target_genes' if feature_type == 'gene_score' else None, # 'target_genes' all_genes
                }
                if feature_type == 'tf_activity':
                    run_workflow_tf_activity(par)
                elif feature_type == 'gene_expression':
                    run_workflow_gene_expression(par)
                elif feature_type == 'gene_score':
                    run_workflow_gene_score(par)


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
    #                     'stats_features': f'{save_dir}/{feature_type}/stats_features_{data_type}_{gender}.csv',
    #                     'stats_all': f'{save_dir}/{feature_type}/stats_all_{data_type}_{gender}.csv', 
    #                     'temp_dir': f'{save_dir}/tmp/',
    #                 }
    #                 if feature_type == 'tf_activity':
    #                     run_workflow_tf_activity(par)
    #                 elif feature_type == 'gene_expression':
    #                     run_workflow_gene_expression(par)
