

import warnings
import sys 
import os 
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import argparse
from pybedtools import BedTool

warnings.filterwarnings("ignore")


argparser = argparse.ArgumentParser()
argparser.add_argument('--task_grn_inference_dir', type=str, default='../', help='Path to task_grn_inference directory')
argparser.add_argument('--output_dir', type=str, default='output/motif_analysis', help='Output directory')

args = argparser.parse_args()
task_grn_inference_dir = args.task_grn_inference_dir
output_dir = args.output_dir



par = {
    'motif_scores_db': f"{task_grn_inference_dir}/task_grn_inference/resources/supp_data/db.regions_vs_motifs.scores.feather",
    'motifs_info': f'{task_grn_inference_dir}/task_grn_inference/resources/supp_data/motifs-v10-nr.hgnc-m0.00001-o0.0.tbl',
    'annotation_file': f'{task_grn_inference_dir}/task_grn_inference/resources/supp_data/gencode.v45.annotation.gtf.gz',
    'atac': f'{task_grn_inference_dir}/task_grn_inference/resources/grn_benchmark/inference_data/op_atac.h5ad', # to get consensus peaks
    'output_dir': f'{output_dir}',
    'consensus_peak': f'{output_dir}/consensus_peak.csv', # where consensus peaks are stored,
    'enhancer_distance': 150_000,  #TODO: this should be distance free
    'promoter_df': f'{output_dir}/promoters.csv',
    'regions2promoters': f'{output_dir}/regions2promoters.csv',
    'tf2gene': f'{output_dir}/tf2gene.csv',
    'n_jobs': 10
}



os.makedirs(par['output_dir'], exist_ok=True)

from helper import run_consensus_peak, run_promotor_analysis, run_region2promotors, run_tf2gene


if __name__ == '__main__':
    # - run consensus peak
    # run_consensus_peak(par) #TODO: plot the interaction between cell types
    
    # # - run promotor analysis
    # run_promotor_analysis(par)

    # # - run region2promotors
    # run_region2promotors(par)

    # - run tf2gene
    run_tf2gene(par)

