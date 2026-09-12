"""
Build the protein-coding gene name whitelist (gene_names.txt) from a GENCODE GTF.

From notebooks/prepare_resources.ipynb ("# Valid gene names").

Usage:
    python src/process_data/prior/build_gene_names.py
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, 'src')
from config import PRIOR_DIR

TASK_GRN_REPO = os.environ['TASK_GRN_BENCHMARK_DIR']
GTF_PATH = f'{TASK_GRN_REPO}/resources/supp_data/gencode.v47.annotation.gtf.gz'

parser = argparse.ArgumentParser()
parser.add_argument('--gtf', default=GTF_PATH)
parser.add_argument('--out', default=f'{PRIOR_DIR}/gene_names.txt')
args = parser.parse_args()

GTF_COLS = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']

gtf_df = pd.read_csv(args.gtf, sep='\t', compression='gzip', comment='#', header=None,
                      names=GTF_COLS, engine='python')

known_chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY', 'chrM']
genes_df = gtf_df[(gtf_df['feature'] == 'gene') & (gtf_df['seqname'].isin(known_chromosomes))].copy()


def extract_attr(attr_str, key):
    match = re.search(fr'{key} "([^"]+)"', attr_str)
    return match.group(1) if match else None


genes_df['gene_name'] = genes_df['attribute'].apply(lambda x: extract_attr(x, 'gene_name'))
genes_df['gene_type'] = genes_df['attribute'].apply(lambda x: extract_attr(x, 'gene_type'))

genes_df = genes_df[genes_df['gene_type'] == 'protein_coding']
genes_df = genes_df.dropna(subset=['gene_name'])
genes_df = genes_df[~genes_df['gene_name'].str.match(r'^ENSG\d+')]

gene_names = genes_df['gene_name'].unique().tolist()
np.savetxt(args.out, gene_names, fmt='%s')
print(f'Final gene count: {len(gene_names)}')
print(f'Saved: {args.out}')
