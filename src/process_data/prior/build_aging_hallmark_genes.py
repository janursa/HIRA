"""
Combine per-hallmark gene-list CSVs into a single aging_hallmark_genes.csv.

Expects the individual CSVs (one per aging hallmark, columns Symbol/Gene_Set) to
already exist under PRIOR_DIR/aging_hallmark_genes/ -- this script only merges them,
it does not source them.

From notebooks/prepare_resources.ipynb ("# Aging hallmark genes").

Usage:
    python src/process_data/prior/build_aging_hallmark_genes.py
"""
import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, 'src')
from config import PRIOR_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--folder', default=f'{PRIOR_DIR}/aging_hallmark_genes/')
parser.add_argument('--out', default=f'{PRIOR_DIR}/aging_hallmark_genes.csv')
args = parser.parse_args()

all_files = glob.glob(os.path.join(args.folder, '*.csv'))
if not all_files:
    raise FileNotFoundError(
        f'No per-hallmark CSVs found in {args.folder}. Populate it manually first '
        '(one CSV per aging hallmark with Symbol/Gene_Set columns).'
    )

dfs = [pd.read_csv(f) for f in all_files]
df = pd.concat(dfs, ignore_index=True)
print(f'Combined dataframe shape: {df.shape}')
print(f'Number of files read: {len(all_files)}')
df.rename({'Symbol': 'gene', 'Gene_Set': 'gene_set'}, axis=1, inplace=True)
print(df.groupby('gene_set').size().sort_values())

df.to_csv(args.out)
print(f'Saved: {args.out}')
