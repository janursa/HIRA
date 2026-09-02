"""Compare each TF's age trend in activity against its own age trend in expression.

Needs two feature analyses to have been run: a TF-activity one and a gene-expression one
(`bash scripts/feature_analysis.sh tfa_major_b` and `... ge_major_b`).

Usage: python src/feature_association/activation_vs_expression.py \
           [--tfa-analysis tfa_major_b] [--ge-analysis ge_major_b] [--dataset aida]
Writes: PLOTS_DIR/activation_vs_expression.png
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from hira.src.config import PLOTS_DIR
from hira.src.feature_association.helper import retrieve_sig_stats
from hira.src.feature_association.plots import plot_activation_vs_expression

COLS = ['cell_type', 'gene', 'slope', 'p_value_adj', 'dataset']


def signed_significance(stats):
    """Signed -log10 adjusted p-value; clipped so p=0 doesn't blow up to inf."""
    return np.sign(stats['slope']) * -np.log10(stats['p_value_adj'].clip(lower=1e-300))


def build_frame(tfa_analysis, ge_analysis, dataset):
    tfs = retrieve_sig_stats(analysis_name=tfa_analysis)[COLS]
    genes = retrieve_sig_stats(analysis_name=ge_analysis)[COLS]
    df = tfs.merge(genes, on=['cell_type', 'gene', 'dataset'], suffixes=('_source', '_target'))
    df = df[df['dataset'] == dataset].copy()
    assert len(df), f'No TFs shared between {tfa_analysis} and {ge_analysis} for {dataset}'
    df['Activation significance'] = signed_significance(
        df.rename(columns={'slope_source': 'slope', 'p_value_adj_source': 'p_value_adj'}))
    df['Expression significance'] = signed_significance(
        df.rename(columns={'slope_target': 'slope', 'p_value_adj_target': 'p_value_adj'}))
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfa-analysis', default='tfa_major_b')
    parser.add_argument('--ge-analysis', default='ge_major_b')
    parser.add_argument('--dataset', default='aida')
    parser.add_argument('--cell-type', default='CD8T')
    args = parser.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    df = build_frame(args.tfa_analysis, args.ge_analysis, args.dataset)
    df = df[df['cell_type'] == args.cell_type]

    plot_activation_vs_expression(df,
                                  col_x='Expression significance',
                                  col_y='Activation significance',
                                  y_label='Activation significance \n (-log10padj)',
                                  x_label='Expression significance \n (-log10padj)',
                                  figsize=(3, 2.7))
    plt.legend([], [], frameon=False)
    plt.suptitle(f'TF activation vs expression ({args.cell_type})', y=.95, fontsize=12, weight='bold')
    plt.tight_layout()
    file_name = os.path.join(PLOTS_DIR, 'activation_vs_expression.png')
    print(f'Saving to {file_name}')
    plt.savefig(file_name, dpi=300)
