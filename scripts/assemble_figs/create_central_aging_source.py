#!/usr/bin/env python
"""
Top central age-associated TFs (out-degree centrality), one panel per cell type.

Output: PLOTS_DIR/assembled/central_aging_source_tfa_major_b.png
"""
import os
import sys
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from hira import retrieve_sig_stats
from hira.src.config import PLOTS_DIR
from hira.src.feature_association.plots import plot_central_features

plt.rcParams.update({"figure.dpi": 150, "font.family": "Arial", "font.size": 10})

ANALYSIS_NAME = 'tfa_major_b'
CELL_TYPES = ['CD4T', 'CD8T', 'NK', 'MONO']
TOP_N = 15


def main():
    stats_sig = retrieve_sig_stats(analysis_name=ANALYSIS_NAME)
    out_dir = os.path.join(PLOTS_DIR, 'assembled')
    os.makedirs(out_dir, exist_ok=True)
    plot_central_features(stats_sig, cell_types=CELL_TYPES, analysis_name=ANALYSIS_NAME,
                          n_top=TOP_N, plots_dir=out_dir)
    plt.close('all')


if __name__ == '__main__':
    main()
