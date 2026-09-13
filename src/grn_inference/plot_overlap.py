"""Upset plots of GRN overlap: per-cohort networks within a cell type, and consensus
networks across cell types.

Usage: python src/grn_inference/plot_overlap.py [--level edge|source|target]
Writes into PLOTS_DIR/exp_analysis/grn_overlap/: grn_<level>_overlap_<cell_type>.png, consensus_grn_<level>_overlap.png
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

from hira.src.config import (
    DISCOVERY_COHORTS,
    GRN_OVERLAP_DIR,
    MAJOR_CTS,
    palette_datasets_pretty,
    palette_major_cts,
    surrogate_names,
)
from hira.src.utils.plots import create_interaction_df, plot_interactions
from hira.src.utils.util import retrieve_net, retrieve_net_consensus

# min_subset_size per level; edges are ~1000x more numerous than TFs, so they need a
# much larger floor to keep the upset plot readable.
MIN_SUBSET_SIZE = {'source': 10, 'target': 10, 'edge': 1000}
CONSENSUS_MIN_SUBSET_SIZE = {'source': 10, 'target': 10, 'edge': 2000}
MIN_DEGREE = {'source': 1, 'target': 1, 'edge': 2}


def _with_edge(net):
    net['edge'] = net['source'] + '_' + net['target']
    return net


def plot_cohort_overlap(level='edge', cell_types=MAJOR_CTS, cohorts=DISCOVERY_COHORTS):
    """One upset per cell type, comparing the per-cohort networks."""
    for cell_type in cell_types:
        by_cohort = {}
        for dataset in cohorts:
            net = _with_edge(retrieve_net(dataset, cell_type))
            print(f'net: {cell_type} {dataset} {net.shape}')
            by_cohort[surrogate_names[dataset]] = net[level].tolist()

        plot_interactions(create_interaction_df(by_cohort),
                          min_degree=MIN_DEGREE[level],
                          min_subset_size=MIN_SUBSET_SIZE[level],
                          color_map=palette_datasets_pretty)
        plt.title(cell_type, y=1.15, weight='bold')
        _save(f'grn_{level}_overlap_{cell_type}.png')


def plot_consensus_overlap(level='edge', cell_types=MAJOR_CTS):
    """One upset comparing the consensus networks across cell types."""
    by_cell_type = {}
    for cell_type in cell_types:
        net = _with_edge(retrieve_net_consensus(cell_type=cell_type))
        print(f'consensus net: {cell_type} {net.shape}')
        by_cell_type[cell_type] = net[level].tolist()

    plot_interactions(create_interaction_df(by_cell_type),
                      min_degree=MIN_DEGREE[level],
                      min_subset_size=CONSENSUS_MIN_SUBSET_SIZE[level],
                      color_map=palette_major_cts)
    _save(f'consensus_grn_{level}_overlap.png')


def _save(name):
    file_name = os.path.join(GRN_OVERLAP_DIR, name)
    print(f'Saving to {file_name}')
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()
    return file_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', default='edge', choices=['edge', 'source', 'target'])
    args = parser.parse_args()

    plot_cohort_overlap(args.level)
    plot_consensus_overlap(args.level)
