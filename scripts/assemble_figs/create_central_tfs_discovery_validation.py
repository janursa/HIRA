#!/usr/bin/env python
"""Merge the op (discovery) and CXCL9 (validation) central-TF dotplots into one panel.

Output: PLOTS_DIR/assembled/central_tfs_discovery_validation_<cell_type>.png
"""
import os
import sys
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from hira.src.config import PLOTS_DIR, palette_trend, palette_treatment
from hira import retrieve_stats, retrieve_sig_stats
from hira.src.utils.util import retrieve_net_consensus
from hira.src.feature_association.plots import plot_tf_act_central_tfs

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10

ANALYSIS_NAME = 'tfa_major_b'
CELL_TYPE = 'CD4T'
TOP_N = 15
# (dataset, group label, renamed comparisons) -- column order follows this list
DISCOVERY = [('op', {'Ruxolitinib': 'Ruxolitinib (ctr: DMSO)'})]
VALIDATION = [('CXCL9', {c: c for c in ['Ruxolitinib (ctr: RPMI)', 'LPS (ctr: RPMI)', 'Ruxolitinib (ctr: LPS)']})]


def condition_stats(specs):
    out = []
    for dataset, rename in specs:
        s = retrieve_stats(dataset=dataset, analysis_name=ANALYSIS_NAME, multi_cohort=False)
        s = s[(s['cell_type'] == CELL_TYPE) & s['comparison'].isin(rename)]
        s['analysis'] = s['comparison'].map(rename)
        out.append(s)
    df = pd.concat(out)
    order = [new for _, rename in specs for new in rename.values()]
    return df, order


def main():
    aging = retrieve_sig_stats(analysis_name=ANALYSIS_NAME).drop_duplicates(subset=['cell_type', 'gene'])
    aging = aging[aging['cell_type'] == CELL_TYPE].copy()
    aging['analysis'] = 'Age-associated'

    disc, disc_cols = condition_stats(DISCOVERY)
    val, val_cols = condition_stats(VALIDATION)

    df = pd.concat([aging, disc, val])
    df = df[df['gene'].isin(aging['gene'].unique())]

    degree = retrieve_net_consensus(cell_type=CELL_TYPE).groupby('source').size().reset_index(name='degree')
    df = df.merge(degree, left_on='gene', right_on='source', how='left')
    top_tfs = df.drop_duplicates('gene').nlargest(TOP_N, 'degree')['gene']
    df = df[df['gene'].isin(top_tfs)].sort_values('degree', ascending=False)
    df['degree'] /= df['degree'].max()

    columns = ['Age-associated'] + disc_cols + val_cols
    df['analysis'] = pd.Categorical(df['analysis'], categories=columns, ordered=True)

    palette = {**palette_trend, **palette_treatment}
    plot_tf_act_central_tfs(df, all_groups=columns, palette_all=palette, figsize=(3, 3),
                            ax2_margins={'x': 0.2, 'y': 0.02}, plot_centrality=True,
                            width_ratios=(1.6, .4))
    ax = plt.gcf().axes[0]

    # group brackets above the dot columns
    for label, cols in [('Discovery', ['Age-associated'] + disc_cols), ('Validation', val_cols)]:
        x0, x1 = columns.index(cols[0]), columns.index(cols[-1])
        ax.plot([x0, x0, x1, x1], [1.02, 1.05, 1.05, 1.02], transform=ax.get_xaxis_transform(),
                color='black', lw=0.8, clip_on=False)
        ax.text((x0 + x1) / 2, 1.09, label, transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', weight='bold')

    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8, label=l)
               for l, c in palette.items() if l != 'Inconsistent']
    plt.gcf().legend(handles=handles, loc='upper left', bbox_to_anchor=(0.62, 0.0),
                     frameon=False, fontsize=9)

    out_dir = os.path.join(PLOTS_DIR, 'assembled')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'central_tfs_discovery_validation_{CELL_TYPE}.png')
    plt.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
