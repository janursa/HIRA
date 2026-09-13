#!/usr/bin/env python
"""Merge the donor-level panels for one TF into a single figure.

Left: CXCL9 RPMI -> LPS -> LPS + ruxolitinib. Right: OP DMSO -> Ruxolitinib.
Output: PLOTS_DIR/assembled/case_donors_merged_<tf>_<cell_type>.png
"""
import os
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from hira.src.config import PLOTS_DIR
from hira import retrieve_stats, retrieve_feature_data

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 9

ANALYSIS_NAME = 'tfa_major_b'
CELL_TYPE = 'CD4T'
CASE_TF = sys.argv[1] if len(sys.argv) > 1 else 'IRF1'
# (dataset, conditions on the x axis, x labels, brackets as (i, j, comparison))
PANELS = [
    ('CXCL9',
     ['RPMI', 'LPS', 'LPS + ruxolitinib'],
     ['RPMI', 'LPS', 'Ruxolitinib\n(ctr: LPS)'],
     [(0, 1, 'LPS (ctr: RPMI)'), (1, 2, 'Ruxolitinib (ctr: LPS)')]),
    ('op',
     ['DMSO', 'Ruxolitinib'],
     ['DMSO', 'Ruxolitinib\n(ctr: DMSO)'],
     [(0, 1, 'Ruxolitinib')]),
]
FIGSIZE = (3, 1.8)


def bracket(ax, i, j, pval, y, h):
    ax.plot([i, i, j, j], [y - h, y, y, y - h], 'k-', lw=1)
    ax.text((i + j) / 2, y, f'p={pval:.3f}', ha='center', va='bottom', fontsize=7)


def load(dataset):
    stats = retrieve_stats(dataset=dataset, analysis_name=ANALYSIS_NAME)
    stats = stats[(stats['gene'] == CASE_TF) & (stats['cell_type'] == CELL_TYPE)]
    pvals = stats.set_index('comparison')['p_value_adj'].to_dict()

    adata = retrieve_feature_data(dataset=dataset, analysis_name=ANALYSIS_NAME, cell_type=CELL_TYPE)
    df = adata.obs[['donor_id', 'condition']].copy()
    df[CASE_TF] = np.asarray(adata[:, CASE_TF].X).flatten()
    df = df.groupby(['donor_id', 'condition'], observed=True)[CASE_TF].mean().reset_index()
    return df, pvals


def main():
    fig, axes = plt.subplots(1, len(PANELS), figsize=FIGSIZE,
                             gridspec_kw={'width_ratios': [len(p[1]) for p in PANELS]})
    for ax, (dataset, conditions, labels, brackets) in zip(axes, PANELS):
        df, pvals = load(dataset)
        donors = sorted(df['donor_id'].unique())
        colors = dict(zip(donors, sns.color_palette('tab10', len(donors))))
        y_min, y_max = df[CASE_TF].min(), df[CASE_TF].max()
        y_range = y_max - y_min

        sub = df[df['condition'].isin(conditions)]
        # ponytail: fixed offsets per donor instead of random jitter, so the panels stay comparable
        jitter = np.linspace(-.12, .12, len(donors))
        x = [conditions.index(c) + jitter[donors.index(d)] for c, d in zip(sub['condition'], sub['donor_id'])]
        ax.scatter(x, sub[CASE_TF],
                   c=[colors[d] for d in sub['donor_id']], s=25, alpha=.8, linewidth=0)
        for k, (i, j, comparison) in enumerate(brackets):
            bracket(ax, i, j, pvals[comparison], y_max + (0.25 + 0.0 * k) * y_range, 0.05 * y_range)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim(y_min - 0.15 * y_range, y_max + 0.55 * y_range)
        ax.set_yticks([])
        ax.margins(x=0.3)
        ax.set_ylabel('TF activity')
        ax.spines[['top', 'right']].set_visible(False)

    out_dir = os.path.join(PLOTS_DIR, 'assembled')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'case_donors_merged_{CASE_TF}_{CELL_TYPE}.png')
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
