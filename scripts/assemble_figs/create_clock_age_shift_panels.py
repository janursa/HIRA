#!/usr/bin/env python
"""Merge the donor-level clock age-shift strips of op (discovery) and CXCL9 (validation).

Reads the saved clock tables (predictions_*.csv, perturbation_*.csv), so it does not rerun
the clock. Output: PLOTS_DIR/assembled/clock_age_shift_<cell_type>.png
"""
import os
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from hira.src.config import PLOTS_DIR, CLOCK_STATS_DIR

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 9

CELL_TYPE = 'CD4T'
# (dataset, [(ctr, treatment, x label)], rejuvenation?)
PANELS = [
    ('op', [('DMSO', 'Ruxolitinib', 'Ruxolitinib\n(ctr: DMSO)')], True),
    ('CXCL9', [('RPMI', 'LPS', 'LPS\n(ctr: RPMI)')], False),
    ('CXCL9', [('LPS', 'LPS + ruxolitinib', 'Ruxolitinib\n(ctr: LPS)'),
               ('RPMI', 'RPMI + ruxolitinib', 'Ruxolitinib\n(ctr: RPMI)')], True),
]
GROUPS = [('Discovery', [0]), ('Validation', [1, 2])]
FIGSIZE = (3.5, 2.2)


def donor_shifts(dataset, ctr, treatment, rejuv):
    obs = pd.read_csv(f'{CLOCK_STATS_DIR}/predictions_{dataset}.csv')
    obs = obs[obs['cell_type'] == CELL_TYPE]
    pivot = obs.pivot_table(index='donor_id', columns='condition', values='predicted_age')
    diff = pivot[treatment] - pivot[ctr]
    return -diff if rejuv else diff


def pvalue(dataset, ctr, treatment):
    stats = pd.read_csv(f'{CLOCK_STATS_DIR}/perturbation_{dataset}.csv')
    row = stats[(stats['cell_type'] == CELL_TYPE) & (stats['ctr'] == ctr)
                & (stats['treatment'] == treatment)]
    return row['p_value'].iloc[0]


def main():
    fig, axes = plt.subplots(1, len(PANELS), figsize=FIGSIZE,
                             gridspec_kw={'width_ratios': [len(p[1]) for p in PANELS]})
    for ax, (dataset, comparisons, rejuv) in zip(axes, PANELS):
        series = [donor_shifts(dataset, ctr, tr, rejuv) for ctr, tr, _ in comparisons]
        donors = sorted(set().union(*[s.index for s in series]))
        colors = dict(zip(donors, sns.color_palette('husl', len(donors))))
        for i, s in enumerate(series):
            ax.scatter([i] * len(s), s.values, c=[colors[d] for d in s.index],
                       s=25, alpha=.85, linewidth=0)
            p = pvalue(dataset, *comparisons[i][:2])
            ax.annotate(f'{p:.2}', (i, s.max()), textcoords='offset points', xytext=(0, 7),
                        ha='center', va='bottom', fontsize=7)
        ax.set_xticks(range(len(comparisons)))
        ax.set_xticklabels([c[2] for c in comparisons], rotation=45, ha='right')
        ax.set_ylabel('Age rejuvenation (yrs)' if rejuv else 'Age acceleration (yrs)', fontsize=8)
        ax.margins(x=.45 if len(comparisons) == 1 else .35, y=.25)
        ax.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    # ponytail: group brackets drawn in figure coords after tight_layout, so they span axes
    top = max(ax.get_position().y1 for ax in axes)
    for label, idx in GROUPS:
        x0 = axes[idx[0]].get_position().x0
        x1 = axes[idx[-1]].get_position().x1
        y = top + 0.10
        fig.add_artist(plt.Line2D([x0, x0, x1, x1], [y - .02, y, y, y - .02],
                                  color='black', lw=.8, transform=fig.transFigure))
        fig.text((x0 + x1) / 2, y + .01, label, ha='center', va='bottom', weight='bold')

    out_dir = os.path.join(PLOTS_DIR, 'assembled')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'clock_age_shift_{CELL_TYPE}.png')
    plt.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
