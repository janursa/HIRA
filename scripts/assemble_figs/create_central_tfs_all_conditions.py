#!/usr/bin/env python
"""
One assembled figure: top central age-associated TFs vs every condition, CD4T | CD8T.

Columns per cell type: natural aging, SLE, JAK inhibition (op + CXCL9), LPS, IL-10 --
grouped by coloured bands. Colour = direction of the TF-activity change, star = FDR < 0.05.
Narrow bar next to each panel: network centrality of each TF.

Output: PLOTS_DIR/assembled/central_tfs_all_conditions.png
"""
import os
import sys
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from hira import retrieve_stats, retrieve_sig_stats
from hira.src.config import (PLOTS_DIR, palette_trend_2, palette_disease_effect,
                             palette_treatment, surrogate_names)
from hira.src.utils.util import retrieve_net_consensus
from hira.src.feature_association.plots import plot_tf_act_central_tfs

plt.rcParams.update({"figure.dpi": 150, "font.family": "Arial", "font.size": 10})

ANALYSIS_NAME = 'tfa_major_b'
CELL_TYPES = ['CD4T', 'CD8T']
TOP_N = 15
X_MARGIN = 0.03

COLUMNS = [
    dict(label='Age-associated',        group='Aging',          dataset=None,              comparison='aging'),
    dict(label='SLE (all)',             group='Disease',        dataset='perez_sle',       comparison='SLE',
         age_group='Both age groups'),
    dict(label='SLE (<40)',             group='Disease',        dataset='perez_sle',       comparison='SLE',
         age_group='Younger than 40'),
    dict(label='SLE (>40)',             group='Disease',        dataset='perez_sle',       comparison='SLE',
         age_group='Older than 40'),
    dict(label='SLE (<50)',             group='Disease',        dataset='perez_sle',       comparison='SLE',
         age_group='Younger than 50'),
    dict(label='SLE (>50)',             group='Disease',        dataset='perez_sle',       comparison='SLE',
         age_group='Older than 50'),
    dict(label='Ruxolitinib (ctr: DMSO)', group='JAK inhibition', dataset='op',           comparison='Ruxolitinib'),
    dict(label='Ruxolitinib (ctr: RPMI)', group='JAK inhibition', dataset='CXCL9',        comparison='Ruxolitinib (ctr: RPMI)'),
    dict(label='Ruxolitinib (ctr: LPS)',  group='JAK inhibition', dataset='CXCL9',        comparison='Ruxolitinib (ctr: LPS)'),
    dict(label='LPS (ctr: RPMI)',      group='Inflammation',   dataset='CXCL9',           comparison='LPS (ctr: RPMI)'),
    dict(label='IL-10 (ctr: PBS)',     group='Cytokine',       dataset='parsebioscience', comparison='IL-10'),
]

GROUP_COLORS = {'Aging': '#B03A2E', 'Disease': '#7B6C9B', 'JAK inhibition': '#3D7B8C',
                'Inflammation': '#C2793F', 'Cytokine': '#5E8C4A'}
PALETTE = {**palette_trend_2, **palette_disease_effect, **palette_treatment, 'Inconsistent': 'gray'}


def load_column(col, cell_type, aging):
    if col['dataset'] is None:
        return aging
    s = retrieve_stats(dataset=col['dataset'], analysis_name=ANALYSIS_NAME, multi_cohort=False)
    s = s[(s['cell_type'] == cell_type) & (s['comparison'] == col['comparison'])]
    if 'age_group' in col:
        s = s[s['age_group'] == col['age_group']]
    s = s.drop_duplicates(subset='gene').copy()
    s['analysis'] = col['label']
    return s


def panel_data(cell_type, stats_sig):
    """Top central age-associated TFs of one cell type, across all conditions."""
    aging = stats_sig[stats_sig['cell_type'] == cell_type].copy()
    aging['analysis'] = COLUMNS[0]['label']

    df = pd.concat([load_column(c, cell_type, aging) for c in COLUMNS])
    df = df[df['gene'].isin(aging['gene'].unique())]

    degree = retrieve_net_consensus(cell_type=cell_type).groupby('source').size().reset_index(name='degree')
    df = df.merge(degree, left_on='gene', right_on='source', how='left')
    top_tfs = df.drop_duplicates('gene').nlargest(TOP_N, 'degree')['gene']
    df = df[df['gene'].isin(top_tfs)].sort_values('degree', ascending=False)
    df['degree'] /= df['degree'].max()
    df['analysis'] = pd.Categorical(df['analysis'], categories=[c['label'] for c in COLUMNS], ordered=True)
    return df


def draw_panel(fig, gs_dots, gs_bar, df, cell_type):
    ax = fig.add_subplot(gs_dots)
    ax_c = fig.add_subplot(gs_bar)
    plot_tf_act_central_tfs(df, all_groups=[c['label'] for c in COLUMNS], palette_all=PALETTE,
                            axes=(ax, ax_c), ax2_margins={'x': 0.2, 'y': 0.02})
    ax.margins(x=X_MARGIN, y=0.05)
    ax.set_title(surrogate_names.get(cell_type, cell_type), fontsize=11, fontweight='bold', pad=52)

    for group in dict.fromkeys(c['group'] for c in COLUMNS):
        idx = [i for i, c in enumerate(COLUMNS) if c['group'] == group]
        x0, x1 = idx[0], idx[-1]
        ax.plot([x0 - 0.35, x1 + 0.35], [1.03, 1.03], transform=ax.get_xaxis_transform(),
                color=GROUP_COLORS[group], lw=2.5, solid_capstyle='butt', clip_on=False)
        ax.text((x0 + x1) / 2, 1.06, group, transform=ax.get_xaxis_transform(), rotation=45,
                ha='center', va='bottom', fontsize=8, fontweight='bold', color=GROUP_COLORS[group])
    return ax


def main():
    stats_sig = retrieve_sig_stats(analysis_name=ANALYSIS_NAME).drop_duplicates(subset=['cell_type', 'gene'])
    data = {ct: panel_data(ct, stats_sig) for ct in CELL_TYPES}

    fig = plt.figure(figsize=(10.0, 3.0))
    gs = gridspec.GridSpec(1, 5, width_ratios=[3.2, 0.4, 0.9, 3.2, 0.4], wspace=0.04)  # col 2 = spacer
    for i, ct in enumerate(CELL_TYPES):
        ax = draw_panel(fig, gs[3 * i], gs[3 * i + 1], data[ct], ct)
        if i:
            ax.set_ylabel('')
        print(f"  {ct}: {data[ct]['gene'].nunique()} TFs")

    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8, label=l)
               for l, c in PALETTE.items() if l != 'Inconsistent']
    handles.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
                          markeredgecolor='black', markersize=8, label='FDR < 0.05'))
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.92, 0.9), frameon=False, fontsize=9)

    out_dir = os.path.join(PLOTS_DIR, 'assembled')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'central_tfs_all_conditions.png')
    fig.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
