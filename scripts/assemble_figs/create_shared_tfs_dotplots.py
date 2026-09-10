#!/usr/bin/env python
"""Merge feature_vs_datasets panels for CD4T/CD8T/NK into one figure with a shared color scale.

Output: PLOTS_DIR/assembled/feature_vs_datasets_shared_<analysis_name>.png
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from hira.src.config import PLOTS_DIR, DISCOVERY_COHORTS, surrogate_names, cmap_trend, get_config_fa
from hira.src.feature_association.helper import retrieve_stats, retrieve_sig_stats
from hira.src.utils.util import retrieve_net_consensus
from hira.src.utils.plots import create_interaction_df

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10

ANALYSIS_NAME = "tfa_major_b"
CELL_TYPES = ['CD4T', 'CD8T', 'NK']
X_MARGIN = 0.15   # x padding around the dataset columns
Y_MARGIN = 0.08   # vertical padding around the gene rows
FIG_HEIGHT = 3.0
FIG_WIDTH = 7.0
SIZES = (60, 110)
N_TOTAL = 15      # rows kept overall, split consistent | divergent
REQUIRED = ['GATA3', 'SATB1']


def panel_data(cell_type, features, feature_type):
    """Same data path as plot_features_vs_datasets, without the plotting."""
    stats_t = retrieve_stats(analysis_name=ANALYSIS_NAME, cell_type=cell_type)
    stats_t = stats_t[stats_t['dataset'].isin(DISCOVERY_COHORTS) & stats_t['gene'].isin(features)]

    net = retrieve_net_consensus(cell_type=cell_type)
    group_col = 'target' if feature_type == 'gene_expression' else 'source'
    c = net.groupby(group_col).size()
    c = (c / c.max()).reset_index(name='degree').rename(columns={group_col: 'gene'})
    stats_t = stats_t.merge(c, on='gene', how='left')

    stats_t['gene'] = pd.Categorical(stats_t['gene'], categories=list(features), ordered=True)
    stats_t = stats_t.sort_values('gene')
    stats_t['neg_log10_adj_pval'] = -np.log10(stats_t['p_value_adj']).replace([np.inf, -np.inf], 1e-20)
    stats_t['dataset'] = pd.Categorical(stats_t['dataset'], categories=DISCOVERY_COHORTS, ordered=True)
    return stats_t


def select(all_stats):
    """Top TFs per block: consistent (one sign across cohorts/cell types) and divergent (sign flips)."""
    s = -np.log10(all_stats['p_value_adj'].clip(lower=1e-300)) * np.sign(all_stats['slope'])
    g = s.groupby(all_stats['gene'].astype(str))
    strength, agree = g.apply(lambda v: v.abs().mean()), g.apply(lambda v: abs(np.sign(v).sum()) / len(v))
    rank = pd.DataFrame({'consistency': agree * strength, 'divergence': (1 - agree) * strength})
    keep = []
    per_side = [(N_TOTAL + 1) // 2, N_TOTAL // 2]
    for n_keep, c in zip(per_side, rank.columns):
        # a required TF is pinned to whichever block it actually scores on
        pin = [t for t in REQUIRED if t in rank.index and rank.loc[t].idxmax() == c]
        rest = [t for t in rank[c].sort_values(ascending=False).index if t not in pin]
        keep.append(pin + rest[:n_keep - len(pin)])
    return keep


def resort(df, features):
    df = df[df['gene'].isin(features)].copy()
    df['gene'] = pd.Categorical(df['gene'], categories=features, ordered=True)
    return df.sort_values('gene')


def main():
    feature_type = get_config_fa(ANALYSIS_NAME)['feature_type']

    # shared significant features across the three cell types
    stats_sig = retrieve_sig_stats(analysis_name=ANALYSIS_NAME).drop_duplicates(subset=['cell_type', 'gene'])
    interaction_df = create_interaction_df(stats_sig.groupby('cell_type')['gene'].apply(list).to_dict())
    mask = interaction_df[CELL_TYPES].sum(axis=1) == len(CELL_TYPES)
    features = list(mask[mask].index.unique())
    print(f"{len(features)} shared features: {features}")

    data = {ct: panel_data(ct, features, feature_type) for ct in CELL_TYPES}
    consistent, divergent = select(pd.concat(data.values()))
    order = consistent + divergent
    print(f"consistent: {consistent}\ndivergent:  {divergent}")
    data = {ct: resort(df, order) for ct, df in data.items()}

    # shared scales
    all_stats = pd.concat(data.values())
    abs_max = np.abs(all_stats['slope']).max()
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    cmap = plt.get_cmap(cmap_trend)
    p_min, p_max = all_stats['neg_log10_adj_pval'].min(), all_stats['neg_log10_adj_pval'].max()
    size = lambda p: SIZES[0] + (SIZES[1] - SIZES[0]) * (p - p_min) / (p_max - p_min)

    n_ct = len(CELL_TYPES)
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    gs = gridspec.GridSpec(1, 2 * n_ct + 1, width_ratios=[1, 0.6] * n_ct + [0.8], wspace=0.3)

    for i, ct in enumerate(CELL_TYPES):
        df = data[ct]
        ax = fig.add_subplot(gs[2 * i])
        x = df['dataset'].cat.codes
        y = df['gene'].map({g: len(order) - 1 - j for j, g in enumerate(order)})
        ax.scatter(x, y, c=[cmap(norm(v)) for v in df['slope']], s=[size(p) for p in df['neg_log10_adj_pval']],
                   edgecolor='black', linewidth=0.1, alpha=1)
        for xi, yi, p in zip(x, y, df['neg_log10_adj_pval']):
            if p >= 1.4:  # ponytail: same fixed p-value cutoff the original dotplot uses
                ax.text(xi, yi - .1, '*', ha='center', va='center', fontsize=6, weight='bold', zorder=10)
        ax.set_xticks(range(len(DISCOVERY_COHORTS)))
        ax.set_xticklabels([surrogate_names.get(d, d) for d in DISCOVERY_COHORTS], rotation=45, ha='right')
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order[::-1] if i == 0 else [])
        ax.set_ylabel('')
        ax.set_title(ct, fontsize=10, fontweight='bold', pad=8)
        ax.margins(x=X_MARGIN, y=Y_MARGIN)
        ax.spines[['top', 'right']].set_visible(False)

        ax_c = fig.add_subplot(gs[2 * i + 1])
        deg = df.drop_duplicates('gene').set_index('gene')['degree'].reindex(order).fillna(0)
        ax_c.barh(range(len(order) - 1, -1, -1), deg.values, color='#56B4E9', alpha=0.7, height=0.7)
        ax_c.set_ylim(ax.get_ylim())
        ax_c.set_yticks([])
        # ponytail: 2 ticks + rotation, the panel is too narrow for auto-placed decimals
        ax_c.set_xticks([0, deg.max()])
        ax_c.set_xticklabels(['0', f'{deg.max():.2f}'], rotation=45, ha='right')
        if i == n_ct - 1:  # ponytail: label the shared quantity once, on the last panel
            ax_c.set_xlabel('Centrality\n(out-degree)' if feature_type != 'gene_expression' else 'Centrality\n(in-degree)')
        ax_c.spines[['top', 'right', 'left']].set_visible(False)

    # shared colorbar
    ax_legend = fig.add_subplot(gs[-1])
    ax_legend.set_axis_off()
    cax = inset_axes(ax_legend, width="130%", height="6%", loc='center left', bbox_to_anchor=(0.1, 0, 1, 1),
                     bbox_transform=ax_legend.transAxes)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=[-abs_max, 0, abs_max],
                        format='%.1f')
    cbar.set_label('Correlation\nwith aging')
    cbar.ax.tick_params(labelsize=10)

    out_dir = os.path.join(PLOTS_DIR, 'assembled')
    os.makedirs(out_dir, exist_ok=True)
    file_name = os.path.join(out_dir, f'feature_vs_datasets_shared_{ANALYSIS_NAME}.png')
    fig.savefig(file_name, dpi=300, bbox_inches='tight', transparent=True)
    print(f'Saved figure to {file_name}')


if __name__ == '__main__':
    main()
