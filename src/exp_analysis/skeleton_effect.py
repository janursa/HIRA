"""How the motif skeleton reshapes the consensus GRNs.

Rebuilds the consensus network per cell type unfiltered, promoter-motif filtered and
ATAC+motif skeleton filtered (no cache -- the consensus cache doesn't encode the skeleton) and compares
edge/TF/target counts, density and weights, plus a PCA of TF out-degree profiles showing
whether filtering moves a network more than cell-type identity does.

Usage: python src/exp_analysis/skeleton_effect.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from hira.src.config import (
    MAJOR_CTS,
    SKELETON_EFFECT_DIR,
    SKELETON_EFFECT_PLOTS_DIR,
)
from hira.src.utils.util import retrieve_net_consensus

VARIANTS = {'Unfiltered': None, 'Promoter filtered': 'promotor', 'Skeleton filtered': 'skeleton'}
VARIANT_COLOR = {'Unfiltered': '#999999', 'Promoter filtered': '#0072B2', 'Skeleton filtered': '#D55E00'}
MARKERS = dict(zip(MAJOR_CTS, ['o', 's', '^', 'D', 'v']))

plt.rcParams.update({'font.family': 'Arial', 'font.size': 10})


def collect():
    """-> (stats dataframe, {(cell_type, variant): TF out-degree Series})."""
    stats, degrees = [], {}
    for cell_type in MAJOR_CTS:
        for name, skeleton in VARIANTS.items():
            net = retrieve_net_consensus(cell_type=cell_type, skeleton=skeleton, cache=False)
            n_tf, n_target = net['source'].nunique(), net['target'].nunique()
            stats.append({
                'cell_type': cell_type, 'variant': name,
                'n_edges': len(net), 'n_tfs': n_tf, 'n_targets': n_target,
                'density': len(net) / (n_tf * n_target),
                'targets_per_tf': len(net) / n_tf,
                'mean_abs_weight': net['weight'].abs().mean(),
                'frac_positive': (net['weight'] > 0).mean(),
            })
            degrees[(cell_type, name)] = net['source'].value_counts()
            print(f'{cell_type} {name}: {len(net)} edges', flush=True)
    return pd.DataFrame(stats), degrees


def plot_stats(stats):
    metrics = [('n_edges', 'Edges'), ('n_tfs', 'TFs'), ('n_targets', 'Targets'), ('density', 'Density')]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3))
    for ax, (col, label) in zip(axes, metrics):
        wide = stats.pivot(index='cell_type', columns='variant', values=col).loc[MAJOR_CTS, list(VARIANT_COLOR)]
        x = np.arange(len(MAJOR_CTS))
        for i, variant in enumerate(VARIANT_COLOR):
            ax.bar(x + (i - 1) * 0.28, wide[variant], width=0.28, color=VARIANT_COLOR[variant],
                   label=variant if ax is axes[-1] else None)
        ax.set_xticks(x)
        ax.set_xticklabels(MAJOR_CTS, rotation=45, ha='right')
        ax.set_ylabel(label)
        ax.margins(x=0.05, y=0.15)
        ax.spines[['top', 'right']].set_visible(False)
    axes[-1].legend(frameon=False, fontsize=9, loc='upper left', bbox_to_anchor=(1, 1), title='Network')
    fig.tight_layout()
    return _save(fig, 'skeleton_stats.png')


def plot_pca(degrees):
    """Each network -> its TF out-degree profile (fraction of edges per TF); PCA over those."""
    keys = list(degrees)
    profiles = pd.DataFrame({k: degrees[k] / degrees[k].sum() for k in keys}).fillna(0).T
    pca = PCA(n_components=2).fit(profiles)
    xy = pca.transform(profiles)
    var = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(4, 3))
    for (cell_type, variant), (px, py) in zip(keys, xy):
        ax.scatter(px, py, color=VARIANT_COLOR[variant], marker=MARKERS[cell_type], s=45,
                   edgecolor='black', linewidth=0.3)
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=v) for v, c in VARIANT_COLOR.items()]
    handles += [plt.Line2D([], [], marker=m, ls='', color='black', label=ct) for ct, m in MARKERS.items()]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xlabel(f'PC1 ({var[0]:.0f}%)')
    ax.set_ylabel(f'PC2 ({var[1]:.0f}%)')
    ax.margins(0.15)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return profiles, _save(fig, 'skeleton_pca.png')


def variance_explained(profiles):
    """Between-group spread of the TF out-degree profiles, per grouping."""
    labels = pd.DataFrame(list(profiles.index), columns=['cell_type', 'variant'])
    total = ((profiles - profiles.mean()) ** 2).to_numpy().sum()
    rows = []
    for factor in ['variant', 'cell_type']:
        between = sum(len(idx) * ((profiles.iloc[idx].mean() - profiles.mean()) ** 2).sum()
                      for idx in labels.groupby(factor).groups.values())
        rows.append({'factor': factor, 'frac_variance': between / total})
    return pd.DataFrame(rows)


def _save(fig, name):
    path = os.path.join(SKELETON_EFFECT_PLOTS_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')
    return path


if __name__ == '__main__':
    stats, degrees = collect()
    stats.to_csv(f'{SKELETON_EFFECT_DIR}/skeleton_stats.csv', index=False)
    print(stats.round(5).to_string(index=False))

    plot_stats(stats)
    profiles, _ = plot_pca(degrees)
    ve = variance_explained(profiles)
    ve.to_csv(f'{SKELETON_EFFECT_DIR}/skeleton_pca_variance.csv', index=False)
    print(ve.to_string(index=False))
    print(f'Tables: {SKELETON_EFFECT_DIR}')
