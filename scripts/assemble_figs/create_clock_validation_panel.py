#!/usr/bin/env python
"""Clock validation panels: CV Spearman on the 3 held-out cohorts, actual-vs-predicted
scatters (CD4T, CD8T), and the benchmark against the published clock.

Reads persisted clock stats (cv_scores.csv, cv_predictions.csv, comparison_scores.csv).
Panel module -- drawn by create_clock_panel.py.
"""
import os
import sys
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from hira.src.config import CLOCK_STATS_DIR
from hira import surrogate_names, palette_datasets_pretty, colors_blind, get_clock_cell_types
from hira.src.clock.plots import plot_scatter_age_vs_predictedAge

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 9

SCATTER_CELL_TYPES = ['CD4T', 'CD8T']
PALETTE_MODELS = {'Wenchao': colors_blind[1], 'GRNdrived': colors_blind[0]}


def n_comp():
    return pd.read_csv(f'{CLOCK_STATS_DIR}/comparison_scores.csv')['dataset'].nunique()


def banner(fig, axes, label, dy=.34):
    """Bracket + label spanning `axes`, drawn in (sub)figure coords."""
    fig.canvas.draw()
    pos = [a.get_position() for a in axes]
    x0, x1 = min(p.x0 for p in pos), max(p.x1 for p in pos)
    y = max(p.y1 for p in pos) + dy
    fig.add_artist(plt.Line2D([x0, x0, x1, x1], [y - .03, y, y, y - .03],
                              color='black', lw=.8, transform=fig.transSubfigure))
    fig.text((x0 + x1) / 2, y + .02, label, ha='center', va='bottom', weight='bold')


def pretty(df):
    df = df.copy()
    df['dataset'] = df['dataset'].map(lambda n: surrogate_names.get(n, n))
    return df


def draw_performance(fig, gs, col0=0):
    """CV Spearman + actual-vs-predicted scatters. Returns (axes, legend handles)."""
    cell_types = get_clock_cell_types()
    scores = pretty(pd.read_csv(f'{CLOCK_STATS_DIR}/cv_scores.csv'))
    scores['cell_type'] = pd.Categorical(scores['cell_type'], categories=cell_types, ordered=True)
    preds = pretty(pd.read_csv(f'{CLOCK_STATS_DIR}/cv_predictions.csv'))

    ax = fig.add_subplot(gs[0, col0])
    axes = [ax]
    sns.stripplot(data=scores, x='cell_type', y='spearman', hue='dataset', dodge=False,
                  jitter=False, s=7, alpha=.7, linewidth=.5, edgecolor='gray',
                  palette=palette_datasets_pretty, ax=ax)
    sns.barplot(data=scores, x='cell_type', y='spearman', estimator='mean', errorbar=None,
                color=colors_blind[1], alpha=.5, ax=ax)
    ax.get_legend().remove()  # ponytail: cohort colors are labelled in the group legend
    ax.set(xlabel='', ylabel='Spearman')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.margins(x=.1, y=.1)
    ax.spines[['top', 'right']].set_visible(False)

    legend = None
    for i, cell_type in enumerate(SCATTER_CELL_TYPES):
        ax = fig.add_subplot(gs[0, col0 + 1 + i])
        axes.append(ax)
        plot_scatter_age_vs_predictedAge(preds[preds['cell_type'] == cell_type], dataset=cell_type,
                                         ax=ax, hue='dataset', palette=palette_datasets_pretty,
                                         s=20, alpha=.7)
        if ax.get_legend():
            legend = legend or ax.get_legend_handles_labels()[0]
            ax.get_legend().remove()
    return axes, legend


def draw_benchmark(fig, gs, col0=0):
    """Benchmark against the published clock. Returns (axes, legend handles)."""
    cell_types = get_clock_cell_types()
    comp = pd.read_csv(f'{CLOCK_STATS_DIR}/comparison_scores.csv')
    comp_datasets = comp['dataset'].unique()
    axes = [fig.add_subplot(gs[0, col0 + i]) for i in range(len(comp_datasets))]
    for ax, dataset in zip(axes, comp_datasets):
        sns.barplot(data=comp[comp['dataset'] == dataset], x='cell_type', y='Spearman',
                    hue='model', order=cell_types, alpha=.8, palette=PALETTE_MODELS, ax=ax)
        ax.set(xlabel='', ylabel='Spearman', title=surrogate_names.get(dataset, dataset))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.margins(x=.1, y=.1)
        ax.spines[['top', 'right']].set_visible(False)
        ax.get_legend().remove()
    return axes, axes[0].get_legend_handles_labels()[0]
