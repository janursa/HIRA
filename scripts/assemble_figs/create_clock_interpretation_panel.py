#!/usr/bin/env python
"""Clock interpretation panels: clock-derived vs empirical TF activity, the clock's most
important genes (importance vs aging direction), and the age trends of its top TFs.

Reads persisted stats (tf_regulation_concordance.csv, clock_gene_importance.csv,
clock_tf_activity.csv) plus the binned feature data.
Panel module -- drawn by create_clock_panel.py.
"""
import os
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from hira.src.config import CLOCK_STATS_DIR, surrogate_names
from hira import CLOCK_PLOTS_DIR, DISCOVERY_COHORTS
from hira.src.clock.run_exp_analysis import plot_trend_panel, top_clock_features
from hira.src.feature_association.plots import _annotate_extreme_tfs

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 9

CELL_TYPES = ['CD4T', 'CD8T']
N_TOP_COEF = 15
BIN_SIZE = 10
COHORT = 'aida'
# aging direction (bars) and clock TF-activity sign (strip), kept in separate hue families
AGING_COLORS = {-1: '#E52B50', 1: '#B0BF1A'}
TFA_COLORS = ('#5E2B97', '#2A9D8F')  # (negative, positive) regulatory influence
COEF_COLORS = ('#4062BB', '#F4A259')  # (negative, positive) clock weight
STYLE = [('consistent', 'darkseagreen', 'darkgreen'), ('opposing', 'indianred', 'darkred')]


def legend_below(fig, handles, title=None, dy=.04, ncol=2):
    """Legend just under the group it belongs to, in (sub)figure coords."""
    fig.canvas.draw()
    # ponytail: box everything drawn in this (sub)figure -- tick labels, xlabel and colorbar included
    from matplotlib.transforms import Bbox
    bb = Bbox.union([a.get_tightbbox() for a in fig.axes]).transformed(fig.transSubfigure.inverted())
    x, y = (bb.x0 + bb.x1) / 2, bb.y0 - dy
    leg = fig.legend(handles=handles, title=title, loc='upper center', bbox_to_anchor=(x, y),
                     frameon=False, fontsize=6.5, ncol=ncol, handlelength=.8, handletextpad=.4,
                     columnspacing=.8, labelspacing=.2, borderpad=0, borderaxespad=0)
    if title:
        leg.get_title().set_fontsize(6.5)
        leg.get_title().set_ha('center')


def signed(p, direction):
    return -np.log10(np.clip(p, 1e-300, None)) * np.sign(direction)


def concordance_frames():
    """Both concordance panels reduced to the same (gene, x, y, agreement) shape."""
    ulm = pd.read_csv(f'{CLOCK_PLOTS_DIR}/tf_regulation_concordance.csv')
    ulm = ulm[ulm['emp_sig']].assign(
        gene=lambda d: d['tf'],
        **{'-log10_p_adj_sl': lambda d: signed(d['ulm_padj'], d['ulm_score']),
           '-log10_p_adj_ref': lambda d: signed(d['emp_padj'], d['emp_slope'])})
    ulm['agreement'] = np.where(ulm['agree'], 'consistent', 'opposing')

    ora = pd.read_csv(f'{CLOCK_PLOTS_DIR}/tf_ora_clock_vs_empirical.csv')
    ora = ora[ora['ora_sig'] & ora['emp_sig'].astype(bool)].dropna(subset=['emp_padj']).assign(
        gene=lambda d: d['tf'],
        **{'-log10_p_adj_sl': lambda d: signed(d['emp_padj'], d['emp_slope']),
           '-log10_p_adj_ref': lambda d: signed(d['ora_pval'], np.where(d['direction'] == 'up', 1, -1))})
    ora['agreement'] = np.where(ora['dir_match'].astype(bool), 'consistent', 'opposing')
    return ulm, ora


def scatter_panel(ax, sub, xlabel, ylabel, title, labels, rank_on):
    for key, color, edge in STYLE:
        g = sub[sub['agreement'] == key]
        if len(g):
            ax.scatter(g['-log10_p_adj_sl'], g['-log10_p_adj_ref'], c=color, s=10, alpha=.6,
                       edgecolors=edge, linewidths=.1, label=f'{labels[key]} \n ({len(g)} TFs)')
    ax.axhline(0, color='black', lw=.8, alpha=.5)
    ax.axvline(0, color='black', lw=.8, alpha=.5)
    ax.set_xlim(-sub['-log10_p_adj_sl'].abs().max() * 1.05, sub['-log10_p_adj_sl'].abs().max() * 1.05)
    ax.set_ylim(-sub['-log10_p_adj_ref'].abs().max() * 1.2, sub['-log10_p_adj_ref'].abs().max() * 1.2)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, y=1.3)  # ponytail: y= here, set_position gets overridden on draw
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(False)
    h, l = ax.get_legend_handles_labels()
    ax.legend(handles=[Line2D([0], [0], marker='o', color='w', markersize=5,
                              markerfacecolor=a.get_facecolor()[0], markeredgecolor=a.get_edgecolor()[0],
                              markeredgewidth=.5, label=b) for a, b in zip(h, l)],
              loc='lower left', bbox_to_anchor=(0, 1.02), frameon=False, fontsize=7,
              ncol=2, columnspacing=-.2)
    _annotate_extreme_tfs(ax, sub, '-log10_p_adj', rank_on=rank_on)


def gene_panel(ax, sub):
    """Top clock genes: bar length = out-of-sample importance, colour = aging correlation."""
    sub = sub.iloc[::-1]
    y = np.arange(len(sub))
    ax.barh(y, sub['importance_t'], height=.7, linewidth=0,
            color=[AGING_COLORS[d] for d in np.sign(sub['pooled_rho'])])
    ax.set_yticks(y)
    ax.set_yticklabels(sub['gene'])
    ax.tick_params(axis='y', length=0, pad=1)
    ax.set(xlabel='Importance to aging clock', ylabel='Gene', ylim=(-.7, len(sub) - .3))
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.grid(False)


def draw_gene_trends(fig, gs, col0=0):
    """a) age trends of the clock's top-weighted genes (sign strip = ridge coefficient)."""
    gene_stats = pd.read_csv(f'{CLOCK_STATS_DIR}/clock_gene_importance.csv')
    sign_axes, heat_axes = [], []
    for c, cell_type in enumerate(CELL_TYPES):
        inner = gs[0, col0 + c].subgridspec(1, 2, width_ratios=[.1, 1], wspace=.05)
        ax_sign, ax_heat = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
        sign_axes.append(ax_sign)
        heat_axes.append(ax_heat)
        sub = gene_stats.query('cell_type == @cell_type')
        sub = sub.reindex(sub['clock_coef'].abs().sort_values(ascending=False).index).head(N_TOP_COEF)
        plot_trend_panel(ax_sign, ax_heat, cell_type, sub['gene'].tolist(), sub['clock_coef'].values,
                         feature_type='gene_expression', dataset=COHORT,
                         show_cbar=cell_type == CELL_TYPES[-1], sign_colors=COEF_COLORS, bin_size=BIN_SIZE)
        ax_heat.set_title(surrogate_names.get(cell_type, cell_type), pad=8)
        ax_sign.tick_params(axis='y', labelsize=6)
        ax_sign.set_ylabel('Gene')
    legend_below(fig, [Line2D([0], [0], color=c, lw=5, label=l) for c, l in
                                  zip(COEF_COLORS[::-1], ['Positive', 'Negative'])], 'Clock weight')
    return sign_axes + heat_axes


def draw_concordance(fig, gs, col0=0):
    """b) clock-derived vs empirical TF activity."""
    ulm, _ = concordance_frames()
    axes = []
    for c, cell_type in enumerate(CELL_TYPES):
        ax = fig.add_subplot(gs[0, col0 + c])
        axes.append(ax)
        scatter_panel(ax, ulm[ulm['cell_type'] == cell_type],
                      'Clock-derived TF activity \n(significance)',
                      'Age-associated TF activity \n(significance)',
                      surrogate_names.get(cell_type, cell_type),
                      {'consistent': 'Consistent', 'opposing': 'Opposing'}, 'x')
    return axes


def draw_gene_importance(fig, gs, col0=0):
    """c) the genes the clock leans on: importance, coloured by aging direction."""
    gene_stats = pd.read_csv(f'{CLOCK_STATS_DIR}/clock_gene_importance.csv')
    axes = []
    for c, cell_type in enumerate(CELL_TYPES):
        ax = fig.add_subplot(gs[0, col0 + c])
        axes.append(ax)
        genes, _ = top_clock_features(gene_stats, None, cell_type, 'gene_expression')
        gene_panel(ax, gene_stats.query('cell_type == @cell_type').set_index('gene')
                   .loc[genes].reset_index())
        ax.set_title(surrogate_names.get(cell_type, cell_type), pad=8)
        ax.tick_params(axis='y', labelsize=6)
    legend_below(fig, [Line2D([0], [0], color=AGING_COLORS[d], lw=5, label=l)
                             for d, l in [(1, 'Increases with age'), (-1, 'Decreases with age')]])
    return axes


def draw_tf_trends(fig, gs, col0=0):
    """d) age trends of the clock's top TFs."""
    tf_stats = pd.read_csv(f'{CLOCK_STATS_DIR}/clock_tf_activity.csv')
    sign_axes, heat_axes = [], []
    for c, cell_type in enumerate(CELL_TYPES):
        inner = gs[0, col0 + c].subgridspec(1, 2, width_ratios=[.1, 1], wspace=.05)
        ax_sign, ax_heat = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
        sign_axes.append(ax_sign)
        heat_axes.append(ax_heat)
        tfs, scores = top_clock_features(None, tf_stats, cell_type, 'tf_activity')
        # ponytail: one colorbar for the pair, else it lands on the next panel's labels
        plot_trend_panel(ax_sign, ax_heat, cell_type, tfs, scores, feature_type='tf_activity',
                         dataset=COHORT, show_cbar=cell_type == CELL_TYPES[-1],
                         sign_colors=TFA_COLORS, bin_size=BIN_SIZE)
        ax_heat.set_title(surrogate_names.get(cell_type, cell_type), pad=8)
        ax_sign.tick_params(axis='y', labelsize=6)
        ax_sign.set_ylabel('TF')
    legend_below(fig, [Line2D([0], [0], color=c, lw=5, label=l) for c, l in
                                  zip(TFA_COLORS[::-1], ['Positive', 'Negative'])],
                 'Regulatory influence on clock')
    return sign_axes + heat_axes
