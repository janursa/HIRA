#!/usr/bin/env python
"""
All UpSet interaction plots in one figure (3 + 5 panels), drawn natively into subfigures.

  top     sets are cell types : age-associated TFs | clock features | consensus GRN edges
  bottom  sets are cohorts    : GRN edge overlap, one panel per cell type

Set memberships are always rebuilt from the current results -- never cached.
Output: PLOTS_DIR/assembled/interactions_all.png
"""
import os
import sys
import warnings

import matplotlib.pyplot as plt
import upsetplot
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from hira.src.config import (CLOCK_V, CLOCKS_DIR, DISCOVERY_COHORTS, MAJOR_CTS,
                             PLOTS_DIR, REF_TFA_ANALYSIS, USE_LOCAL_CLOCK, palette_datasets_pretty,
                             palette_major_cts, surrogate_names)

ASSEMBLED_DIR = os.path.join(PLOTS_DIR, 'assembled')

plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'axes.labelsize': 8, 'axes.linewidth': .6})

BAR_COLOR = '#8c8279'
DOT_COLOR = '#3f3a36'
N_FIRST_ROW = 3  # cell-type panels on top, the per-cell-type cohort panels below
COL_W = 0.34     # width per intersection column, kept equal across all panels
PANEL_PAD = 1.0  # fixed width per panel for set names / y axis

MIN_SHARE = 0.05  # bars below this share of the union are dropped ...
MIN_BARS = 4      # ... unless that would leave fewer than this many

# (set key, title, set axis, colour map, min_degree)
PANELS = [
    ('aging_tfs', 'Age-associated TFs', 'cell types', palette_major_cts, 1),
    ('clock', 'Clock features', 'cell types', palette_major_cts, 1),
    ('consensus', 'Consensus GRN edges', 'cell types', palette_major_cts, 2),
] + [(f'cohorts_{ct}', surrogate_names.get(ct, ct), 'cohorts',
      palette_datasets_pretty, 2) for ct in MAJOR_CTS]

thousands = FuncFormatter(lambda v, _: f'{v/1000:g}k' if v >= 1000 else f'{v:g}')


def build_sets():
    """Boolean membership tables, read fresh from the current results every run."""
    from grnimmuneclock import retrieve_function

    from hira import get_clock_cell_types, retrieve_net_consensus, retrieve_sig_stats
    from hira.src.utils.plots import create_interaction_df
    from hira.src.utils.util import retrieve_net

    def edges(net):
        return (net['source'] + '_' + net['target']).tolist()

    sig = retrieve_sig_stats(analysis_name=REF_TFA_ANALYSIS).drop_duplicates(subset=['cell_type', 'gene'])
    sets = {
        'aging_tfs': create_interaction_df(sig.groupby('cell_type')['gene'].apply(list).to_dict()),
        'clock': create_interaction_df({
            ct: list(retrieve_function(cell_type=ct, version=CLOCK_V,
                                       model_dir=CLOCKS_DIR if USE_LOCAL_CLOCK else None)[1])
            for ct in get_clock_cell_types()}),
        'consensus': create_interaction_df({ct: edges(retrieve_net_consensus(cell_type=ct))
                                            for ct in MAJOR_CTS}),
    }
    for ct in MAJOR_CTS:
        sets[f'cohorts_{ct}'] = create_interaction_df(
            {surrogate_names[d]: edges(retrieve_net(d, ct)) for d in DISCOVERY_COHORTS})
    return sets


def panel_cut(df, min_degree):
    """(min_subset_size, n bars): MIN_SHARE of the union, relaxed to keep MIN_BARS bars."""
    d = upsetplot.from_indicators(indicators=lambda a: a == True, data=df)
    size = d.groupby(level=list(range(d.index.nlevels))).size()
    sizes = sorted(size[size.index.to_frame().sum(axis=1).values >= min_degree], reverse=True)
    cut = MIN_SHARE * len(df)
    n = sum(v >= cut for v in sizes)
    if n < MIN_BARS and len(sizes) >= MIN_BARS:
        cut, n = sizes[MIN_BARS - 1], MIN_BARS
    return cut, n


def scale_dots(ax, frac=0.6):
    """upsetplot hardcodes the dot size when element_size is None; rescale it to the cell size."""
    n_col = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    n_row = abs(ax.get_ylim()[1] - ax.get_ylim()[0])
    bb = ax.get_window_extent()
    d = frac * min(bb.width / n_col, bb.height / n_row) * 72 / ax.figure.dpi
    for coll in ax.collections:
        if isinstance(coll, PathCollection):
            coll.set_sizes([d ** 2])
        else:
            coll.set_linewidth(d * 0.14)


def draw(subfig, df, title, set_axis, color_map, min_degree, min_subset_size, show_ylabel):
    """One UpSet inside a SubFigure. element_size=None -> the grid fills the subfigure."""
    axes = upsetplot.plot(
        upsetplot.from_indicators(indicators=lambda a: a == True, data=df), fig=subfig,
        show_counts=False, show_percentages=False, sort_by='degree',
        min_degree=min_degree, min_subset_size=min_subset_size,
        facecolor=DOT_COLOR, other_dots_color=.10, shading_color=.03, with_lines=True,
        element_size=None, intersection_plot_elements=5, totals_plot_elements=2,
    )
    ints, totals, matrix = axes['intersections'], axes['totals'], axes['matrix']

    # intersection bars: absolute size on the y-axis, share of the union above each bar
    n_items = len(df)
    for bar in ints.patches:
        bar.set_facecolor(BAR_COLOR)
        bar.set_edgecolor('none')
        ints.annotate(f'{bar.get_height() / n_items:.0%}',
                      (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      ha='center', va='bottom', fontsize=6, color='0.35',
                      xytext=(0, 1.5), textcoords='offset points')
    ints.set_ylim(0, ints.get_ylim()[1] * 1.15)  # headroom for the labels
    ints.set_ylabel('Intersection size' if show_ylabel else '')
    ints.yaxis.set_major_formatter(thousands)
    ints.grid(axis='y', lw=.4, color='0.85')
    ints.set_axisbelow(True)
    ints.spines[['top', 'right']].set_visible(False)
    ints.tick_params(length=2, width=.6)

    # set totals: one coloured bar per set, annotated with the actual count
    for bar, name in zip(totals.patches, [t.get_text() for t in matrix.get_yticklabels()]):
        bar.set_facecolor(color_map[name])
        bar.set_edgecolor('none')
        totals.annotate(f'{int(round(bar.get_width())):,}',
                        (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        ha='right', va='center', fontsize=6, color='0.25',
                        xytext=(-3, 0), textcoords='offset points')
    hi = totals.get_xlim()[0]  # axis is inverted; pad so full-length bars clear the set names
    totals.set_xlim(hi, -hi * 0.22)
    totals.set_axis_off()

    for lbl in matrix.get_yticklabels():
        lbl.set_fontsize(7.5)
    subfig.suptitle(title, fontsize=9, fontweight='bold', y=1.0)
    subfig.text(0.5, 0.955, f'sets: {set_axis}', fontsize=6.5, style='italic',
                color='0.45', ha='center', va='top')
    return matrix


def main():
    sets = build_sets()

    cuts = {key: panel_cut(sets[key], mind) for key, _, _, _, mind in PANELS}
    rows = [PANELS[:N_FIRST_ROW], PANELS[N_FIRST_ROW:]]
    # ponytail: panels hold wildly different column counts -> width each by its own columns
    row_w = [[PANEL_PAD + COL_W * cuts[p[0]][1] for p in panels] for panels in rows]
    fig_w = max(sum(w) for w in row_w)
    fig = plt.figure(figsize=(fig_w, 2.7 * len(rows)))
    matrices = []

    rowfigs = fig.subfigures(len(rows), 1, hspace=0.25)
    for rowfig, panels, widths in zip(rowfigs, rows, row_w):
        slack = fig_w - sum(widths)  # short row -> trailing blank cell, panels keep their width
        cells = rowfig.subfigures(1, len(widths) + 1, wspace=0.02,
                                  width_ratios=widths + [max(slack, 1e-3)])
        for i, (sub, (key, *rest)) in enumerate(zip(cells, panels)):
            matrices.append(draw(sub, sets[key], *rest, cuts[key][0], show_ylabel=(i == 0)))
        cells[-1].set_visible(False)

    # bracket over the whole second row
    rf, frac = rowfigs[1], sum(row_w[1]) / fig_w
    rf.add_artist(Line2D([0, 0, frac, frac], [1.06, 1.10, 1.10, 1.06], color='0.35', lw=.8,
                         transform=rf.transSubfigure, clip_on=False))
    rf.text(frac / 2, 1.12, 'Inferred GRNs', ha='center', va='bottom', fontsize=9,
            fontweight='bold', transform=rf.transSubfigure)

    fig.canvas.draw()
    for ax in matrices:
        scale_dots(ax)

    os.makedirs(ASSEMBLED_DIR, exist_ok=True)
    out = os.path.join(ASSEMBLED_DIR, 'interactions_all.png')
    fig.savefig(out, dpi=300, bbox_inches='tight', transparent=True)
    print('Saved assembled figure:', out)


if __name__ == '__main__':
    main()
