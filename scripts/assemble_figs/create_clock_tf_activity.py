#!/usr/bin/env python
"""CD4T clock TF activity: clock-derived vs empirical concordance + age trends of the top TFs.

The concordance subset of clock_interpretation_panel.py, whose panels it reuses.
Output: PLOTS_DIR/assembled/clock_tf_activity.png
"""
import os
import sys
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from hira.src.config import PLOTS_DIR, CLOCK_STATS_DIR  # noqa: E402
from hira.src.clock.run_exp_analysis import plot_trend_panel, top_clock_features  # noqa: E402
from create_clock_interpretation_panel import (TFA_COLORS,  # noqa: E402
                                               concordance_frames, scatter_panel)

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 9

CELL_TYPE = 'CD4T'
MIN_AGE = 30
# onek1k is the largest healthy cohort (932 donors >=30, ages to 97) -- the others leave 5-year
# bins too thin to separate trend from jitter. MIN_BIN_N drops the sparse 90+ bins, which
# otherwise anchor an end of each row's min-max colour scale on noise.
COHORT = 'onek1k'
BIN_SIZE = 5
MIN_BIN_N = 20


def main():
    fig = plt.figure(figsize=(5.5, 2.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=.55)

    ax = fig.add_subplot(gs[0, 0])
    ulm, _ = concordance_frames()
    scatter_panel(ax, ulm[ulm['cell_type'] == CELL_TYPE],
                  'Clock-derived TF activity \n(significance)',
                  'Age-associated TF activity \n(significance)', '',
                  {'consistent': 'Consistent', 'opposing': 'Opposing'}, 'x')
    # ponytail: scatter_panel puts the legend to the side -- move it above instead
    handles = ax.get_legend().legend_handles
    ax.get_legend().remove()
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, 1.0), frameon=False,
              fontsize=7, ncol=2, columnspacing=1)

    inner = gs[0, 1].subgridspec(1, 2, width_ratios=[.1, 1], wspace=.05)
    ax_sign, ax_heat = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
    tf_stats = pd.read_csv(f'{CLOCK_STATS_DIR}/clock_tf_activity.csv')
    tfs, scores = top_clock_features(None, tf_stats, CELL_TYPE, 'tf_activity')
    plot_trend_panel(ax_sign, ax_heat, CELL_TYPE, tfs, scores, feature_type='tf_activity',
                     dataset=COHORT, show_cbar=True, sign_colors=TFA_COLORS,
                     min_age=MIN_AGE, bin_size=BIN_SIZE, min_bin_n=MIN_BIN_N)
    ax_heat.tick_params(axis='x', labelsize=6)
    ax_sign.tick_params(axis='y', labelsize=6)
    ax_sign.set_ylabel('TF')
    leg = ax_heat.legend(handles=[Line2D([0], [0], color=c, lw=5, label=l) for c, l in
                                  zip(TFA_COLORS[::-1], ['Positive', 'Negative'])],
                         title='Regulatory influence on clock', loc='lower center',
                         bbox_to_anchor=(.5, 1.0), frameon=False, fontsize=7, handlelength=1,
                         ncol=2, columnspacing=1)
    leg.get_title().set_fontsize(7)

    out = os.path.join(PLOTS_DIR, 'assembled', 'clock_tf_activity.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    plt.close('all')
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
