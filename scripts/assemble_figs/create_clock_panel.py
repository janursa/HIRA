#!/usr/bin/env python
"""Clock figure: validation + concordance on the first row, interpretation on the second.

Panels come from create_clock_validation_panel.py and create_clock_interpretation_panel.py.
Output: PLOTS_DIR/assembled/clock_panel.png
"""
import os
import sys
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from hira.src.config import PLOTS_DIR
import create_clock_validation_panel as val
import create_clock_interpretation_panel as interp

FIGSIZE = (13, 5)
BANNER_DY = .34  # ponytail: same offset in every top group -> banners line up
BOTTOM = [(interp.draw_gene_trends, 'Clock genes with age'),
          (interp.draw_tf_trends, 'Clock TFs with age')]


def main():
    fig = plt.figure(figsize=FIGSIZE)
    top, bottom = fig.subfigures(2, 1, hspace=.42)

    # ponytail: one subfigure per group, top and bottom -- tight inside, wide between
    n_comp = val.n_comp()
    g1, g2, g3 = top.subfigures(1, 3, width_ratios=[3, n_comp, 2.6], wspace=.1)

    axes, handles = val.draw_performance(g1, g1.add_gridspec(1, 3, wspace=.55))
    val.banner(g1, axes, 'Performance on held-out samples', dy=BANNER_DY)
    interp.legend_below(g1, handles, ncol=4)

    axes, handles = val.draw_benchmark(g2, g2.add_gridspec(1, n_comp, wspace=.55))
    val.banner(g2, axes, 'Comparitive performance', dy=BANNER_DY)
    interp.legend_below(g2, handles, 'Model')

    axes = interp.draw_concordance(g3, g3.add_gridspec(1, 2, wspace=.75))
    val.banner(g3, axes, 'Concordance with aging', dy=BANNER_DY)

    for sub, (draw, label) in zip(bottom.subfigures(1, len(BOTTOM), wspace=.06), BOTTOM):
        axes = draw(sub, sub.add_gridspec(1, 2, wspace=.55))
        val.banner(sub, axes, label, dy=.13)

    out_dir = os.path.join(PLOTS_DIR, 'assembled')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'clock_panel.png')
    fig.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    plt.close('all')
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
