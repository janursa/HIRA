#!/usr/bin/env python
"""Donor-level clock age shift for every significant cytokine (parsebioscience) / drug (op).

Uses src.clock.plots.plot_group_strip -- one column per perturbation, one dot per donor,
donor-coloured, p above each column. Acceleration and rejuvenation merged into one axes,
separated by the colour banner on top. Rows = cell type, columns = cytokines | drugs.
Output: PLOTS_DIR/assembled/clock_perturbation_landscape.png
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
from hira.src.config import PLOTS_DIR, CLOCK_STATS_DIR  # noqa: E402
from hira.src.clock.plots import plot_group_strip  # noqa: E402

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10

CELL_TYPES = ["CD4T", "CD8T"]
ACC_COLOR, DEC_COLOR = "indianred", "darkseagreen"
ALPHA = 0.05
TOP_N = 15  # per direction; None = every significant perturbation
DATASETS = [("parsebioscience", "PBS", "Cytokines", "#5E8C4A"),
            ("op", "DMSO", "Drugs", "#3D7B8C")]
HEADROOM = 0.45  # extra y range so the rotated p-values stay clear of the banner
MOCK = {"op": "Ruxolitinib"}  # dataset -> the only rejuvenating compound shown by name


def selection(stats, cell_type):
    """Perturbations significant in this cell type: rejuvenating then accelerating,
    each block ranked by p-value."""
    sig = stats[(stats["cell_type"] == cell_type) & (stats["p_value"] < ALPHA)]
    blocks = [sig[sig["delta"] < 0].sort_values("p_value"),
              sig[sig["delta"] > 0].sort_values("p_value")]
    if TOP_N:
        blocks = [b.head(TOP_N) for b in blocks]
    # each block reads outward from the centre: p rises toward the middle, falls to the edges
    out = pd.concat([blocks[0], blocks[1][::-1]])
    return pd.Series(out["delta"].values, index=out["treatment"].values)


def mock_map(stats, sel, keep):
    """Rejuvenating compounds -> "Compound N", ranked by p, shared across cell types."""
    rej = set().union(*(s[s < 0].index for s in sel.values())) - {keep}
    order = (stats[stats["treatment"].isin(rej)].groupby("treatment")["p_value"].min()
             .sort_values().index)
    return {t: f"Compound {i}" for i, t in enumerate(order, 1)}


def banner(ax, delta):
    """Colour bands over the rejuvenating / accelerating blocks of columns."""
    for label, color, mask in (("Age rejuvenation", DEC_COLOR, delta < 0),
                               ("Age acceleration", ACC_COLOR, delta > 0)):
        idx = np.flatnonzero(mask.values)
        if not len(idx):
            continue
        x0, x1 = idx[0] - 0.4, idx[-1] + 0.4
        tr = ax.get_xaxis_transform()
        ax.plot([x0, x1], [1.15, 1.15], transform=tr, color=color, lw=3, clip_on=False,
                solid_capstyle="butt")
        # ponytail: a short block cannot centre its label without running off the axes
        narrow = len(idx) < 6
        ax.text(x0 if narrow else (x0 + x1) / 2, 1.17, label, transform=tr,
                ha="left" if narrow else "center", va="bottom", fontsize=10,
                weight="bold", color=color)


def main():
    cols = []
    for dataset, ctr, kind, color in DATASETS:
        stats = pd.read_csv(f"{CLOCK_STATS_DIR}/perturbation_{dataset}.csv")
        obs_all = pd.read_csv(f"{CLOCK_STATS_DIR}/predictions_{dataset}.csv")
        pval_map = {(r.cell_type, r.ctr, r.treatment): (r.p_value, r.delta)
                    for r in stats.itertuples()}
        sel = {ct: selection(stats, ct) for ct in CELL_TYPES}
        keep = MOCK.get(dataset)
        cols.append(dict(ctr=ctr, kind=kind, color=color, obs=obs_all, pval_map=pval_map,
                         sel=sel, keep=keep,
                         names=mock_map(stats, sel, keep) if keep else {}))

    widths = [max(len(c["sel"][ct]) for ct in CELL_TYPES) for c in cols]
    fig, axes = plt.subplots(len(CELL_TYPES), len(cols),
                             figsize=(.26 * sum(widths) + 1.6, 3.0 * len(CELL_TYPES)),
                             gridspec_kw={"width_ratios": widths})

    for j, c in enumerate(cols):
        for i, cell_type in enumerate(CELL_TYPES):
            ax, delta = axes[i, j], c["sel"][cell_type]
            order = list(delta.index)
            plot_group_strip(c["obs"][c["obs"]["cell_type"] == cell_type],
                             [(c["ctr"], t) for t in order], c["kind"], cell_type,
                             c["pval_map"], ctr=c["ctr"], order=order, ax=ax, max_len=18, p_as_stars=True,
                             name_mapping=c["names"],
                             highlight_treatments=[c["keep"]] if c["keep"] else None)
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + HEADROOM * (hi - lo))
            ax.set_ylabel("Shift in predicted age (yrs)" if j == 0 else "", fontsize=10)
            ax.set_title(cell_type, fontsize=10, weight="bold", pad=44)
            banner(ax, delta)
            # donor legend under the bottom row only (donors differ between datasets)
            if i == len(CELL_TYPES) - 1:
                h, l = ax.get_legend_handles_labels()
                ax.legend_.remove()
                ax.legend(h, l, loc="upper center", bbox_to_anchor=(.5, -.85), ncol=6,
                          frameon=False, fontsize=9, title=None, handletextpad=.2,
                          columnspacing=1.0)
            else:
                ax.legend_.remove()
            print(f"  {c['kind']} | {cell_type}: {len(order)} perturbations")

    fig.subplots_adjust(hspace=1.6, wspace=0.10, top=0.80)

    # dataset band above the top-row titles
    for j, c in enumerate(cols):
        pos = axes[0, j].get_position()
        y = 0.965
        fig.add_artist(Line2D([pos.x0, pos.x1], [y, y], color=c["color"], lw=3,
                              transform=fig.transFigure, solid_capstyle="butt"))
        fig.text((pos.x0 + pos.x1) / 2, y + .008, c["kind"], ha="center", va="bottom",
                 fontsize=11, weight="bold", color=c["color"])

    out = os.path.join(PLOTS_DIR, "assembled", "clock_perturbation_landscape.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
