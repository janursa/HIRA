#!/usr/bin/env python
"""
One assembled figure: GSEA (hallmark ORA) of TF-activity hits, panels side by side.

Panels: natural aging, SLE, Ruxolitinib (op), IL-10 (parsebioscience).
x = cell type (grouped by panel band), y = union of enriched terms,
colour = direction of the TF-activity change, size = -log10 FDR.

Output: PLOTS_DIR/assembled/gsea_panels.png
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from hira import retrieve_sig_stats
from hira.src.config import PLOTS_DIR, surrogate_names
from hira.src.pathway_analysis.util import gsea_func

plt.rcParams.update({"figure.dpi": 150, "font.family": "Arial", "font.size": 10})

ANALYSIS_NAME = "tfa_major_b"
CELL_ORDER = ["CD4T", "CD8T", "NK", "MONO", "B"]
UP_COLOR, DOWN_COLOR = "indianred", "steelblue"

PANELS = [
    dict(label="Aging",        dataset=None,              comparison="aging",       pval="meta_p_adj",   color="#8A8A8A"),
    dict(label="SLE",          dataset="perez_sle",       comparison="SLE",         pval="p_value_adj",  color="#7B6C9B"),
    dict(label="Ruxolitinib",  dataset="op",              comparison="Ruxolitinib", pval="p_value_adj",  color="#3D7B8C"),
    dict(label="IL-10",        dataset="parsebioscience", comparison="IL-10",       pval="p_value_adj",  color="#5E8C4A"),
]


def load_panel(panel):
    kw = {} if panel["dataset"] is None else dict(dataset=panel["dataset"], multi_cohort=False)
    stats = retrieve_sig_stats(analysis_name=ANALYSIS_NAME, **kw)
    stats = stats[stats["comparison"] == panel["comparison"]]
    res = gsea_func(stats, pvalue_col=panel["pval"], gene_sets="hallmark", feature_col="gene")
    if res is None or res.empty:
        return None
    res["panel"] = panel["label"]
    res["direction"] = np.where(res["trend"].str.startswith("Increase"), "Up", "Down")
    return res


def main():
    res = pd.concat([r for r in (load_panel(p) for p in PANELS) if r is not None], ignore_index=True)
    panels = [p for p in PANELS if p["label"] in set(res["panel"])]

    terms = (res.groupby("Term")["neg_log10_adj_pval"].max().sort_values().index.tolist())
    res["y"] = res["Term"].map({t: i for i, t in enumerate(terms)})

    # x positions: cell types laid out panel by panel
    x_pos, x_ticks, bands = {}, [], []
    x = 0
    for p in panels:
        cts = [c for c in CELL_ORDER if c in set(res.loc[res["panel"] == p["label"], "cell_type"])]
        start = x
        for ct in cts:
            x_pos[(p["label"], ct)] = x
            x_ticks.append((x, surrogate_names.get(ct, ct)))
            x += 1
        bands.append((p, start, x - 1))
        x += 0.8  # gap between panels
    res["x"] = [x_pos[(pl, ct)] for pl, ct in zip(res["panel"], res["cell_type"])]

    smax = res["neg_log10_adj_pval"].max()
    fig, ax = plt.subplots(figsize=(5, 4))
    for direction, color in (("Up", UP_COLOR), ("Down", DOWN_COLOR)):
        d = res[res["direction"] == direction]
        ax.scatter(d["x"], d["y"], s=20 + 110 * d["neg_log10_adj_pval"] / smax,
                   c=color, alpha=0.6, edgecolors="black", linewidths=0.3)

    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms, fontsize=9)
    ax.set_xticks([t[0] for t in x_ticks])
    ax.set_xticklabels([t[1] for t in x_ticks], rotation=45, ha="right", fontsize=9)
    ax.set_ylim(-0.8, len(terms) + 1.3)
    ax.set_xlim(-0.8, bands[-1][2] + 0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    for p, x0, x1 in bands:
        ax.plot([x0 - 0.35, x1 + 0.35], [len(terms) + 0.5] * 2, color=p["color"], lw=2.5,
                solid_capstyle="butt", clip_on=False)
        ax.text((x0 + x1) / 2, len(terms) + 0.65, p["label"], ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=p["color"])

    size_vals = np.round(np.linspace(res["neg_log10_adj_pval"].min(), smax, 3), 1)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=UP_COLOR, markeredgecolor="black",
                      markeredgewidth=0.3, markersize=7, alpha=0.6, label="Increase"),
               Line2D([0], [0], marker="o", color="none", markerfacecolor=DOWN_COLOR, markeredgecolor="black",
                      markeredgewidth=0.3, markersize=7, alpha=0.6, label="Decrease")]
    l1 = ax.legend(handles=handles, title="Direction", loc="upper left", bbox_to_anchor=(1.02, 1.0),
                   frameon=False, fontsize=9, title_fontsize=9)
    size_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor="grey", markeredgecolor="none",
                           markersize=np.sqrt(20 + 110 * v / smax), alpha=0.6, label=f"{v}")
                    for v in size_vals]
    ax.legend(handles=size_handles, title="$-$log$_{10}$ FDR", loc="upper left", bbox_to_anchor=(1.02, 0.72),
              frameon=False, fontsize=9, title_fontsize=9, labelspacing=1.1)
    ax.add_artist(l1)

    out_dir = os.path.join(PLOTS_DIR, "assembled")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "gsea_panels.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
