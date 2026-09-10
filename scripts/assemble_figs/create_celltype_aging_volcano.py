#!/usr/bin/env python
"""Weight x meta p-value per cell type: which cell types are most affected by aging.

One dot per TF. x = mean regression slope over the discovery cohorts (signed:
>0 increases with age), y = meta-analytic FDR across those cohorts. Significant
TFs are coloured by cell type, the rest grey. A cell type that is strongly
affected is both wide (large weights) and tall (small meta p).

Output: PLOTS_DIR/assembled/celltype_aging_volcano.png
"""
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from hira.src.config import (DISCOVERY_COHORTS, MAJOR_CTS, PLOTS_DIR,  # noqa: E402
                             palette_trend)
from hira.src.feature_association.helper import retrieve_stats  # noqa: E402

plt.rcParams.update({"font.family": "Arial", "font.size": 10})
ANALYSIS_NAME = "tfa_major_b"


def load():
    """{cell type: one row per TF -- pooled rho + meta FDR}."""
    out = {}
    for ct in MAJOR_CTS:
        st = retrieve_stats(analysis_name=ANALYSIS_NAME, cell_type=ct)
        st = st[st["dataset"].isin(DISCOVERY_COHORTS)]
        g = st.groupby("gene").agg(weight=("pooled_rho", "first"),
                                   meta_p_adj=("meta_p_adj", "first"),
                                   is_significant=("is_significant", "first")).reset_index()
        g["y"] = -np.log10(g["meta_p_adj"].clip(lower=1e-300))
        out[ct] = g
    return out


def main():
    data = load()
    order = sorted(data, key=lambda ct: data[ct].loc[data[ct]["is_significant"], "y"].median(),
                   reverse=True)
    xmax = max(d["weight"].abs().max() for d in data.values()) * 1.1

    fig, axes = plt.subplots(1, len(order), figsize=(7.0, 1.8), sharex=True, sharey=True)
    for ax, ct in zip(axes, order):
        d = data[ct]
        sig = d["is_significant"].values
        up = sig & (d["weight"].values > 0)
        dn = sig & (d["weight"].values < 0)
        ax.scatter(d["weight"][~sig], d["y"][~sig], s=4, color="0.8", linewidth=0,
                   rasterized=True)
        for m, lab in ((up, "Increase in aging"), (dn, "Decrease in aging")):
            ax.scatter(d["weight"][m], d["y"][m], s=4, color=palette_trend[lab],
                       linewidth=0, rasterized=True, label=lab)
            ax.text(.97 if lab.startswith("Inc") else .03, .97, f"{int(m.sum())}",
                    transform=ax.transAxes, ha="right" if lab.startswith("Inc") else "left",
                    va="top", fontsize=9, color=palette_trend[lab])
        ax.axvline(0, ls="--", lw=.8, color="grey", zorder=0)
        ax.set_title(ct, fontsize=10, fontweight="bold", pad=4)
        ax.set_xlim(-xmax, xmax)
        ax.spines[["top", "right"]].set_visible(False)
        if ct != order[0]:
            ax.spines["left"].set_visible(False)
            ax.tick_params(left=False)

    axes[0].set_ylabel("Meta FDR\n(-log10)")
    fig.supxlabel("TF activity association with age (mean across cohorts)", fontsize=10, y=-.28)
    h, l = axes[0].get_legend_handles_labels()
    axes[-1].legend(h, l, loc="upper left", bbox_to_anchor=(1, 1), frameon=False, fontsize=9,
                    markerscale=2.5)
    fig.subplots_adjust(wspace=.18)

    out = os.path.join(PLOTS_DIR, "assembled", "celltype_aging_volcano.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    print(pd.DataFrame([dict(cell_type=ct, n_tf=len(data[ct]),
                             n_sig=int(data[ct]["is_significant"].sum()),
                             med_y_sig=data[ct].loc[data[ct]["is_significant"], "y"].median(),
                             med_abs_w_sig=data[ct].loc[data[ct]["is_significant"], "weight"].abs().median(),
                             max_y=data[ct]["y"].max())
                        for ct in order]).round(3).to_string(index=False))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
