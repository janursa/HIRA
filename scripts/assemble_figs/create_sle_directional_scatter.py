#!/usr/bin/env python
"""
SLE TF activity directional consistency scatter — 2×3 assembled figure.

Rows: CD4T, CD8T
Columns: Both age groups | Younger than 50 | Older than 50
Title per panel (centred): "{cell_type}: {age_group}"
Shared x-label (bottom) and y-label (left).

Output: PLOTS_DIR/assembled/sle_directional_cd4t_cd8t.png
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

from hira import retrieve_stats, retrieve_sig_stats
from hira.src.config import PLOTS_DIR, surrogate_names

plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "Arial"

ANALYSIS_NAME = "tfa_major_b"
DATASET = "perez_sle"
ASSEMBLED_DIR = os.path.join(PLOTS_DIR, "assembled")
os.makedirs(ASSEMBLED_DIR, exist_ok=True)

CELL_TYPES = ["CD4T", "CD8T"]
AGE_GROUPS = ["Both age groups", "Younger than 50", "Older than 50"]

# Match color convention from plot_directional_consistency_scatter (agreement='same')
CONSISTENT_COLOR = "darkseagreen"
OPPOSING_COLOR = "indianred"
ASSOCIATION_COL = "-log10_p_adj"
PVALUE_COL = "p_value_adj"


def prepare_cell_data(stats_ct, stats_ref, cell_type):
    """
    Replicate the per-cell-type data preparation from plot_directional_consistency_scatter.
    Returns merged DataFrame with consistent/inconsistent flags and log10 columns.
    """
    ref_ct = stats_ref[stats_ref["cell_type"] == cell_type].copy()
    if len(ref_ct) == 0:
        return None

    sl_ct = stats_ct[stats_ct["gene"].isin(ref_ct["gene"])].copy()
    if len(sl_ct) == 0:
        return None

    sl_ct["abs_log10_p_adj_sl"] = sl_ct[PVALUE_COL].apply(lambda x: -np.log10(x + 1e-300))

    merged = ref_ct.merge(
        sl_ct[["gene", "slope", "comparison", PVALUE_COL]],
        on="gene",
        how="left",
        suffixes=("_ref", "_sl"),
    )
    merged = merged[~merged["slope_sl"].isna()].copy()
    if len(merged) == 0:
        return None

    merged["consistent"] = np.sign(merged["slope_ref"]) == np.sign(merged["slope_sl"])
    merged[f"{ASSOCIATION_COL}_sl"] = (
        -np.log10(merged[PVALUE_COL] + 1e-300) * np.sign(merged["slope_sl"])
    )
    merged[f"{ASSOCIATION_COL}_ref"] = (
        -np.log10(merged["meta_p_adj"] + 1e-300) * np.sign(merged["slope_ref"])
    )
    return merged


def draw_scatter(ax, cell_data, title, show_xlabel, show_ylabel):
    """Draw a single scatter panel into the given ax."""
    consistent = cell_data[cell_data["consistent"]]
    inconsistent = cell_data[~cell_data["consistent"]]

    s, lw = 10, 0.1
    if len(inconsistent) > 0:
        ax.scatter(
            inconsistent[f"{ASSOCIATION_COL}_sl"],
            inconsistent[f"{ASSOCIATION_COL}_ref"],
            c=OPPOSING_COLOR, s=s, alpha=0.6,
            edgecolors="darkred", linewidths=lw,
            label=f"Rejuvenation\n({len(inconsistent)} TFs)",
        )
    if len(consistent) > 0:
        ax.scatter(
            consistent[f"{ASSOCIATION_COL}_sl"],
            consistent[f"{ASSOCIATION_COL}_ref"],
            c=CONSISTENT_COLOR, s=s, alpha=0.6,
            edgecolors="darkgreen", linewidths=lw,
            label=f"Acceleration\n({len(consistent)} TFs)",
        )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.grid(False)

    x_max = max(abs(cell_data[f"{ASSOCIATION_COL}_sl"].min()),
                abs(cell_data[f"{ASSOCIATION_COL}_sl"].max())) * 1.05
    y_max = max(abs(cell_data[f"{ASSOCIATION_COL}_ref"].min()),
                abs(cell_data[f"{ASSOCIATION_COL}_ref"].max())) * 1.2
    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-y_max, y_max)

    # Title centred in the plot area
    ax.set_title(title, fontsize=9, fontweight="bold", pad=35, loc="center")

    # Labels only on outer edges
    ax.set_xlabel("SLE (significance)" if show_xlabel else "", fontsize=9)
    ax.set_ylabel("Natural aging (significance)" if show_ylabel else "", fontsize=9)

    # Per-panel legend above the scatter
    legend_elements = []
    if len(inconsistent) > 0:
        legend_elements.append(Line2D([0], [0], marker="o", color="w",
            markerfacecolor=OPPOSING_COLOR, markersize=5,
            markeredgecolor="darkred", markeredgewidth=0.5,
            label=f"Rejuvenation ({len(inconsistent)})"))
    if len(consistent) > 0:
        legend_elements.append(Line2D([0], [0], marker="o", color="w",
            markerfacecolor=CONSISTENT_COLOR, markersize=5,
            markeredgecolor="darkgreen", markeredgewidth=0.5,
            label=f"Acceleration ({len(consistent)})"))
    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, 1.35), frameon=False,
              fontsize=7, ncol=2, columnspacing=0.3)


def main():
    stats = retrieve_stats(analysis_name=ANALYSIS_NAME, dataset=DATASET, multi_cohort=False)
    stats_sig = stats[stats["is_significant"]].copy()
    # Aggregate within cell_type/gene/comparison as the original function does
    stats_agg = (
        stats_sig.groupby(["cell_type", "gene", "comparison", "age_group"])
        .agg({"slope": "mean", PVALUE_COL: "max"})
        .reset_index()
    )
    stats_ref = retrieve_sig_stats(analysis_name=ANALYSIS_NAME)
    stats_ref = (
        stats_ref.groupby(["cell_type", "gene", "comparison"])
        .agg({"slope": "mean", "meta_p_adj": "max"})
        .reset_index()
    )

    fig, axes = plt.subplots(
        2, 3,
        figsize=(7.5, 4),
        gridspec_kw={"hspace": 1.1, "wspace": 0.3},
    )

    for row, cell_type in enumerate(CELL_TYPES):
        for col, age_group in enumerate(AGE_GROUPS):
            ax = axes[row, col]
            stats_ct_ag = stats_agg[
                (stats_agg["cell_type"] == cell_type) &
                (stats_agg["age_group"] == age_group)
            ]
            cell_data = prepare_cell_data(stats_ct_ag, stats_ref, cell_type)

            ct_label = surrogate_names.get(cell_type, cell_type)
            title = f"{ct_label}: {age_group}"
            show_xlabel = (row == len(CELL_TYPES) - 1)
            show_ylabel = (col == 0)

            if cell_data is None or len(cell_data) == 0:
                ax.set_visible(False)
                print(f"  Skipping {cell_type} | {age_group}: no data")
                continue

            print(f"  {cell_type} | {age_group}: {len(cell_data)} TFs")
            draw_scatter(ax, cell_data, title, show_xlabel, show_ylabel)

    output_path = os.path.join(ASSEMBLED_DIR, "sle_directional_cd4t_cd8t.png")
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    print("Generating SLE directional 2×3 figure ...\n")
    main()
