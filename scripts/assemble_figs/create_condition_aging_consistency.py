#!/usr/bin/env python
"""
One assembled figure: directional consistency with natural aging across all conditions.

Rows: cell types (CD4T, CD8T).  Columns: condition (SLE, JAK inhibition, LPS, IL-10,
independent aging cohort), grouped by banner bands on top.
x = condition/cohort signed significance, y = natural-aging signed significance.
Colour = sign agreement with aging (shared legend, one per figure).

Output: PLOTS_DIR/assembled/condition_aging_consistency.png
"""

import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from hira import retrieve_stats, retrieve_sig_stats
from hira.src.feature_association.plots import plot_overlap
from hira.src.config import PLOTS_DIR, surrogate_names, palette_trend_2, REF_GE_ANALYSIS

plt.rcParams.update({"figure.dpi": 150, "font.family": "Arial", "font.size": 10})

ANALYSIS_NAME = "tfa_major_b"
CELL_TYPES = ["CD4T", "CD8T"]
SAME_COLOR, OPP_COLOR = "indianred", "darkseagreen"
PVAL = "p_value_adj"
MIN_TFS = 5  # panels with fewer overlapping TFs say nothing
N_LABEL = 3  # discordant TFs named per end in the activity-vs-expression panel
BAR_SCALE, PANEL_SCALE = 0.55, 0.82  # shrink of the overlap bars / their panels

# column spec: (label, group band, dataset, comparison, keep only significant, age_group)
COLUMNS = [
    dict(label="SLE\n(all donors)",       group="Disease",            dataset="perez_sle",       comparison="SLE",                     sig=True,  age_group="Both age groups"),
    dict(label="SLE\n(age < 50)",         group="Disease",            dataset="perez_sle",       comparison="SLE",                     sig=True,  age_group="Younger than 50"),
    dict(label="SLE\n(age \u2265 50)",      group="Disease",            dataset="perez_sle",       comparison="SLE",                     sig=True,  age_group="Older than 50"),
    dict(label="SLE\n(age < 40)",         group="Disease",            dataset="perez_sle",       comparison="SLE",                     sig=True,  age_group="Younger than 40"),
    dict(label="SLE\n(age \u2265 40)",      group="Disease",            dataset="perez_sle",       comparison="SLE",                     sig=True,  age_group="Older than 40"),
    dict(label="Ruxolitinib\n(ctr: DMSO)", group="JAK inhibition",   dataset="op",              comparison="Ruxolitinib",             sig=True),
    dict(label="Ruxolitinib\n(ctr: RPMI)", group="JAK inhibition",   dataset="CXCL9",           comparison="Ruxolitinib (ctr: RPMI)", sig=False),
    dict(label="Ruxolitinib\n(ctr: LPS)",  group="JAK inhibition",   dataset="CXCL9",           comparison="Ruxolitinib (ctr: LPS)",  sig=False),
    dict(label="LPS\n(ctr: RPMI)",       group="Inflammation",       dataset="CXCL9",           comparison="LPS (ctr: RPMI)",         sig=False),
    dict(label="IL-10\n(ctr: PBS)",      group="Cytokine",           dataset="parsebioscience", comparison="IL-10",                   sig=True),
    dict(label="Aging\n(SoundLife cohort)",    group="Replication",        dataset="soundlife",       comparison="aging",                   sig=False),
    dict(label="SoundLife\n(other cell types)", group="Replication",     dataset="soundlife",       comparison="aging",                   sig=False,
         cell_types=["NK", "MONO"]),
    dict(label="Overlap with\nreference aging", group="Replication",    dataset="soundlife",       comparison="aging",                   sig=True,  kind="overlap"),
    dict(label="Overlap\n(other cell types)", group="Replication",      dataset="soundlife",       comparison="aging",                   sig=True,  kind="overlap",
         cell_types=["NK", "MONO"], title="Overlap with\nreference aging", legend=True),
    dict(label="Aging trend\n(meta $-$log$_{10}$ p)", group="TF activity vs expression", kind="actexpr"),
]

GROUP_COLORS = {"Disease": "#7B6C9B", "JAK inhibition": "#3D7B8C", "Inflammation": "#C2793F",
                "Cytokine": "#5E8C4A", "Replication": "#8A8A8A",
                "TF activity vs expression": "#B05A7A"}


def load_overlap(col, stats_ref):
    """(condition sig TFs, reference aging sig TFs) per cell type -- input for plot_overlap."""
    stats = retrieve_stats(analysis_name=ANALYSIS_NAME, dataset=col["dataset"], multi_cohort=False)
    stats = stats[stats["is_significant"] & (stats["comparison"] == col["comparison"])]
    out = {}
    for ct in col.get("cell_types", CELL_TYPES):
        cond = (stats[stats["cell_type"] == ct][["gene", "cell_type", "slope"]]
                .drop_duplicates("gene").rename(columns={"slope": "slope_condition"})
                .reset_index(drop=True))
        ref = (stats_ref[stats_ref["cell_type"] == ct][["gene", "cell_type", "slope"]]
               .drop_duplicates("gene").reset_index(drop=True))
        if len(cond) and len(ref):
            out[ct] = (cond, ref)
    return out


def draw_overlap(ax, d, legend):
    if d is None:
        ax.set_visible(False)
        return
    cond, ref = d
    plot_overlap(cond, ref, agreement="same", how="left", col="cell_type", ax=ax,
                 legend=legend, legend_loc=(0.0, 0.0))
    if legend:  # re-lay it out horizontally in the gap below the block
        leg = ax.get_legend()
        h = leg.legend_handles
        labels = [t.get_text() for t in leg.get_texts()]
        leg.remove()
        ax.legend(h, labels, ncol=3, loc="upper right", bbox_to_anchor=(1.45, -0.50),
                  frameon=False, fontsize=9, columnspacing=1.2, handletextpad=0.5)
    # plot_overlap hardcodes the bar width -- thin the bars in place, keeping their positions
    for bar in ax.patches:
        c, w = bar.get_x() + bar.get_width() / 2, bar.get_width() * BAR_SCALE
        bar.set_width(w)
        bar.set_x(c - w / 2)
    ax.set_ylabel("Number of TFs", fontsize=10)
    ax.tick_params(labelsize=9)
    pos = ax.get_position()  # narrower/shorter box than the scatter panels
    ax.set_position([pos.x0 + pos.width * (1 - PANEL_SCALE) / 2,
                     pos.y0 + pos.height * (1 - PANEL_SCALE) / 2,
                     pos.width * PANEL_SCALE, pos.height * PANEL_SCALE])


def load_actexpr(col):
    """Per cell type: signed meta -log10 p of each TF's aging trend, activity vs expression."""
    def signed(analysis):
        # unfiltered: filtering both sides to their own significant sets before the merge
        # kept only TFs significant in BOTH, which made every point look concordant
        st = retrieve_stats(analysis_name=analysis)
        st = st.groupby(["cell_type", "gene"]).agg({"slope": "mean", "meta_p_adj": "max"}).reset_index()
        st["s"] = -np.log10(st["meta_p_adj"] + 1e-300) * np.sign(st["slope"])
        return st[["cell_type", "gene", "s"]]

    m = signed(ANALYSIS_NAME).merge(signed(REF_GE_ANALYSIS), on=["cell_type", "gene"],
                                    suffixes=("_tfa", "_ge"))
    return {ct: g for ct, g in m.groupby("cell_type") if ct in CELL_TYPES and len(g) >= MIN_TFS}


def draw_actexpr(ax, d):
    if d is None or len(d) == 0:
        ax.set_visible(False)
        return
    x, y = d["s_ge"], d["s_tfa"]
    # ponytail: neutral colour -- the figure legend encodes agreement with aging, not with expression
    ax.scatter(x, y, c="#4C72B0", s=8, alpha=0.6, edgecolors="none")
    # discordant TFs (significant in activity, not in expression) circled at both ends,
    # top N per end labelled in the empty opposite quadrant
    disc = d[(d["s_tfa"].abs() > 1.4) & (d["s_ge"].abs() < 1.4)]
    ax.scatter(disc["s_ge"], disc["s_tfa"], s=30, facecolors="none", edgecolors="black",
               linewidths=0.5, zorder=3)
    x_lo, x_hi = x.min(), x.max()
    y_lo, y_hi = y.min(), y.max()
    for end in (1, -1):
        tail = disc[np.sign(disc["s_tfa"]) == end]
        tail = tail.loc[tail["s_tfa"].abs().sort_values(ascending=False).index].head(N_LABEL)
        x_lab = x_lo if end > 0 else 0.35 * x_hi
        y_start, y_step = (y_hi, -0.09 * (y_hi - y_lo)) if end > 0 else (y_lo, 0.09 * (y_hi - y_lo))
        for k, (_, row) in enumerate(tail.iterrows()):
            y_lab = y_start + k * y_step
            ax.plot([row["s_ge"], x_lab], [row["s_tfa"], y_lab], lw=0.3, color="grey",
                    alpha=0.7, zorder=2)
            ax.text(x_lab, y_lab, row["gene"], fontsize=5, ha="left", va="center", zorder=4)
    for v in (-1.4, 1.4):
        ax.axvline(v, ls="--", color="red", alpha=0.4, lw=0.5)
        ax.axhline(v, ls="--", color="red", alpha=0.4, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.set_xlabel("Expression", fontsize=9, labelpad=2)
    ax.set_ylabel("TF activity", fontsize=9, labelpad=2)


def load_column(col, stats_ref):
    """Signed -log10 p for condition (x) vs natural aging (y), per cell type."""
    stats = retrieve_stats(analysis_name=ANALYSIS_NAME, dataset=col["dataset"], multi_cohort=False)
    if col["sig"]:
        stats = stats[stats["is_significant"]]
    if "age_group" in col:
        stats = stats[stats["age_group"] == col["age_group"]]
    stats = stats[stats["comparison"] == col["comparison"]]
    stats = stats.groupby(["cell_type", "gene"]).agg({"slope": "mean", PVAL: "max"}).reset_index()

    out = {}
    for ct in col.get("cell_types", CELL_TYPES):
        ref_ct = stats_ref[stats_ref["cell_type"] == ct]
        m = ref_ct.merge(stats[stats["cell_type"] == ct], on="gene", suffixes=("_ref", "_c"))
        if len(m) == 0:
            continue
        m["x"] = -np.log10(m[PVAL] + 1e-300) * np.sign(m["slope_c"])
        m["y"] = -np.log10(m["meta_p_adj"] + 1e-300) * np.sign(m["slope_ref"])
        m["same"] = np.sign(m["slope_c"]) == np.sign(m["slope_ref"])
        if len(m) >= MIN_TFS:
            out[ct] = m
    return out


def draw(ax, d):
    if d is None or len(d) == 0:
        ax.text(0.5, 0.55, f"< {MIN_TFS} TFs", ha="center", va="center", fontsize=9,
                color="grey", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        return
    for mask, color, edge in ((~d["same"], OPP_COLOR, "darkgreen"), (d["same"], SAME_COLOR, "darkred")):
        sub = d[mask]
        if len(sub):
            ax.scatter(sub["x"], sub["y"], c=color, s=8, alpha=0.6, edgecolors=edge, linewidths=0.1)

    ax.axhline(0, color="black", lw=0.6, alpha=0.4)
    ax.axvline(0, color="black", lw=0.6, alpha=0.4)
    x_max = d["x"].abs().max() * 1.15
    y_max = d["y"].abs().max() * 1.15
    ax.set_xlim(-x_max, x_max); ax.set_ylim(-y_max, y_max)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.grid(False)

    n_same, n_opp = int(d["same"].sum()), int((~d["same"]).sum())
    ax.text(0.5, 1.14, f"Concordant ({n_same})", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, color=SAME_COLOR)
    ax.text(0.5, 1.01, f"Opposing ({n_opp})", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, color=OPP_COLOR)


def main():
    stats_ref = retrieve_sig_stats(analysis_name=ANALYSIS_NAME)
    stats_ref = stats_ref.groupby(["cell_type", "gene"]).agg({"slope": "mean", "meta_p_adj": "max"}).reset_index()

    loaders = {"overlap": lambda c: load_overlap(c, stats_ref), "actexpr": load_actexpr}
    data = {c["label"]: loaders.get(c.get("kind"), lambda c: load_column(c, stats_ref))(c)
            for c in COLUMNS}
    columns = [c for c in COLUMNS if data[c["label"]]]  # drop conditions with no usable panel

    # stacked blocks: replication, SLE, perturbations; one row per cell type inside each
    order = [["Replication", "TF activity vs expression"], ["Disease"]]
    blocks = [[c for c in columns if c["group"] in g] for g in order]
    placed = {g for gs in order for g in gs}
    blocks.append([c for c in columns if c["group"] not in placed])
    ncol = max(len(b) for b in blocks)
    nrow = len(CELL_TYPES)
    fig = plt.figure(figsize=(1.9 * ncol + 1.0, 1.85 * nrow * len(blocks) + 1.0))
    # one gridspec per block so the gap between blocks is set independently of the row gap
    fig_h = 1.85 * nrow * len(blocks) + 1.0
    band_off = (32 + 26) / (fig_h * 72)  # title pad + two title lines, in figure fraction
    top, bottom, block_gap = 0.955, 0.055, band_off + 0.075
    height = (top - bottom - block_gap * (len(blocks) - 1)) / len(blocks)
    axes = np.vstack([
        np.array(gridspec.GridSpec(nrow, ncol, figure=fig, left=0.085, right=0.995,
                                   top=top - b * (height + block_gap),
                                   bottom=top - b * (height + block_gap) - height,
                                   wspace=0.55, hspace=0.55).subplots())
        for b in range(len(blocks))])

    for b, block in enumerate(blocks):
        for j in range(ncol):
            if j >= len(block):
                for i in range(nrow):
                    axes[b * nrow + i, j].set_visible(False)
                continue
            col = block[j]
            col_cts = col.get("cell_types", CELL_TYPES)
            for i, ct in enumerate(col_cts):
                ax = axes[b * nrow + i, j]
                if col.get("kind") == "overlap":
                    draw_overlap(ax, data[col["label"]].get(ct),
                                 legend=col.get("legend", False) and i == nrow - 1)
                elif col.get("kind") == "actexpr":
                    draw_actexpr(ax, data[col["label"]].get(ct))
                else:
                    draw(ax, data[col["label"]].get(ct))
                if i == 0:
                    ax.set_title(col.get("title", col["label"]), fontsize=10, pad=32,
                                 color=GROUP_COLORS[col["group"]])
                if (j == 0 or col_cts != CELL_TYPES) and col.get("kind") not in ("overlap", "actexpr"):
                    ax.set_ylabel(surrogate_names.get(ct, ct), fontsize=10, fontweight="bold", labelpad=6)
                print(f"  {col['label']!r} | {ct}: {len(data[col['label']].get(ct, []))} TFs")
            for i in range(len(col_cts), nrow):
                axes[b * nrow + i, j].set_visible(False)

        # group bands above this block's column titles
        band_y = axes[b * nrow, 0].get_position().y1 + band_off
        for group in dict.fromkeys(c["group"] for c in block):
            idx = [j for j, c in enumerate(block) if c["group"] == group]
            x0 = axes[b * nrow, idx[0]].get_position().x0
            x1 = axes[b * nrow, idx[-1]].get_position().x1
            fig.add_artist(Line2D([x0, x1], [band_y, band_y], color=GROUP_COLORS[group], lw=2.5,
                                  solid_capstyle="butt"))
            fig.text((x0 + x1) / 2, band_y + 0.006, group, ha="center", va="bottom", fontsize=10,
                     fontweight="bold", color=GROUP_COLORS[group])

    fig.text(0.5, 0.030, "Condition effect  (signed $-$log$_{10}$ p)", ha="center", fontsize=10)
    fig.text(0.002, 0.50, "Natural aging  (signed $-$log$_{10}$ p)", va="center",
             rotation=90, fontsize=10)

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=SAME_COLOR, markersize=6,
                      markeredgecolor="darkred", markeredgewidth=0.5,
                      label="Same direction as aging (acceleration)"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor=OPP_COLOR, markersize=6,
                      markeredgecolor="darkgreen", markeredgewidth=0.5,
                      label="Opposite direction (rejuvenation)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2,
               frameon=False, fontsize=9, columnspacing=1.5)

    out_dir = os.path.join(PLOTS_DIR, "assembled")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "condition_aging_consistency.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
