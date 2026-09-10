#!/usr/bin/env python
"""Assemble all clock-based age acceleration / deceleration panels into one figure.

Reads persisted clock tables (predictions_*.csv, perturbation_*.csv) -- no clock rerun.
Perturbation panels: one dot per donor = shift vs. the panel's control.
Case/control panels: two arms side by side (healthy | SLE), one dot per donor =
age acceleration, with the between-arm p on top.
Output: PLOTS_DIR/assembled/clock_age_shift_main.png
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

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10

CELL_TYPES = ["CD4T", "CD8T"]
ACC_COLOR, DEC_COLOR = "indianred", "darkseagreen"

# paired=True: same donors treated/control -> one column of per-donor differences.
# paired=False: case/control cohorts -> two columns of per-donor age acceleration.
COLUMNS = [
    dict(label="SLE\n(age < 50)", group="Disease", dataset="perez_sle",
         ctr="healthy", cond="SLE", paired=False, age_max=50, bin="Young"),
    dict(label="SLE\n(age ≥ 50)", group="Disease", dataset="perez_sle",
         ctr="healthy", cond="SLE", paired=False, age_min=50, bin="Old"),
    dict(label="SLE\n(age < 40)", group="Disease", dataset="perez_sle",
         ctr="healthy", cond="SLE", paired=False, age_max=40, bin="Young40"),
    dict(label="SLE\n(age ≥ 40)", group="Disease", dataset="perez_sle",
         ctr="healthy", cond="SLE", paired=False, age_min=40, bin="Old40"),
    dict(label="Ruxolitinib\n(ctr: DMSO)", group="JAK inhibition", dataset="op",
         ctr="DMSO", cond="Ruxolitinib", paired=True),
    dict(label="Ruxolitinib\n(ctr: RPMI)", group="JAK inhibition", dataset="CXCL9",
         ctr="RPMI", cond="RPMI + ruxolitinib", paired=True),
    dict(label="Ruxolitinib\n(ctr: LPS)", group="JAK inhibition", dataset="CXCL9",
         ctr="LPS", cond="LPS + ruxolitinib", paired=True),
    dict(label="LPS\n(ctr: RPMI)", group="Inflammation", dataset="CXCL9",
         ctr="RPMI", cond="LPS", paired=True),
    dict(label="IL-10\n(ctr: PBS)", group="Cytokine", dataset="parsebioscience",
         ctr="PBS", cond="IL-10", paired=True),
]
GROUP_COLORS = {"Disease": "#7B6C9B", "JAK inhibition": "#3D7B8C",
                "Inflammation": "#C2793F", "Cytokine": "#5E8C4A"}
# ponytail: same encoding as clock_perez_sle_bins -- |predicted - age| per donor, same colours
ARM_COLORS = {"Healthy": "#56B4E9", "SLE": "#F0E442"}


def arms(col, cell_type):
    """[(arm label, values)] -- one arm for a paired panel, two for case/control."""
    obs, (ctr, cond) = _arms(col, cell_type)
    if col["paired"]:
        pivot = obs.pivot_table(index="donor_id", columns="condition", values="predicted_age")
        if col["ctr"] not in pivot or col["cond"] not in pivot:
            return [("", np.array([]))]
        return [("", (pivot[col["cond"]] - pivot[col["ctr"]]).dropna().values)]
    return [("Healthy", ctr.values), ("SLE", cond.values)]


def _arms(col, cell_type):
    """(filtered obs, (control age acceleration, case age acceleration))."""
    obs = pd.read_csv(f"{CLOCK_STATS_DIR}/predictions_{col['dataset']}.csv")
    obs = obs[obs["cell_type"] == cell_type]
    if "age_max" in col:
        obs = obs[obs["age"] < col["age_max"]]
    if "age_min" in col:
        obs = obs[obs["age"] >= col["age_min"]]
    if col["paired"]:
        return obs, (None, None)
    res = obs.assign(age_residual=(obs["predicted_age"] - obs["age"]).abs())
    res = res.groupby(["donor_id", "condition"], as_index=False)["age_residual"].median()
    arm = lambda c: res[res["condition"] == c]["age_residual"]
    return obs, (arm(col["ctr"]), arm(col["cond"]))


def stats_of(col):
    """{cell_type: (p, delta or None)} -- always the persisted test, never recomputed here."""
    if col["paired"]:
        st = pd.read_csv(f"{CLOCK_STATS_DIR}/perturbation_{col['dataset']}.csv")
        st = st[(st["ctr"] == col["ctr"]) & (st["treatment"] == col["cond"])]
        return {r.cell_type: (r.p_value, None) for r in st.itertuples()}
    # ponytail: same table clock_perez_sle_bins annotates from
    st = pd.read_csv(f"{CLOCK_STATS_DIR}/disease_bins_{col['dataset']}.csv")
    st = st[st["age_bin"] == col["bin"]]
    return {r.cell_type: (r.p_value_adj, r.delta_residual) for r in st.itertuples()}


def draw(ax, panel, p, delta=None):
    if all(len(v) == 0 for _, v in panel):
        ax.text(.5, .5, "n/a", transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="grey")
        ax.set_xticks([])
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        return
    rng = np.random.default_rng(0)
    meds, lim = [], 0.0
    for k, (name, vals) in enumerate(panel):
        x = k + rng.uniform(-.18, .18, len(vals))
        # one arm -> colour by sign of the shift; two arms -> colour by arm
        colors = ([ACC_COLOR if v > 0 else DEC_COLOR for v in vals] if len(panel) == 1
                  else ARM_COLORS[name])
        ax.scatter(x, vals, c=colors, s=18, alpha=.8, linewidth=.2,
                   edgecolor="black" if len(panel) > 1 else "white")
        med = float(np.median(vals))
        meds.append(med)
        ax.plot([k - .3, k + .3], [med, med], color="black", lw=1.5,
                solid_capstyle="butt", zorder=3)
        lim = max(lim, np.abs(vals).max(), abs(med))

    ax.axhline(0, ls="--", color="grey", lw=.8, zorder=0)
    ax.set_ylim(-lim * .05 if len(panel) > 1 else -lim * 1.25, lim * 1.25)
    ax.set_xlim(-.55, len(panel) - 1 + .55)
    if len(panel) == 1:
        ax.set_xticks([])
    else:
        ax.set_xticks(range(len(panel)))
        ax.set_xticklabels([n for n, _ in panel], rotation=45, ha="right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    if delta is None:
        delta = meds[-1] - meds[0] if len(meds) > 1 else meds[0]
    n = " vs ".join(str(len(v)) for _, v in panel)
    stars = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
    ax.text(.5, 1.14, f"{delta:+.1f} yr (n={n})", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9,
            color=ACC_COLOR if delta > 0 else DEC_COLOR)
    ax.text(.5, 1.01, f"p = {p:.1e} {stars}", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, color="black")


def main():
    data = {c["label"]: {ct: arms(c, ct) for ct in CELL_TYPES} for c in COLUMNS}
    pvals = {c["label"]: stats_of(c) for c in COLUMNS}

    # two-arm panels get less than double the width -- 2 columns of dots are narrow
    widths = [1.6 if len(data[c["label"]][CELL_TYPES[0]]) > 1 else 1 for c in COLUMNS]
    # ponytail: gridspec has no per-gap wspace -- an empty column is the gap
    n_case = sum(1 for c in COLUMNS if not c["paired"])
    widths.insert(n_case, 0.7)
    slot = lambda j: j if j < n_case else j + 1
    nrow = len(CELL_TYPES)
    fig, axes = plt.subplots(nrow, len(widths), figsize=(.95 * sum(widths) + 1.0, 2.1 * nrow + 1.0),
                             gridspec_kw={"width_ratios": widths})
    fig.subplots_adjust(left=0.065, right=0.995, top=0.78, bottom=0.16, wspace=0.45, hspace=0.95)
    for i in range(nrow):
        axes[i, n_case].set_axis_off()

    for j, col in enumerate(COLUMNS):
        for i, ct in enumerate(CELL_TYPES):
            ax = axes[i, slot(j)]
            p, delta = pvals[col["label"]].get(ct, (np.nan, None))
            draw(ax, data[col["label"]][ct], p, delta)
            if i == 0:
                ax.set_title(col["label"], fontsize=10, pad=32,
                             color=GROUP_COLORS[col["group"]])

    # group banner bands above the column titles
    for group in dict.fromkeys(c["group"] for c in COLUMNS):
        idx = [j for j, c in enumerate(COLUMNS) if c["group"] == group]
        x0 = axes[0, slot(idx[0])].get_position().x0
        x1 = axes[0, slot(idx[-1])].get_position().x1
        fig.add_artist(Line2D([x0, x1], [.955, .955], color=GROUP_COLORS[group], lw=3,
                              transform=fig.transFigure, solid_capstyle="butt"))
        fig.text((x0 + x1) / 2, .967, group, ha="center", va="bottom", fontsize=10,
                 weight="bold", color=GROUP_COLORS[group])

    # ponytail: SLE arms plot age residuals, perturbations plot paired shifts -> two y labels
    for j, lab in [(0, "Age residual (years)"), (n_case + 1, "Age shift (years)")]:
        for i in range(nrow):
            axes[i, j].set_ylabel(lab, fontsize=9, labelpad=4)
    for i, ct in enumerate(CELL_TYPES):
        axes[i, 0].annotate(ct, xy=(-.62, .5), xycoords="axes fraction", rotation=90,
                            ha="center", va="center", fontsize=10, weight="bold")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=ACC_COLOR,
                               label="Age acceleration"),
                        Line2D([], [], marker="o", ls="", color=DEC_COLOR,
                               label="Age rejuvenation"),
                        Line2D([], [], color="black", lw=1.5, label="Median")],
               loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.005))

    out = os.path.join(PLOTS_DIR, "assembled", "clock_age_shift_main.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
