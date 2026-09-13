#!/usr/bin/env python
"""Top-15 rejuvenating perturbations of parsebioscience (CD4T), the readable subset of
clock_<dataset>_<cell_type>_rejuvenating.png.

Reads the persisted clock tables -- no clock rerun.
Output: PLOTS_DIR/clock/<dataset>/clock_<dataset>_<cell_type>_rejuvenating_subset.png
"""
import os
import sys
import warnings

import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from hira.src.config import PLOTS_DIR, CLOCK_STATS_DIR  # noqa: E402
from hira.src.clock.plots import plot_group_strip  # noqa: E402

plt.rcParams.update({"font.family": "Arial", "font.size": 8})

DATASET, CELL_TYPE, CTR = "parsebioscience", "CD4T", "PBS"
TOP_N, ALPHA = 15, 0.05


def main():
    st = pd.read_csv(f"{CLOCK_STATS_DIR}/perturbation_{DATASET}.csv")
    st = st[(st["cell_type"] == CELL_TYPE) & (st["p_value"] < ALPHA) & (st["delta"] < 0)]
    st = st.sort_values("p_value").head(TOP_N)

    obs = pd.read_csv(f"{CLOCK_STATS_DIR}/predictions_{DATASET}.csv")
    obs = obs[obs["cell_type"] == CELL_TYPE]
    pval_map = {(r.cell_type, r.ctr, r.treatment): (r.p_value, r.delta)
                for r in pd.read_csv(f"{CLOCK_STATS_DIR}/perturbation_{DATASET}.csv").itertuples()}

    # ponytail: narrower than the default width formula, and the donor legend is noise here
    _, ax = plot_group_strip(obs, [(CTR, t) for t in st["treatment"]], "Rejuvenating", CELL_TYPE,
                             pval_map, ctr=CTR, order=list(st["treatment"]), p_as_stars=True,
                             figsize=(4.2, 2.5))
    ax.get_legend().remove()
    plt.title('')
    out = os.path.join(PLOTS_DIR, 'clock', DATASET,
                       f'clock_{DATASET}_{CELL_TYPE}_rejuvenating_subset.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    plt.close('all')
    print(f"Saved: {out} ({len(st)} perturbations)")


if __name__ == "__main__":
    main()
