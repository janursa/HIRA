#!/usr/bin/env python
"""
Assemble overview heatmap of significant TFs across three conditions:
  Panel 1 (left):   SLE           (perez_sle,       disease)
  Panel 2 (center): IL-10         (parsebioscience,  perturbation)
  Panel 3 (right):  Ruxolitinib   (op,               perturbation)

Reuses the exact code path from post_condition_analysis.main() +
wrapper_plots_tfa_major_b_condition to produce each overview heatmap,
then assembles the three saved PNGs into a single figure.

Output: PLOTS_DIR/assembled/overview_sig_tfs.png
"""

import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from argparse import Namespace
from PIL import Image

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from hira import retrieve_stats
from hira.src.config import PLOTS_DIR, MAJOR_CTS, get_config_fa
from hira.src.feature_association.plots_condition import plot_overview_heatmap

plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "Arial"

ASSEMBLED_DIR = os.path.join(PLOTS_DIR, "assembled")
os.makedirs(ASSEMBLED_DIR, exist_ok=True)

ANALYSIS_NAME = "tfa_major_b"


def make_args(dataset, analysis_type):
    """Replicate the args namespace that post_condition_analysis.main() builds."""
    analysis_config = get_config_fa(ANALYSIS_NAME)
    args = Namespace(
        dataset=dataset,
        analysis_type=analysis_type,
        analysis_name=ANALYSIS_NAME,
        output_dir=PLOTS_DIR,
        cell_types=MAJOR_CTS,
        granularity=analysis_config['granularity'],
        feature_type=analysis_config['feature_type'],
        data_type=analysis_config['data_type'],
        skip_overview=False,
        skip_pathway=True,
        skip_case_studies=True,
        skip_heatmap=True,
        top_aging_tfs=15,
        agreement='same',
        sig_threshold=0.05,
        association_type='grouped',
        case_tfs=['KLF6', 'PRDM1', 'LEF1', 'ZEB2'],
    )
    return args


def generate_overview(dataset, analysis_type, stats_for_overview):
    """
    Call plot_overview_heatmap exactly as the wrapper does, saving to
    PLOTS_DIR/overview_{dataset}.png, then return the saved path.
    """
    args = make_args(dataset, analysis_type)
    plot_overview_heatmap(stats_for_overview, args)
    return os.path.join(PLOTS_DIR, f"overview_{dataset}.png")


def get_stats_for_overview(dataset):
    """
    Load and filter stats exactly as wrapper_plots_tfa_major_b_condition does
    for each dataset before passing to plot_overview_heatmap.
    """
    stats = retrieve_stats(dataset=dataset, analysis_name=ANALYSIS_NAME, multi_cohort=False)
    stats_sig = stats[stats['is_significant']]

    if dataset == 'perez_sle':
        age_group = 'Both age groups'
        stats_overview = stats_sig[stats_sig['age_group'] == age_group]
    elif dataset in ('parsebioscience', 'op'):
        stats_overview = stats_sig
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return stats_overview


# ---------------------------------------------------------------------------
# Panel definitions: (dataset, analysis_type, panel_title)
# ---------------------------------------------------------------------------
PANELS = [
    ("perez_sle",       "disease",      "SLE"),
    ("parsebioscience", "perturbation", "IL-10"),
    ("op",              "perturbation", "Ruxolitinib"),
]


def assemble():
    saved_paths = []
    for dataset, analysis_type, title in PANELS:
        print(f"  Generating overview: {title} ({dataset}) ...")
        stats_overview = get_stats_for_overview(dataset)
        path = generate_overview(dataset, analysis_type, stats_overview)
        saved_paths.append((path, title))

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), gridspec_kw={"wspace": 0.05})

    for ax, (path, title) in zip(axes, saved_paths):
        img = np.array(Image.open(path))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=2)

    plt.tight_layout()

    output_path = os.path.join(ASSEMBLED_DIR, "overview_sig_tfs.png")
    fig.savefig(output_path, bbox_inches="tight", dpi=300, transparent=False)
    plt.close(fig)
    print(f"\nSaved assembled figure: {output_path}")


if __name__ == "__main__":
    print("Assembling overview significant TFs figure ...")
    assemble()
