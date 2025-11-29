"""
Perturbation analysis module for drug and treatment studies.

This module contains:
- config.py: Dataset configuration and helper functions
- compute_all_drug_stats.py: Compute drug stats for all TFs
- drug_reversal.py: Core reversal analysis functions
- drug_reversal_plots.py: Plotting functions for reversal analysis
- run_drug_reversal.py: Main script to run reversal analysis
"""

from .config import DATASET_CONFIG, get_dataset_config, run_stats
from .compute_all_drug_stats import compute_all_drug_stats
from .drug_reversal import (
    compute_tf_overlap,
    fisher_exact_test_reversal,
    calculate_reversal_score,
    classify_drug_effect,
    analyze_drug_reversal,
    filter_rejuvenating_drugs,
    save_reversal_results,
    print_reversal_summary
)
from .drug_reversal_plots import (
    plot_reversal_heatmap,
    plot_drug_ranking,
    plot_contingency_tables,
    plot_reversal_overview
)
from .run_drug_reversal import run_reversal_analysis

__all__ = [
    'DATASET_CONFIG',
    'get_dataset_config',
    'run_stats',
    'compute_all_drug_stats',
    'compute_tf_overlap',
    'fisher_exact_test_reversal',
    'calculate_reversal_score',
    'classify_drug_effect',
    'analyze_drug_reversal',
    'filter_rejuvenating_drugs',
    'save_reversal_results',
    'print_reversal_summary',
    'plot_reversal_heatmap',
    'plot_drug_ranking',
    'plot_contingency_tables',
    'plot_reversal_overview',
    'run_reversal_analysis'
]
