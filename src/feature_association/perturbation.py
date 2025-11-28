"""
Perturbation analysis functions for drug and treatment studies.
"""

import os
from collections import OrderedDict
from ciim.src.common import SAVE_DIR
from ciim.src.feature_association.helper import wrapper_association_with_age_condition


# Dataset configurations for perturbation studies
DATASET_CONFIG = {
    "op": {
        "test_type": "mixed-effect",
        "name_mapping": OrderedDict({
            "Ruxolitinib": "Ruxolitinib",
            "LY2090314": "LY2090314",
        }),
        "comparisons": ["Ruxolitinib"],
        "show_sig_tfs": True,
    },
    "CXCL9": {
        "test_type": "mixed-effect",
        "name_mapping": OrderedDict({
            # "24 h LPS": "LPS (ctr: RPMI)",
            "24 h RPMI + ruxolitinib": "Ruxolitinib (ctr: RPMI)",
            "24 h LPS + ruxolitinib": "Ruxolitinib (ctr: LPS)",
        }),
        "comparisons": ["Ruxolitinib (ctr: RPMI)", "Ruxolitinib (ctr: LPS)"],
        "show_sig_tfs": False,
    },
    "parsebioscience": {
        "test_type": "mixed-effect",
        "name_mapping": OrderedDict({}),
        "comparisons": ["IL-10"],  # extend as needed
        "show_sig_tfs": True,
    },
}


def run_stats(features, par, cfg, SAVE_DIR, dataset, data_type, feature_type):
    """
    Run statistical analysis for perturbation experiments.
    
    Parameters
    ----------
    features : list or None
        List of features to analyze, None for all
    par : dict
        Parameters dictionary containing:
        - feature_type: 'tf_activity' or 'gene_expression'
        - datasets: list of dataset names
        - cell_types: list of cell types
        - type: 'bulk' or 'sc'
        - cell_data_typeresolution: resolution level
        - association_type: e.g., 'spearman'
    cfg : dict
        Configuration dictionary from DATASET_CONFIG
    SAVE_DIR : str
        Directory to save results
    dataset : str
        Dataset name
    data_type : str
        Data type (e.g., 'bulk', 'sc')
    feature_type : str
        Feature type ('tf_activity' or 'gene_expression')
    
    Returns
    -------
    pd.DataFrame
        Statistics dataframe with comparison column added
    """
    os.makedirs(f"{SAVE_DIR}/stats", exist_ok=True)

    stats_drugs = wrapper_association_with_age_condition(
        par, features=features, test_type=cfg["test_type"]
    )
    stats_drugs["comparision"] = stats_drugs["condition"].replace(cfg["name_mapping"])
    stats_drugs = stats_drugs.dropna(subset=["comparision"])
    stats_drugs["comparision"] = stats_drugs["comparision"].astype("category")

    out_csv = f"{SAVE_DIR}/stats/stats_{dataset}_{data_type}_{feature_type}_{cfg['test_type']}.csv"
    stats_drugs.to_csv(out_csv)
    
    return stats_drugs


def get_dataset_config(dataset):
    """
    Get configuration for a specific dataset.
    
    Parameters
    ----------
    dataset : str
        Dataset name
    
    Returns
    -------
    dict
        Configuration dictionary
    
    Raises
    ------
    ValueError
        If dataset is not in DATASET_CONFIG
    """
    if dataset not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset {dataset}. Available: {list(DATASET_CONFIG.keys())}")
    return DATASET_CONFIG[dataset]
