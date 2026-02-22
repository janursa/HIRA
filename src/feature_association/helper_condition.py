#!/usr/bin/env python
"""
Helper functions for condition analysis.
"""

import pandas as pd
from collections import defaultdict
from hiara.src.config import surrogate_names, palette_trend, palette_disease_effect, palette_treatment
from hiara import retrieve_feature_data


def format_tf_activity_for_disease_trend_plot(dataset, analysis_name, cell_type, tf, age_limit=[20, 75], condition_col='disease'):
    """
    Format TF activity data for disease trend plotting.
    
    Parameters
    ----------
    dataset : str
        Dataset name
    analysis_name : str
        Analysis configuration name
    cell_type : str
        Cell type to analyze
    tf : str
        Transcription factor name
    age_limit : list, optional
        Age range [min, max], default [20, 75]
    condition_col : str, optional
        Column name for condition (e.g., 'disease', 'Max_WHO_Group'), default 'disease'
    
    Returns
    -------
    pd.DataFrame or None
        Binned dataframe with disease/condition as rows, age bins as columns
    """
    tf_acts = retrieve_feature_data(dataset=dataset, cell_type=cell_type, analysis_name=analysis_name, condition=None)
    tf_acts = tf_acts[(tf_acts.obs['age'] >= age_limit[0]) & (tf_acts.obs['age'] <= age_limit[1])]
    tf_acts = tf_acts[:, tf_acts.var_names == tf]

    
    if tf_acts.shape[1] == 0:
        return None

    expr = tf_acts.to_df()
    expr = expr.merge(tf_acts.obs[['age', condition_col]], left_index=True, right_index=True, how='left')
    expr[condition_col] = expr[condition_col].astype('category')
    expr['age'] = expr['age'].astype(int)

    # Pivot table: rows = disease, columns = age, values = expression
    expr_table = expr.pivot_table(index=condition_col, columns='age', values=tf)

    # Bin ages into 5-year intervals
    df = expr_table.copy()
    age_columns = df.columns
    age_bins = defaultdict(list)

    for col in age_columns:
        bin_start = (col // 5) * 5
        age_bins[bin_start].append(col)

    # Average across bins
    binned_means = {bin_start: df[bin_ages].mean(axis=1) for bin_start, bin_ages in age_bins.items()}
    binned_df = pd.DataFrame(binned_means)

    # Sort bins by age
    binned_df = binned_df[sorted(binned_df.columns)]
    
    return binned_df


def get_condition_palette(analysis_type):
    """Get appropriate color palette based on analysis type."""
    if analysis_type == 'disease':
        palette = palette_disease_effect
    elif analysis_type == 'perturbation':
        palette = palette_treatment
    elif analysis_type == 'aging':
        palette = palette_trend
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    palette_all = {**palette_trend, **palette}
    return palette_all
