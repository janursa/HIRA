"""
Disease-specific analysis functions for feature association.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from ciim.src.config import surrogate_names
from ciim.src.feature_association.helper import retrieve_feature_data
from ciim.src.feature_association.plots import heatplot_age_trend


def format_tf_activity_for_disease_trend_plot(dataset, data_type, cell_type, tf, age_limit=[20, 75], condition_col='disease'):
    """
    Format TF activity data for disease trend plotting.
    
    Parameters
    ----------
    dataset : str
        Dataset name
    type : str
        Data type (e.g., 'bulk', 'sc')
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
    tf_acts = retrieve_feature_data(dataset=dataset, cell_type=cell_type, type=data_type, condition=None)
    tf_acts = tf_acts[(tf_acts.obs['age'] >= age_limit[0]) & (tf_acts.obs['age'] <= age_limit[1])]
    tf_acts = tf_acts[:, tf_acts.var_names == tf]
    # print(tf_acts)
    # aaa
    
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


def plot_healthy_disease_trend(dataset, data_type, cell_type, case_tf, condition_col, ax=None, normalize=False):
    """
    Plot healthy vs disease trend for a TF across age.
    
    Parameters
    ----------
    dataset : str
        Dataset name
    type : str
        Data type (e.g., 'bulk', 'sc')
    cell_type : str
        Cell type to analyze
    case_tf : str
        Transcription factor name
    condition_col : str
        Column name for condition (e.g., 'disease', 'Max_WHO_Group')
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, creates new if None
    normalize : bool, optional
        Whether to normalize values, default False
    
    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot
    """
    binned_df = format_tf_activity_for_disease_trend_plot(
        dataset=dataset, data_type=data_type, cell_type=cell_type, tf=case_tf, condition_col=condition_col
    )
    
    if binned_df is None:
        return None
    
    binned_df.index = [surrogate_names.get(name, name) for name in binned_df.index]
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(2.5, .5), sharey=False, sharex=True)
    
    if normalize:
        binned_df = binned_df.sub(binned_df.min(axis=1), axis=0)
        binned_df = binned_df.div(binned_df.abs().max(axis=1), axis=0)
    
    heatplot_age_trend(
        binned_df, 
        cmap='magma', 
        cbar_title="TF activity\n(normalized)" if normalize else "TF activity", 
        y_label="Condition", 
        ax=ax, 
        show_cbar=True,
        cbar_kws={
            "shrink": 1.2,
            "aspect": 3,
            "fraction": 0.1
        }
    )
    ax.set_xlabel('Age')
    
    return ax
