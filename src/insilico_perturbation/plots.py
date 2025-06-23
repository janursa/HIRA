import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

from ciim.src.common import save_dir, surrogate_names, palette_datasets_pretty
from ciim.src.tf_activity.helper import retrieve_sig_stats
from ciim.src.tf_activity.helper import get_consensus_net
from ciim.src.common import datasets_all
def plot_age_acceleration_donors(df_pivot, ax=None, ctr='baseline', treatment='perturb', id_vars='donor_age'):
    if False: # line plot
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.scatterplot(data=df_pivot, x=ctr, y=treatment, alpha=0.7, ax=ax)
        min_age, max_age = df_pivot[ctr].min(), df_pivot[treatment].max()
        ax.plot([min_age, max_age], [min_age, max_age], color='gray', linestyle='--', label='Ideal')

    if True: # donor plot
        df_plot = df_pivot.reset_index().melt(id_vars=id_vars, 
                                            value_vars=[ctr, treatment],
                                            var_name='condition', 
                                            value_name='predicted_age')

        # plt.figure(figsize=(3, 2.5))
        if ax is None:
            fig, ax = plt.subplots(figsize=(2.5, 2))
        sns.lineplot(data=df_plot, x='condition', y='predicted_age', 
                    hue=id_vars, marker='o', alpha=0.6, legend=False, ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel('Predicted age (years)')
        ax.set_xticklabels(ax.get_xticklabels(),rotation=45, ha='right')
        ax.margins(x=.2, y=0.1)