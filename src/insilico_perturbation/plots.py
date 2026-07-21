import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

from hira.src.config import OUTPUT_DIR, surrogate_names, palette_datasets_pretty

    
def wrapper_plot_age_acceleration_for_tf_perturbation(
                    age_shift_mean_t, 
                    top_n=30, 
                    features=None, 
                    value_col='signed_neg_log10_pval',
                    figsize = (3, 5)):
    from hira.src.config import OUTPUT_DIR, surrogate_names, palette_datasets_pretty, colors_blind, palette_trend_2
    

    # Sort TFs by signed mean_diff for plotting


    fig, ax = plt.subplots(figsize=figsize)
    age_shift_mean_t['dataset'] = age_shift_mean_t['dataset'].apply(lambda x: surrogate_names.get(x, x))
    # Background bars
    sns.barplot(
        data=age_shift_mean_t,
        x='tf',
        y=value_col,
        hue='trend',
        palette=palette_trend_2,
        edgecolor='black',
        linewidth=0.01,  
        alpha=.5,
        ci=None,
        ax=ax
    )

    ax.get_legend().remove()  # Remove legend for the background bars
    if False:
        # Dataset-colored points
        sns.stripplot(
            data=age_shift_mean_t,
            x='tf',
            y=value_col,
            hue='dataset',
            palette=palette_datasets_pretty,
            dodge=True,
            alpha=0.8,
            size=5,
            jitter=False,
            # orient='h',
            ax=ax
        )
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D
        
        # Create circular legend handles with larger size
        legend_handles = [
            Line2D([0], [0], marker='o', color='w',
                markerfacecolor=palette_datasets_pretty[ds],
                markersize=12, label=ds, linestyle='None')
            for ds in age_shift_mean_t['dataset'].unique()
        ]

        ax.legend(handles=legend_handles, title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    # ax.axvline(0, color='gray', linestyle='--')
    ax.set_title(f'Top {top_n} TFs')
    ax.set_ylabel('Significance of age shift \n (z-score)')
    ax.set_xlabel('TF')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')
    ax.margins(y=.1, x=.05)
    
    return fig

def plot_age_acceleration(df_all, 
                            x_col='cell_type', 
                            log_y=False, 
                            margins=(0.1, 0.2),
                            figsize=(3, 3),
                            color='blue',
                            ax=None):
    # If a specific order is given, enforce it
    order = df_all[x_col].unique()
    print(df_all['cell_type'].unique())
    # Calculate median per category
    df_median = df_all.groupby(x_col)['age_shift'].median().reset_index()
    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    df_all['dataset'] = df_all['dataset'].apply(lambda name: surrogate_names.get(name, name))
    sns.stripplot(ax=ax, data=df_all, y='age_shift', x=x_col, hue='dataset',
                  palette=palette_datasets_pretty, alpha=0.7, order=order)
    sns.barplot(ax=ax, data=df_median, y='age_shift', x=x_col, alpha=0.5, color=color, order=order)
    
    # Axes and labels
    ax.legend(loc=(1.05, .2), frameon=False, title='Dataset')
    ax.margins(x=margins[0], y=margins[1])
    ax.set_ylabel("Age shift \n (pseudo-years)")
    ax.set_xlabel("")
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    if log_y:
        ax.set_yscale('symlog')
    # Create circular legend handles with larger size
    df = df_all
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='o', color='w',
            markerfacecolor=palette_datasets_pretty[ds],
            markersize=10, label=ds, linestyle='None')
        for ds in df['dataset'].cat.categories
    ]

    ax.legend(handles=legend_handles, title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

def plot_age_acceleration_donors(df_pivot, 
                                ax=None,
                                ctr='baseline', 
                                treatment='perturb', 
                                id_vars='donor_age',
                                figsize=(2.5, 2)):
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
            fig, ax = plt.subplots(figsize=figsize)
        sns.lineplot(data=df_plot, x='condition', y='predicted_age', 
                    hue=id_vars, marker='o', alpha=0.6, legend=False, ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel('Predicted age (years)')
        ax.set_xticklabels(ax.get_xticklabels(),rotation=45, ha='right')
        ax.margins(x=.2, y=0.1)