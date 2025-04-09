
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import colors
from matplotlib import cm
from matplotlib import rcParams
import matplotlib.patches as mpatches
import scipy
import networkx as nx
from scipy.stats import spearmanr, linregress
from ciim.src.common import surrogate_names, palette_datasets, palette_regulation
from ciim.src.tf_activity.helper import adata_lambda, net_lambda, calculate_tf_activity


def heatplot_centrality(df, cmap="viridis", cbar_title="Gene expression", y_label="Genes", figsize=(2.5, 3), quantile=.9):
    # Handle color normalization
    ordered_source = df['source'].unique()
    pivot_df = df.pivot(index="source", columns="cell_type", values="centrality")

    pivot_df = pivot_df.loc[ordered_source]

    normed_data = pivot_df.clip(upper=pivot_df.quantile(quantile), axis=1)
    

    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(normed_data, cmap=cmap, linewidths=0.5, cbar=True, cbar_kws={"shrink": 0.7}, ax=ax)

    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel(cbar_title, rotation=90, labelpad=5)

    ax.set_xlabel("Cell Type")
    ax.set_ylabel(y_label)
    plt.xticks(rotation=45)


def dotplot(df, ax, color_col='trend', size_col='neg_log10_adj_pval', 
            x='cell_type', 
            y='tf', 
            palette=None, sizes=(20, 200),
            show_color_legend=True, 
            show_size_legend=True,
            y_label="Transcription Factor",
            size_legend_title='-log10(p-value)',
            color_legend_title='Trend',
            size_legend_loc=(1.1, 0.1),
            color_legend_loc=(1.1, 0.7),
            alpha=0.5):
    
    
    scatter = sns.scatterplot(
        data=df, 
        x=x, 
        y=y, 
        size=size_col, 
        hue=color_col, 
        sizes=sizes,
        palette=palette, 
        edgecolor="black",
        legend=False,  
        alpha=alpha,
        ax=ax
    )
    # Labels and formatting
    ax.set_ylabel(y_label)
    # ax.set_xlabel("Cell type")
    # ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    # ax.xaxis.set_label_position('top')  # Move the x-axis label to the top
    # ax.xaxis.tick_top() 

    ax.set_xlabel('')
    # ax.spines[['right']].set_visible(False)
    ax.margins(x=.1, y=.1)
    # Create Legends
    if show_size_legend:
        size_legend_values = np.linspace(df[size_col].min() , df[size_col].max(), num=6)
        size_legend_handles = [plt.scatter([], [], s=s * 10, color="black", label=f"{s:.1f}") for s in size_legend_values]
        size_legend_handle = plt.legend(handles=size_legend_handles, title=size_legend_title, loc=size_legend_loc, frameon=False)
    
    if show_color_legend:
        color_legend = [mpatches.Patch(color=color, label=name,alpha=alpha) for name, color in palette.items()]
        color_legend_handle = plt.legend(handles=color_legend, title=color_legend_title, loc=color_legend_loc, frameon=False)
    
    if show_color_legend:
        plt.gca().add_artist(size_legend_handle)