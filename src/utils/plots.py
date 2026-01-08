
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
from ciim.src.config import surrogate_names, palette_datasets, palette_regulation
from ciim.src.feature_association.helper import calculate_tf_activity
from ciim.src.utils.util import retrieve_adata, retrieve_net



def plot_umap(adata, color='', palette=None, ax=None, X_label='X_umap', on_data=False, sort_colors=True,
              bbox_to_anchor=None, legend=True, legend_title='', margins=dict(x=.1, y=.1), **kwrds):
    latent = adata.obsm[X_label]
    if sort_colors:
        var_unique_sorted = sorted(adata.obs[color].unique())
    else:
        var_unique_sorted = adata.obs[color].cat.categories[adata.obs[color].cat.categories.isin(adata.obs[color].unique())].values
        print(var_unique_sorted)
    legend_handles = []
    
    for i_group, group in enumerate(var_unique_sorted):
        mask = adata.obs[color] == group
        sub_data = latent[mask]
        if palette is None:
            c = None 
        else:
            c = palette[group]
        # Plot scatter points
        scatter = ax.scatter(sub_data[:, 0], sub_data[:, 1], label=group, c=c, **kwrds)
        if palette is None:
            solid_color = scatter.get_facecolor()[0]
        else:
            solid_color = c
        # plot legend
        legend_handles.append(plt.Line2D([0], [0], linestyle='none', marker='o', markersize=8, color=solid_color))
        if on_data:
            mean_x = np.mean(sub_data[:, 0])
            mean_y = np.mean(sub_data[:, 1])
            ax.text(mean_x, mean_y, group, fontsize=9, ha='center', va='top', color='black', weight='bold')
    ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.margins(**margins)

    if legend and not on_data:
        legend = ax.legend(handles=list(legend_handles), labels=list(var_unique_sorted), loc=(1.1,.3), 
                           bbox_to_anchor=bbox_to_anchor, frameon=False, title=legend_title, 
                           title_fontproperties={'weight': 'bold', 'size': 9})
        legend.get_title().set_ha('left')
        legend._legend_box.align = "left" 

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


def dotplot(df, ax, 
            ax_legend,
            color_col='trend', 
            size_col='neg_log10_adj_pval', 
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
            bbox_to_anchor_cbar=(0.8, -.2, 1, 1),
            alpha=0.5,
            linewidth=0.5,
            cbar_height='4%',
            cbar_width="60%",
            size_legend_scale=10,):
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.colors import TwoSlopeNorm
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.gridspec as gridspec

    df[size_col] = df[size_col].replace([np.inf, -np.inf], 1E-20) # replace inf with a small value
    vmin = df[color_col].min()
    vmax = df[color_col].max()
    abs_max = max(abs(vmin), abs(vmax))
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    if True:
        # Get numeric positions for x/y if categorical
        x_vals = df[x].astype('category').cat.codes
        y_vals = df[y].astype('category').cat.codes

        # Store tick labels
        x_labels = df[x].astype('category').cat.categories
        y_labels = df[y].astype('category').cat.categories

        # Map slope to color using manual colormap
        cmap = plt.get_cmap(palette if palette else 'RdBu_r')
        df['mapped_color'] = df[color_col].apply(lambda val: cmap(norm(val)))
        scaler = MinMaxScaler(feature_range=sizes)
        scaled_sizes = scaler.fit_transform(df[[size_col]]).flatten()
        y_vals = y_vals.max() - y_vals  # Reverse y-values for plotting
        y_labels = y_labels[::-1]
        # Then continue plotting as before
        ax.scatter(x_vals, y_vals, 
                c=df['mapped_color'], 
                s=scaled_sizes, 
                edgecolor='black', 
                linewidth=linewidth, 
                alpha=alpha)

        # Set y-ticks in the reversed order
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)

        # Set axis ticks and labels
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')

    # sns.scatterplot(data=df, x=x, y=y, ax=ax)

    ax.set_xlabel('')
    ax.set_ylabel(y_label)
    ax.margins(x=.1, y=.1)

    # -------  plot sigs
    pval_threshold = 1.4
    for i, row in df.iterrows():
        if row['neg_log10_adj_pval'] >= pval_threshold:
            ax.text(x_vals[i], y_vals[i]-.1, '*', ha='center', va='center', alpha=.9, 
                    fontsize=6, weight='bold', color='black', zorder=10)

    # ---------- Legends
    if show_size_legend:
        print(show_size_legend)
        size_legend_values = np.linspace(df[size_col].min() , df[size_col].max(), num=4)
        size_legend_handles = [plt.scatter([], [], s=s * size_legend_scale, color="black", label=f"{s:.1f}") for s in size_legend_values]
        size_legend_handle = plt.legend(handles=size_legend_handles, title=size_legend_title, loc='upper right', 
                                        bbox_to_anchor=size_legend_loc, frameon=False, title_fontsize=9)
    
    vmin = df[color_col].min()
    vmax = df[color_col].max()
    if vmin < 0 and vmax > 0:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = plt.Normalize(vmin=vmin, vmax=vmax) 
    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    sm.set_array([])

    # - Create the colorbar
    axins = inset_axes(
        ax_legend,
        width=cbar_width,
        height=cbar_height,
        loc='upper right',
        bbox_to_anchor=bbox_to_anchor_cbar,
        bbox_transform=ax.transAxes,
        borderpad=0
    )

    cbar = plt.colorbar(sm, cax=axins, orientation='horizontal')

    # Force ticks to show symmetric values or desired range
    tick_values = [vmin, 0, vmax ]  # or manually: [-1, -0.5, 0, 0.5, 1]
    cbar.set_ticks(tick_values)
    cbar.ax.set_xticklabels([f"{x:.2f}" for x in tick_values])

    cbar.ax.tick_params(labelsize=8, direction='out')
    cbar.ax.set_title(color_legend_title, fontsize=9, pad=5)
    


    # plt.gca().add_artist(size_legend_handle)

    