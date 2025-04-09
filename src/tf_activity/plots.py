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
from scipy.stats import spearmanr, linregress
from pandas.api.types import CategoricalDtype

from ciim.src.common import datasets_healthy, datasets_all ,surrogate_names, palette_datasets, palette_regulation, palette_twoagegroups, palette_trend, palette_datasets_pretty
from ciim.src.tf_activity.helper import adata_lambda, net_lambda, calculate_tf_activity, binarize_expression, read_tf_acts



def plot_targets_across_datasets(net, ax=None, show_legend=True):
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from pandas.api.types import CategoricalDtype
    # Define a diverging palette: Set the normalization to center at 0
    cmap = plt.cm.RdYlGn  # Red = negative, Green = positive
    norm = TwoSlopeNorm(vmin=-.1, vcenter=0, vmax=.1)

    net['dataset'] = net['dataset'].astype(CategoricalDtype(categories=datasets_healthy, ordered=True))
    net['dataset'] = net['dataset'].apply(lambda name: surrogate_names.get(name, name))
    net['slope_direction'] = net['slope'].apply(lambda x: 'Increase in aging' if x > 0 else 'Decrease in aging')

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    # Sort by target alphabetically
    net = net.sort_values(by='target')
    sns.scatterplot(
        data=net,
        x='target',
        y='dataset',
        hue='weight',
        palette=cmap,
        hue_norm=norm,
        style='slope_direction',
        markers={'Increase in aging': '^', 'Decrease in aging': 'v'},
        ax=ax,
        size='negative_log10_p_value',
        sizes=(20, 200),
        legend=False  # Suppress default legend
        )
    # Custom legend handles
    if show_legend:
        style_legend = [
            Line2D([0], [0], marker='^', color='w', label='Increase in aging', markerfacecolor='gray', markersize=8),
            Line2D([0], [0], marker='v', color='w', label='Decrease in aging', markerfacecolor='gray', markersize=8)
        ]

        weight_values = [net['weight'].min(), 0, net['weight'].max()]
        color_legend = [
            Line2D([0], [0], marker='o', color='w', label=f'Regulation: {w:.2f}',
                markerfacecolor=cmap(norm(w)), markersize=10) for w in weight_values
        ]

        size_values = np.percentile(net['negative_log10_p_value'], [25, 50, 75])
        size_legend = [
            Line2D(
                [0], [0],
                marker='o',
                color='none',  # no line
                markeredgecolor='none',  # no border
                markerfacecolor='gray',
                label=f'-log10(p): {s:.1f}',
                markersize=np.interp(s, [min(size_values), max(size_values)], [6, 14])
            )
            for s in size_values
        ]

        # Combine and place legends
        spacer = Line2D([0], [0], linestyle="none", label="")

        all_handles = style_legend + [spacer] + color_legend + [spacer] + size_legend

        ax.legend(
            handles=all_handles,
            loc='center left',
            bbox_to_anchor=(1.01, 0.5),
            borderaxespad=0,
            title='',
            frameon=False
        )

    # Tweak layout
    ax.set_ylabel('')
    plt.xticks(rotation=90)
    # plt.tight_layout()

def heatmap_tf_validation(mean_expr, ax, cmap="magma"):
    sns.heatmap(mean_expr, cmap=cmap, linewidths=0.5, cbar=True, cbar_kws={"shrink": 0.7}, ax=ax)

    # Labels and formatting
    ax.set_xlabel("Age Group")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    
    ax.set_ylabel("TFs")
    
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([])
    cbar.ax.set_ylabel('TF activity', rotation=90, labelpad=5)


def process_trends_validation(cell_type, dataset, cut_off=50):
    # - calculate mean activation across age groups
    adata_all = adata_lambda(dataset)

    adata = adata_all[adata_all.obs['cell_type'] == cell_type]
    nets = net_lambda(dataset, cell_type)
    tf_acts = calculate_tf_activity(adata, nets)
    
    ordered_age_groups = ['Under 50', 'Over 50']

    mask = tf_acts.obs['age']>cut_off
    groups = ['Over 50' if m else 'Under 50' for m in mask]
    tf_acts.obs['age_group'] = pd.Categorical(groups, categories=ordered_age_groups, ordered=True)

    # Preserve the original order of age groups
    mean_expr = tf_acts.to_df().groupby(tf_acts.obs['age_group']).mean().T
    mean_expr = mean_expr[ordered_age_groups]  # Keep original order
    # Normalize expression
    min_vals = mean_expr.min(axis=1)
    max_vals = mean_expr.max(axis=1)
    mean_expr = (mean_expr.sub(min_vals, axis=0)).div(max_vals - min_vals, axis=0)
    return mean_expr
    

def wrapper_heatmap_tf_validation(cell_type, tfs, ax, cut_off=50, dataset = 'data12', figsize=(4, 4), cmap="magma"):
    mean_expr = process_trends_validation(cell_type, dataset, cut_off=cut_off)

    tfs = [tf for tf in tfs if tf in mean_expr.index]
    mean_expr = mean_expr.loc[tfs]
    # - actual plot
    heatmap_tf_validation(mean_expr, ax, cmap=cmap)
    if i == 0:
        ax.set_ylabel("TFs")
        
        cbar = ax.collections[0].colorbar
        cbar.remove()  # This removes the colorbar
    else:
        # ax.set_yticklabels([])
        # Adjust colorbar
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([])
        cbar.ax.set_ylabel('TF activity', rotation=90, labelpad=5)

        ax.set_ylabel("")
    ax.set_title(cell_type, pad=15)
def binarize_age(obs):
    obs = obs.copy()
    obs['age_donor'] = obs['age'].astype(str) + '_' + obs['donor_id'].astype(str)
    # obs = obs[obs.age<=75]
    obs['age'] = pd.to_numeric(obs['age'], errors='coerce')
    min_age = obs.age.min()
    bins = [min_age,45, 100]  
    age_groups = ['45-', '45+']  
    obs['age_group'] = pd.cut(obs['age'], bins=bins, labels=age_groups, right=False)
    return obs
def plot_trend_tfs(cell_type, tfs, dataset='data1', ax=None):
    adata = adata_lambda(dataset)
    adata = adata[adata.obs['cell_type'] == cell_type]
    nets = net_lambda(dataset, cell_type)
    tf_acts = calculate_tf_activity(adata, nets)
    from ciim.src.process_dataset.preprocess.helper import binarize_age
    tf_acts.obs = binarize_age(tf_acts.obs)

    tf_acts_s = tf_acts[:, tf_acts.var_names.isin(tfs)]
    mean_expr = cluster_trends(tf_acts_s)
    

    heatplot_age_trend(mean_expr, cmap='magma', cbar_title="TF activity", y_label="TFs", ax=ax)
    plt.title(f'{cell_type}: TF activity trend', pad=20)

    sorted_tfs = mean_expr.index
    return sorted_tfs
def plot_trend_targets_binarized(cell_type, genes, dataset='data1', ax=None):
    adata = adata_lambda(dataset)
    adata = adata[adata.obs['cell_type'] == cell_type]
    adata = adata[:, adata.var_names.isin(genes)]
    from ciim.src.process_dataset.preprocess.helper import binarize_age
    adata.obs = binarize_age(adata.obs)
    # print(adata.obs.groupby(['age_group'])['age_donor'].nunique())

    mean_expr = cluster_trends(adata)

    heatplot_age_trend(mean_expr, cmap="viridis", cbar_title="Gene expression", y_label="Targets", ax=ax)
    plt.title(f'{cell_type}: gene expression trend', pad=20)

    sorted_tfs = mean_expr.index
    return sorted_tfs

def wrapper_heatmap_tf_d_v(cell_type, tfs, cut_off=50, figsize=(4, 4), cmap="magma"):
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True, gridspec_kw={'width_ratios': [1, 0.5, 0.5]})

    # - discovery
    ax = axes[0]
    dataset = 'data1'
    adata = adata_lambda(dataset)
    adata = adata[adata.obs['cell_type'] == cell_type]
    nets = net_lambda(dataset, cell_type)
    tf_acts = calculate_tf_activity(adata, nets)
    from ciim.src.process_dataset.preprocess.helper import binarize_age
    tf_acts.obs = binarize_age(tf_acts.obs)
    tf_acts_s = tf_acts[:, tf_acts.var_names.isin(tfs)]
    mean_expr = cluster_trends(tf_acts_s)
    heatplot_age_trend(mean_expr, cmap=cmap, cbar_title="TF activity", y_label="TFs", ax=ax, show_cbar=False)
    sorted_tfs = mean_expr.index
    ax.set_xlabel('')
    ax.set_title(f"{surrogate_names[dataset]}", pad=15)
    ax.set_title(f"Discovery", pad=10)

    # - validation
    for i, dataset in enumerate(['data12', 'data13']):
        ax = axes[i+1]
        mean_expr = process_trends_validation(cell_type, dataset, cut_off=cut_off)
        
        # Identify missing TFs
        missing_tfs = [tf for tf in sorted_tfs if tf not in mean_expr.index]
        print("These TFs are not present in the validation set:", missing_tfs)

        # Ensure all TFs are included, adding NaN where necessary
        mean_expr = mean_expr.reindex(sorted_tfs)
        # - actual plot
        heatmap_tf_validation(mean_expr, ax, cmap=cmap)
        ax.set_ylabel("")
        if i == 0:
            cbar = ax.collections[0].colorbar
            cbar.remove()  # This removes the colorbar
        else:
            cbar = ax.collections[0].colorbar
            cbar.set_ticks([])
            cbar.ax.set_ylabel('TF activity', rotation=90, labelpad=5)
        ax.set_xlabel('')
        ax.set_title(f"V {i+1}", pad=10)


def tf_gene_interaction_legend():
    # - plot the legend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    # Create a blank figure
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.axis("off")

    # 1. Color Legend
    palette_sig = {
        'Increase in aging': "green",
        'Decrease in aging': "red",
        "Non-sig.": "gray"    
    }

    color_legend_sig = [plt.scatter([], [], s=100, marker='h', color=color, label=name) for name, color in palette_sig.items()]
    color_legend_sig = ax.legend(handles=color_legend_sig, title=" Trend in aging", loc=(0.0, 0.7), frameon=False)


    color_legend_reg = [plt.scatter([], [], s=100, color=color, label=name) for name, color in palette_regulation.items()]
    color_legend_reg = ax.legend(handles=color_legend_reg, title="Regulatory sign", loc=(0.4, 0.76), frameon=False)

    # 2. Regulatory Weight Legend (Circle markers)
    size_legend_values = np.linspace(0, 1, num=6)
    size_legend_handles = [
        plt.scatter([], [], s=s * 200, color="black", label=f"{s:.1f}") for s in size_legend_values
    ]
    size_legend_handle = ax.legend(
        handles=size_legend_handles, title="  Regulatory \n importance", loc=(0.4, 0.1), frameon=False
    )

    # 3. Ageing Significance Legend (Hexagon markers)
    ageing_legend_handles = [
        plt.scatter([], [], s=s * 200, marker="h", color="black", label=f"{s:.1f}") for s in size_legend_values
    ]
    ageing_legend_handle = ax.legend(
        handles=ageing_legend_handles, title="Significance \n   in aging", loc=(0, 0.1), frameon=False
    )

    # Add legends manually
    ax.add_artist(color_legend_sig)
    ax.add_artist(color_legend_reg)
    ax.add_artist(size_legend_handle)
    return fig


def plot_joint_scatter(stats_all, col='cell_type', vars=['CD4T', 'CD8T'], annotate=True):
    # - plot
    xy_vars = [f'{v}_pval' for v in vars]
    trend_vars = [f'{v}_trend' for v in vars]

    stats_all_table = stats_all.pivot(index='tf', columns=col, values='neg_log10_adj_pval').reset_index().fillna(0)
    stats_all_trend = stats_all.pivot(index='tf', columns=col, values='trend').reset_index()
    stats_all_table = stats_all_table.merge(stats_all_trend, on='tf', suffixes=('_pval', '_trend'))
    stats_all_table[trend_vars] = stats_all_table[trend_vars].fillna('Inconsistent')

    stats_all_table['trend'] = stats_all_table[trend_vars].apply(
                            lambda x: 'Increase in aging' if (x[trend_vars[0]]=='Increase in aging' and x[trend_vars[1]]=='Increase in aging') else ('Decrease in aging' if (x[trend_vars[0]]=='Decrease in aging' and x[trend_vars[1]]=='Decrease in aging') else 'Inconsistent') , axis=1)
    stats_all_table['trend'] = stats_all_table['trend'].astype(CategoricalDtype(categories=['Increase in aging', 'Decrease in aging', 'Inconsistent'], ordered=True))


    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    scatter = sns.scatterplot(
        data=stats_all_table, 
        x=xy_vars[0], y=xy_vars[1], 
        hue='trend',  # Use trend as color
        palette= palette_trend,
        s=50,  # Adjust point size
        edgecolor='black', 
        linewidth=0.2,
        alpha=0.5,
        ax=ax
    )

    # Improve labels and colorbar
    ax.set_xlabel(f'-log10 adj p-value ({vars[0]})')
    ax.set_ylabel(f'-log10 adj p-value ({vars[1]})')
    ax.axvline(x=1.4, color='b', linestyle='--')
    ax.axhline(y=1.4, color='b', linestyle='--')

    ax.margins(x=0.1, y=0.1)
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend(loc=(1.05, 0.5), title='Trend', frameon=False)
    if annotate:
        high_tf_points = stats_all_table[(stats_all_table[xy_vars[0]] > stats_all_table[xy_vars[0]].quantile(.95))&
                                                stats_all_table[xy_vars[1]] > stats_all_table[xy_vars[1]].quantile(0)
                                            ]
        for idx, row in high_tf_points.iterrows():
            ax.text(
                row[xy_vars[0]], row[xy_vars[1]], row['tf'], 
                color='black', fontsize=8, ha='right', va='bottom'
            )
def cluster_trends(adata):
    
    # Ensure consistent formatting
    adata.obs['age_group'] = adata.obs['age_group'].str.replace('_', '-')
    unique_age_groups = adata.obs['age_group'].unique()
    age_group_order = sorted(unique_age_groups)

    # take the mean within age groups
    mean_expr = adata.to_df().groupby(adata.obs['age_group']).mean().T
    mean_expr = mean_expr[age_group_order]  # Keep original order

    # Normalize expression
    min_vals = mean_expr.min(axis=1)
    max_vals = mean_expr.max(axis=1)
    mean_expr = (mean_expr.sub(min_vals, axis=0)).div(max_vals - min_vals, axis=0)

    # KMeans clustering
    from sklearn.cluster import KMeans
    if len(mean_expr) > 2:
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(mean_expr)
        mean_expr['cluster'] = clusters
        mean_expr = mean_expr.sort_values('cluster').drop(columns=['cluster'])
    mean_expr = mean_expr[age_group_order]  # Keep original order
    return mean_expr
def plot_net_nx(net, figsize=(6, 6)):
    import networkx as nx

    G = nx.DiGraph()

    # Add nodes and edges
    for source, target, weight in net.values:
        G.add_edge(source, target, weight=weight)
    
    sources = net['source'].unique()
    targets = net['target'].unique()
    targets = np.setdiff1d(targets, sources)

    # Define node sizes
    node_size = {node: 1600 if node in sources else 200 for node in G.nodes()}

    # Define edge colors based on regulation
    edge_colors = [
        palette_regulation["Positive"] if G[u][v]["weight"] > 0 else palette_regulation["Negative"]
        for u, v in G.edges()
    ]

    # Define edge widths based on absolute weight
    # edge_weights = [abs(G[u][v]["weight"]) * 20 for u, v in G.edges()]  # Scale factor 2 for visibility

    # Define layout
    # pos = nx.spring_layout(G, k=20, iterations=100, seed=42)
    pos = nx.circular_layout(G)
    fig, ax = plt.subplots(figsize=figsize)
    # nx.draw(G, pos, with_labels=True, edgecolors='black', ax=ax)
    # Draw source nodes (circular)
    nx.draw_networkx_nodes(
        G, pos, 
        nodelist=sources, 
        node_color=['white' for n in sources], 
        node_size=[node_size[n] for n in sources],
        edgecolors="black", 
        linewidths=.5,
        alpha=0.8,
        ax=ax
    )

    # Draw target nodes (squares)
    nx.draw_networkx_nodes(
        G, pos, 
        nodelist=targets, 
        node_color=['white' for n in targets], 
        node_size=[node_size[n] for n in targets],
        node_shape="d",  # Square shape for target nodes
        edgecolors="black", 
        linewidths=0.05,
        alpha=.01,
        ax=ax
    )

    # Draw edges with variable thickness
    nx.draw_networkx_edges(
        G, pos, 
        edge_color=edge_colors, 
        arrowstyle="-|>", 
        arrowsize=20, 
        width=2,  # Use weight for thickness
        min_target_margin=25, 
        min_source_margin=25,
        alpha=1,
        ax=ax
    )

    # --- Draw labels
    font_size = 16
    nx.draw_networkx_labels(
        G, pos,
        labels={n: n for n in sources},
        font_size=font_size,
        font_weight='bold',
        ax=ax
    )

    # - Draw normal labels for target nodes
    nx.draw_networkx_labels(
        G, pos,
        labels={n: n for n in targets},
        font_size=font_size,
        font_weight='normal',
        ax=ax
    )

    plt.axis("off")

def plot_net_nx_consensus(net, figsize=(6, 6)):
    import networkx as nx

    G = nx.DiGraph()
    weight = net.groupby(['source', 'target'])['weight'].mean().reset_index()
    edges = net.groupby(['source', 'target'])['dataset'].unique().reset_index()
    edges = edges.merge(weight, on=['source', 'target'], how='left').values
    
    # Add nodes and edges
    for source, target, datasets, weight in edges:
        G.add_edge(source, target, weight=weight, datasets=datasets)
    
    sources = net['source'].unique()
    targets = net['target'].unique()
    targets = np.setdiff1d(targets, sources)

    # Define node sizes
    node_size = {node: 1600 if node in sources else 200 for node in G.nodes()}

    # Define edge colors based on regulation
    edge_colors = [
        palette_regulation["Positive"] if G[u][v]["weight"] > 0 else palette_regulation["Negative"]
        for u, v in G.edges()
    ]

    # Define edge widths based on absolute weight
    # edge_weights = [abs(G[u][v]["weight"]) * 20 for u, v in G.edges()]  # Scale factor 2 for visibility

    # Define layout
    # pos = nx.spring_layout(G, k=20, iterations=100, seed=42)
    pos = nx.circular_layout(G)
    fig, ax = plt.subplots(figsize=figsize)
    # nx.draw(G, pos, with_labels=True, edgecolors='black', ax=ax)
    # Draw source nodes (circular)
    nx.draw_networkx_nodes(
        G, pos, 
        nodelist=sources, 
        node_color=['white' for n in sources], 
        node_size=[node_size[n] for n in sources],
        edgecolors="black", 
        linewidths=.5,
        alpha=0.8,
        ax=ax
    )

    # Draw target nodes (squares)
    nx.draw_networkx_nodes(
        G, pos, 
        nodelist=targets, 
        node_color=['white' for n in targets], 
        node_size=[node_size[n] for n in targets],
        node_shape="d",  # Square shape for target nodes
        edgecolors="black", 
        linewidths=0.05,
        alpha=.01,
        ax=ax
    )

    # Draw edges with variable thickness
    nx.draw_networkx_edges(
        G, pos, 
        edge_color=edge_colors, 
        arrowstyle="-|>", 
        arrowsize=20, 
        width=2,  # Use weight for thickness
        min_target_margin=25, 
        min_source_margin=25,
        alpha=1,
        ax=ax
    )

    # --- Draw labels
    font_size = 16
    nx.draw_networkx_labels(
        G, pos,
        labels={n: n for n in sources},
        font_size=font_size,
        font_weight='bold',
        ax=ax
    )

    # - Draw normal labels for target nodes
    nx.draw_networkx_labels(
        G, pos,
        labels={n: n for n in targets},
        font_size=font_size,
        font_weight='normal',
        ax=ax
    )

    plt.axis("off")

    # ----------- Draw supporting evidence as stacked bars on edges
    for (source, target, data) in G.edges(data=True):
        # Compute midpoint of the edge
        x, y = np.mean([pos[source], pos[target]], axis=0)

        # Offset downward from edge (along y-axis)
        offset_y = -0.05  # move below the edge line

        datasets = data['datasets']
        vertical_spacing = 0.06
        box_height = 0.05
        box_width = 0.2

        for i, dataset in enumerate(datasets):
            ax.add_patch(
                plt.Rectangle(
                    (x - box_width / 2, y + offset_y - i * vertical_spacing),  # center horizontally
                    box_width,
                    box_height,
                    color=palette_datasets[dataset],
                    alpha=0.9
                )
            )
    # Show legend
    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in palette_datasets_pretty.values()]
    plt.legend(handles, palette_datasets_pretty.keys(), title="", loc=(1.05, .3), frameon=False, fontsize=14)



def heatplot_age_trend(mean_expr, cmap="viridis", cbar_title="Gene expression", y_label="Genes", figsize=(2.5, 3), ax=None, show_cbar=True, shrink=.7):
    
    # Plot heatmap
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(mean_expr, cmap=cmap, cbar=show_cbar, 
                cbar_kws={
                    "shrink": shrink,
                    "aspect": 10,       # Lower values = thicker colorbar (default is ~20)
                    "fraction": 0.1    # Controls the width space the cbar takes in the figure
                },
                 ax=ax)
    ax.set_yticks(np.arange(mean_expr.shape[0]) + 0.5)
    ax.set_yticklabels(mean_expr.index, rotation=0)

    # Adjust colorbar
    if show_cbar:
        cbar = ax.collections[0].colorbar
        # cbar.set_ticks([])
        cbar.ax.set_ylabel(cbar_title, rotation=90, labelpad=5)

    # Labels and formatting
    ax.set_xlabel("Age Group")
    ax.set_ylabel(y_label)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    

def wrapper_plot_targets(net_dict):
    for i, cell_type in enumerate(net_dict.keys()):
    # for i, cell_type in enumerate(['CD8T', 'CD4T', 'NK']):
        net = net_dict[cell_type]
        # - remove those with conflicting slopes across datasets
        slope_signs = net.pivot(index='target', columns='dataset', values='slope').apply(np.sign)
        mask = slope_signs.sum(axis=1).abs() == len(datasets_healthy)
        consistent_targets = slope_signs[mask].index
        net = net[net['target'].isin(consistent_targets)]
        # - keep those that are at least in three datasets 
        top_targets = net.groupby(['target'])['dataset'].nunique().sort_values(ascending=False)
        top_targets = top_targets[top_targets==len(datasets_healthy)].head(10).index
        
        if True: # network plot
            net = net[net['target'].isin(top_targets)]
            plot_net_nx_consensus(net, figsize=(5,4))
            # plt.title(cell_type, fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()

        if True: # target trend plot
            # - plot target gene expression trend    
            fig, axes = plt.subplots(1, 4, figsize=(20, .4*len(top_targets)), sharey=False)
            for i, (dataset) in enumerate(datasets_healthy):
                # - plot target gene expression trend    
                ax = axes[i]
                mean_expr = binarize_expression(cell_type, top_targets, dataset=dataset)
                if i == 0:
                    from sklearn.cluster import KMeans
                    if len(mean_expr) > 2:
                        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                        clusters = kmeans.fit_predict(mean_expr)
                        mean_expr['cluster'] = clusters
                        mean_expr = mean_expr.sort_values('cluster').drop(columns=['cluster'])
                    ordered_targets = mean_expr.index
                else:
                    mean_expr = mean_expr.loc[ordered_targets]
                heatplot_age_trend(mean_expr, cmap='viridis', cbar_title="Gene expression", y_label="Targets", ax=ax)
                ax.set_title(surrogate_names[dataset])
            plt.tight_layout()
            plt.suptitle(cell_type, fontsize=12, fontweight='bold', y=1.05)
            plt.show()
def plot_tfs_trend_across_datasets(cell_type, tf, datasets=['data1'], ax=None, show_cbar=True, tf_acts_dir='output/tf_activation/tf_acts/'):
    mean_expr_store = []
    for dataset in datasets:
        tf_acts = read_tf_acts(dataset,cell_type,  type='bulk', read_dir=tf_acts_dir)
        tf_acts = tf_acts[:, tf_acts.var_names==tf]

        expr = tf_acts.to_df()
        expr = expr.merge(tf_acts.obs[['age']], left_index=True, right_index=True, how='left').set_index('age')
        expr.sort_index(inplace=True)
        expr['age_bin'] = (expr.index.astype(int) // 5) * 5
        expr = expr.groupby('age_bin').mean()
        expr = expr.T

        # Normalize expression
        min_vals = expr.min(axis=1)
        max_vals = expr.max(axis=1)
        expr = (expr.sub(min_vals, axis=0)).div(max_vals - min_vals, axis=0)


        expr.index = [dataset]
        
        mean_expr_store.append(expr)
    mean_expr = pd.concat(mean_expr_store)

    mean_expr.index = mean_expr.index.map(surrogate_names)
    ages = sorted(mean_expr.columns)
    # print(mean_expr)
    heatplot_age_trend(mean_expr[ages], cmap='magma', cbar_title="TF activity", y_label="Cohorts", ax=ax, show_cbar=show_cbar, shrink=1)
    plt.title(f'{cell_type}: TF activity trend', pad=20)

    sorted_tfs = mean_expr.index
    return sorted_tfs
class DotPlotTFtarget:
    trend_palette = {'Increase in aging': 'green', 'Decrease in aging': 'red', 'Non-Sig': 'gray', 'Inconsistent': 'yellow'}
  
    def __init__(self, tf_all, skeleton):
        self.skeleton = skeleton
        self.tf_all = tf_all
 
    @staticmethod
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min())
    def plot_dotplot(self, 
                    data: pd.DataFrame, 
                    title='', 
                    figsize=(10, 20), 
                    height_ratios=(4, .2), 
                    width_ratios=[0.3, 4], 
                    ax_main_margin=[0, 0],
                    sized_left_panel=(10, 100),
                    sizes_targets=(10, 100)):
        data=data.sort_values('in_degree_c', ascending=False)
        # data['target'] = pd.Categorical(data['target'], categories=sorted_targets, ordered=True)
        
        
        
        # Compute -log10(meta_p_value) sizes for sources and targets
        data['source_size'] = -np.log10(data['meta_p_adj'])  
        data['target_size'] = -np.log10(data['meta_p_adj_target']) 
        data['regulation_sig'] = data['weight'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')

        n_tfs = len(data['source'].unique())
        n_targets = len(data['target'].unique())
        fig, axes = plt.subplots(figsize=figsize, 
                                    nrows=2, ncols=2, 
                                    gridspec_kw={'width_ratios': width_ratios, 'height_ratios': height_ratios})
        
        # print(f"n_tfs: {n_tfs} , {figsize}")
        ## --- LEFT PANEL: TF Outdegree & Linear Trend ---
        ax = axes[0, 0]
        df = data[['source', 'source_size', 'trend']].drop_duplicates()
        
        sns.scatterplot(data=df, 
                        x=np.zeros(len(df)),  
                        y='source', 
                        size='source_size',  
                        hue='trend', 
                        palette=self.trend_palette,
                        alpha=0.7, 
                        edgecolor=None,
                        marker='h',
                        sizes=sized_left_panel,
                        legend=False,
                        ax=ax)

        ax.set_xticks([])
        ax.set_xlabel("")
        ax.set_ylabel("TF", fontsize=11)
        ax.spines[['top', 'right', 'bottom']].set_visible(False)
        ax.margins(y=ax_main_margin[1])

        ## --- MAIN PLOT: TF-Target Interactions ---
        ax = axes[0, 1]
        edge_colors = data['in_skeleton'].map({True: '#E69F00', False: 'white'})
        sns.scatterplot(data=data, 
                        x='target', 
                        y='source', 
                        size=self.normalize(data['weight']),
                        hue='regulation_sig',
                        palette=palette_regulation, 
                        alpha=0.9, 
                        legend=False,
                        sizes=(10, 200),
                        edgecolor=edge_colors,  # Add edge coloring
                        linewidth=.5,  # Adjust for visibility
                        ax=ax)

        ax.set_ylabel("")
        ax.set_xlabel("")
        
        ax.set_title(title, fontsize=12, pad=20)
        ax.tick_params(axis="both", length=0)  
        ax.spines[:].set_visible(False)
        ax.set_yticklabels([], rotation=0)  
        ax.set_xticklabels([], rotation=0)  
        ax.margins(x=ax_main_margin[0], y=ax_main_margin[1])
        ax.grid(True, linestyle="--", alpha=0.5)

        ## --- BOTTOM PANEL: Target Indegree ---
        df = data[['target', 'target_size', 'trend_target']].drop_duplicates()
        df = df[df['trend_target']!= 'Inconsistent']
        # print(df['trend_target'].unique())
        ax = axes[1, 1]
        sns.scatterplot(data=df, 
                        x='target', 
                        y=np.zeros(len(df)),  
                        size='target_size',  
                        hue='trend_target', 
                        palette=self.trend_palette,
                        alpha=0.7, 
                        marker='h',
                        sizes=sizes_targets,
                        legend=False,
                        edgecolor=None,
                        ax=ax)

        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xlabel("Target gene", labelpad=20, fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90) 
        ax.margins(x=ax_main_margin[0])
        for label in ax.get_xticklabels():
            label.set_color('#CC79A7' if label.get_text() in self.tf_all else 'black')

        # Hide bottom-left panel (empty)
        axes[1, 0].axis("off")
        
        plt.tight_layout()
        plt.show()
    def plot_legend(self):
        tf_gene_interaction_legend()

    def combine_data(self, stats_source, stats_target, net):
        from ciim.src.tf_activity.helper import compute_trend
        from ciim.src.helper import determine_centrality

        stats_source = compute_trend(stats_source, pval_col='meta_p_adj', slope_col='slope', col='source')
        stats_source = stats_source[['source', 'meta_p_adj', 'trend']].drop_duplicates()
        

        stats_target = compute_trend(stats_target, pval_col='meta_p_adj', slope_col='slope', col='target')
        stats_target = stats_target.rename(columns={'meta_p_adj': 'meta_p_adj_target', 'trend': 'trend_target'})
        stats_target = stats_target[['target', 'meta_p_adj_target', 'trend_target']].drop_duplicates()

        out_degree_c = determine_centrality(net, use_weight=False).reset_index().rename(columns={'index': 'source', 'centrality': 'out_degree_c'})
        in_degree_c = net.groupby('target').size().reset_index(name='in_degree_c')

        net = net.merge(out_degree_c, on='source', how='left')
        net = net.merge(in_degree_c, on='target', how='left')

        # Merge precomputed expression significance stats
        
        data = net.merge(stats_target, on='target', how='left')
        data = data.merge(stats_source, on='source', how='right')

        return data
    def wrapper_combine_data(self, stats_source, stats_target, net, filter_criteria='meta_p_adj_target', n_top_targets=20):   
        stats_s = stats_source.rename(columns={'tf': 'source'})
        stats_t = stats_target
        data_all = self.combine_data(stats_s, stats_t, net)

        skeleton = self.skeleton

        skeleton['link'] = skeleton['source'] + '_' + skeleton['target']
        data_all['link'] = data_all['source'] + '_' + data_all['target']
        data_all['in_skeleton'] = data_all['link'].isin(skeleton['link'])

        # Subset to top TFs and targets
        if filter_criteria == 'meta_p_adj_target':
            top_targets = data_all.sort_values('meta_p_adj_target').dropna()['target'].unique()[:n_top_targets]
        elif filter_criteria == 'in_degree_c':
            # print(data[data['target'].isin(['HES4'])])
            top_targets = data_all[data_all['meta_p_adj_target']<.05]
            
            top_targets = top_targets.sort_values('in_degree_c', ascending=False).dropna()['target'].unique()[:n_top_targets]
        else:
            raise ValueError(f"Invalid filter criteria: {filter_criteria}")
        data_all = data_all[data_all['target'].isin(top_targets)]
        return data_all

# def plot_enrich(df, ax, palette, scale=50):
#     sizes=None
#     # Identify term-cell type pairs that have both Increase and Decrease
#     multi_trend = df.groupby(["Term", "cell_type"])["trend"].nunique()
#     multi_trend_pairs = multi_trend[multi_trend > 1].index

#     # Set color mapping
#     # Scatter plot for each trend (Increase, Decrease)
#     for trend, color in palette.items():
#         subset = df[df["trend"] == trend]
#         ax.scatter(
#             subset["cell_type"], subset["Term"], 
#             s=np.log10(subset["neg_log10_padj"])*scale,  # Scale size

#             color=color, edgecolors=None,
#             label=trend, alpha=0.5,
#             linewidth=0.8
#         )

#     ax.set_xlabel("Cell Type")
#     # ax.set_ylabel("Term")
#     ax.set_xlabel("")
#     ax.margins(x=.2)
#     ax.grid(True, linestyle="--", alpha=0.5)
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
#     # ax.spines[['right', 'top']].set_visible(False)
#     # ax.legend(title="Trend", loc=(1.1, .8), frameon=False)
def compare_nets_plot(datasets, cell_types, to_show='source'):
    """Compare and plot GRNs for each cell type between two datasets.

    Parameters:
    - datasets: list containing two datasets.
    - cell_types: list of cell types to analyze.

    Returns:
    - None (displays plots).
    """
    net_store = []
    for cell_type in cell_types:
        
        for dataset in datasets:
            net = net_lambda(dataset, cell_type)
            net['link'] = net['source'] + '_' + net['target']
            net['cell_type'] = cell_type
            net['dataset'] = dataset
            net_store.append(net)
    net_all = pd.concat(net_store)

    from task_grn_inference.src.exp_analysis.helper import plot_interactions, create_interaction_df

    for cell_type in cell_types:
        nets = net_all[net_all['cell_type']==cell_type]
        # edges
        df_dict = nets.groupby(['dataset'])[to_show].apply(list).to_dict()

        interaction_df = create_interaction_df(df_dict)
        aa = plot_interactions(interaction_df, min_subset_size=1)
        plt.title(cell_type)

def plot_tf_act_validation(stats_df):
    tfs = stats_df['tf'].unique()
    fig, axes = plt.subplots(1, len(tfs), figsize=(2 * len(tfs), 2.5))

    if len(tfs) == 1:
        axes = [axes]  # Ensure axes is iterable for a single TF
    i = 0
    for ax, tf in zip(axes, tfs):
        plot_tf = stats_df[stats_df['tf'] == tf]

        # Extract values
        values_young = np.concatenate(plot_tf['values_young'].values)
        spread = np.abs(values_young.max() - values_young.min())
        values_old = np.concatenate(plot_tf['values_old'].values)

        # Prepare DataFrame for plotting
        plot_df = pd.DataFrame({
            "Activity": np.concatenate([values_young, values_old]),
            "Age Group": ["Below 50"] * len(values_young) + ["Above 50"] * len(values_old)
        })

        # Strip plot
        sns.stripplot(x="Age Group", y="Activity", data=plot_df, palette=palette_twoagegroups, alpha=0.7, ax=ax)

        # Annotate p-value with bracket
        adj_p = plot_tf["adj_p_value"].values[0]
        y_max = plot_df["Activity"].max()  # Highest point in the plot

        # Coordinates for the bracket
        x1, x2 = 0, 1  # X positions of Young and Old
        y_bracket = y_max * 1.1
        y_text = y_max * 1.15

        # Draw the bracket
        ax.plot([x1, x1, x2, x2], [y_bracket, y_bracket * 1.01, y_bracket * 1.01, y_bracket], lw=1, color="black")

        # Add p-value text
        ax.text((x1 + x2) / 2, y_text, f"p={adj_p:.3e}", ha="center", fontsize=10, color="black")

        ax.set_title(tf)
        
        ax.set_ylabel("TF Activity" if ax == axes[0] else "")
        ax.margins(y=0.2, x=0.2)
        ax.set_yticks([])
        if i == 0:
            ax.set_xlabel("Age group")
        i+=1
        
    plt.tight_layout()
    plt.show()
def plot_trends(top_tfs, datasets, data_dict, palette, cell_type, surrogate_names={}, stats_df=None, is_expression=False, y_label='TF Activity score'):
    """
    Plots transcription factor (TF) activity/expression trends across datasets.

    Parameters:
    - top_tfs: list of top transcription factors to plot
    - datasets: list of dataset names
    - data_dict: dictionary containing either tf_acts or adata objects
    - palette: dictionary mapping datasets to colors
    - cell_type: cell type to include in the title
    - is_adata: if True, expects `data_dict` to contain AnnData objects instead of DataFrames
    """
    
    for tf in top_tfs:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        legend_handles = []  # Store handles for the legend

        for dataset in datasets:
            
            adata = data_dict[dataset].to_memory()
            mask_tf = adata.var_names == tf
            adata_sub = adata[:, mask_tf]
            assert adata_sub.shape[1]>0, f"TF {tf} not found in dataset {dataset}"

            ages = adata_sub.obs['age'].values
            X = adata_sub.X.todense().A if hasattr(adata_sub.X, 'todense') else adata_sub.X
            expression = X.flatten() 
            cell_count = adata_sub.obs['cell_count'].values
            
            # expression = expression / expression[0]

            cell_count_n = cell_count / max(cell_count)

            print(ages.shape, expression.shape, cell_count_n.shape)
            # Scatter plot
            ax.scatter(
                ages, expression, 
                color=palette[dataset], 
                alpha=0.4,  
                linewidth=1,
                s=cell_count_n * 100
            )

            # Fit linear regression
            if len(ages) > 1:
                age_range = np.linspace(min(ages), max(ages), 100)
                spearman_corr, spearman_p = spearmanr(ages, expression)
                slope, intercept, r_value, p_value, _ = linregress(ages, expression)
                r2 = r_value**2
                # - correct for multiple testing
                if stats_df is not None:
                    stats_df_sub = stats_df[(stats_df['tf'] == tf) & (stats_df['dataset'] == dataset)]
                    p_value_adj = stats_df_sub.loc[:, 'meta_p_adj'].values[0]
                    p_value_adj = min([p_value_adj, 1])  # Ensure p-value is not greater than 1
                else:
                    n_tests = len(top_tfs)*len(datasets)
                    p_value_adj = p_value * n_tests
                # Plot fitted line
                fitted_line = slope * age_range + intercept
                ax.plot(age_range, fitted_line, color=palette[dataset], linestyle='-', linewidth=2)

                # Create legend handle with both R² and Spearman ρ
                dataset_name = surrogate_names.get(dataset, dataset)
                legend_label = '    ' + dataset_name + f' ({slope.round(2)})'+'\n' + r' ($p_{adj}$=' + "{:.2e}".format(p_value_adj) + ")"

                handle = mpatches.Patch(color=palette[dataset], label=legend_label)
                legend_handles.append(handle)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xlabel('Age')
        ax.set_ylabel(y_label)
        ax.set_title(f'{cell_type}: {tf}', pad=20)

        # Add properly formatted legend
        ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)

def plot_activation_vs_expression(df_combined, 
                                  col_x = 'signed_-log10_pval_exp', 
                                  col_y = 'signed_-log10_pval', 
                                  y_label = "Activation\nsigned -log10(p adj)",
                                  x_label = "Expression\nsigned -log10(p adj)"):
    from src.tf_activity.helper import add_centrality
    import matplotlib.patches as mpatches
    cell_types_local = df_combined["cell_type"].unique()
    n_cell_types = len(cell_types_local)
    datasets = df_combined["dataset"].unique()
    
    
    for i, cell_type in enumerate(cell_types_local):
        
        df_cell_type = df_combined[df_combined["cell_type"] == cell_type]
        
        df_cell_type = add_centrality(df_cell_type) 
        
        fig, axes = plt.subplots(1, 2, figsize=(6, 2.7), sharey=False, sharex=False)
        for j, ax in enumerate(axes):
            df_dataset = df_cell_type[df_cell_type["dataset"] == datasets[j]]
            df_dataset['dataset'] = df_dataset['dataset'].map(surrogate_names)
            assert df_dataset.shape[0]>0, f"No data for {cell_type} in {datasets[j]}"
            # Plot scatter with correct color mapping
            sns.scatterplot(
                data=df_dataset,
                x=col_x,
                y=col_y,
                palette=palette_datasets,  # Use the consistent color mapping
                size="centrality",
                hue="dataset",
                # sizes=(20, 100),
                edgecolor=None,
                alpha=0.4,
                ax=ax,
            )

            # Set the limits to be the same for both axes
            xmin, xmax = df_dataset[col_x].min(), df_dataset[col_x].max()
            ymin, ymax = df_dataset[col_y].min(), df_dataset[col_y].max()

            global_min = min(xmin, ymin)
            global_max = max(xmax, ymax)

            placed_positions = []

            if j == 0:
                
                ax.set_ylabel(y_label)
            else:
                ax.set_ylabel("")
            ax.set_xlabel(x_label, labelpad=15)
            ax.get_legend().remove()  # Remove legend from individual plot

            
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            ax.set_aspect("equal", adjustable="datalim")

            # Define padding as a percentage of the total range
            padding = 0.15 * (global_max - global_min)

            # Apply the same limits to both axes
            # ax.set_xlim(global_min - padding, global_max + padding)
            # ax.set_ylim(global_min - padding, global_max + padding)
            # Add significance threshold lines
            sig_threshold = 1.3
            ax.margins(x=0.1, y=0.1)
            ax.axvline(sig_threshold, linestyle="--", color="red", alpha=0.6, linewidth=1)  # Vertical
            ax.axvline(-sig_threshold, linestyle="--", color="red", alpha=0.6, linewidth=1)  # Vertical
            ax.axhline(sig_threshold, linestyle="--", color="red", alpha=0.6, linewidth=1)  # Horizontal
            ax.axhline(-sig_threshold, linestyle="--", color="red", alpha=0.6, linewidth=1)  # Horizontal
        # plt.suptitle(cell_type, y=1.05)
        handles, labels = ax.get_legend_handles_labels()
        # Create color legend for trend
        color_legend = [mpatches.Patch(color=value, label=name) for name, value in palette_datasets.items()]
        color_legend_handle = plt.legend(handles=color_legend, title='Dataset', loc=(1.08, .75), frameon=False)

        # Create size legend for TF centrality
        size_legend_handle = plt.legend(handles=handles[-5:], labels=labels[-5:], title="TF centrality", 
                                        loc=(1.1, -.2),  frameon=False)

        # Add the color legend manually after the size legend
        plt.gca().add_artist(color_legend_handle)
def compare_stats_across_datasets(df, y_label='TFs'):
    """Compare TFs for each cell type across multiple datasets.

    Parameters:
    - df: A DataFrame with columns 'cell_type', 'dataset', and 'gene'.

    Returns:
    - None (displays plots).
    """
    cell_types_local = df['cell_type'].unique()
    datasets = df['dataset'].unique()

    palette_datasets_local = {**palette_datasets_pretty, 'Common': 'green'}
    fig, axes = plt.subplots(1, len(cell_types_local), figsize=(5 * len(cell_types_local), 3), sharey=True)
    if len(cell_types_local) == 1:
        axes = [axes]  # ensure iterable

    for ii, cell_type in enumerate(cell_types_local):
        df_cell_type = df[df['cell_type'] == cell_type]

        # TFs significant in meta analysis
        common_tfs = set(df_cell_type[df_cell_type['meta_p_adj'] < 0.05]['gene'])

        counts = {'Common': len(common_tfs)}

        for dataset in datasets:
            tfs_dataset = set(df_cell_type[(df_cell_type['dataset'] == dataset) & (df_cell_type['p_value_adj'] < 0.05)]['gene'])
            unique_tfs = tfs_dataset
            counts[surrogate_names.get(dataset, dataset)] = len(unique_tfs)

        # Prepare data for plotting
        data = pd.DataFrame({
            'Category': list(counts.keys()),
            'Count': list(counts.values())
        })

        ax = axes[ii]
        sns.barplot(data=data, x='Category', y='Count', palette=palette_datasets_local, ax=ax)
        ax.set_title(f'{cell_type}')
        ax.set_xlabel('')
        ax.set_ylabel(y_label)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.spines[['top', 'right']].set_visible(False)
        ax.margins(x=0.1, y=0.1)

    plt.tight_layout()
    # plt.suptitle('Sig. number', fontsize=12, y=1.05)