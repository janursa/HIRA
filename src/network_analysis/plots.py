import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, linregress
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.cluster.hierarchy import linkage
from matplotlib.patches import Patch
import networkx as nx

from hiara.src.config import GRNS_DIR, PRIOR_DIR, MAJOR_CTS, OUTPUT_DIR, PLOTS_DIR, get_config, colors_blind, DISCOVERY_COHORTS, \
    surrogate_names, palette_datasets, mapping_minor_2_major, \
    palette_trend_2, palette_major_cts, palette_datasets, palette_trend_2, colors_blind, \
        get_config_fa, cmap_trend
from hiara.src.feature_association.helper import calculate_tf_activity, bin_feature_values, retrieve_feature_data, \
                                                    retrieve_sig_stats, retrieve_stats
from hiara.src.utils.util import retrieve_net, retrieve_adata, retrieve_net_consensus

class ModularizedNetPlot:
    @staticmethod
    def prepare_net_only_tfs(cell_type, race, data_type, min_degree=3):
        from hiara.src.feature_association.helper import retrieve_nets, retrieve_sig_stats

        stats_sig = retrieve_sig_stats(race=race, data_type=data_type).drop_duplicates(subset=['cell_type', 'gene'])
        stats_sig_t = stats_sig[stats_sig['cell_type'] == cell_type].set_index(['gene'])
        sig_tfs = stats_sig_t.index.unique()
        datasets = DISCOVERY_COHORTS
        cell_type_major = mapping_minor_2_major.get(cell_type, cell_type)
        net = retrieve_nets(datasets, cell_type_major, promotor_only=True)

        # - subset to only sig tfs
        net = net[(net['source'].isin(sig_tfs) & net['target'].isin(sig_tfs))]
        # - keep edges with min degree
        net['sign'] = np.sign(net['weight'])
        net = net.groupby(['source', 'target']).filter(lambda x: len(x) >= min_degree)
        degress = net.groupby(['source', 'target', 'sign']).size()
        tuple_index = degress[degress >= min_degree].index
        net = net.set_index(['source', 'target', 'sign']).loc[tuple_index].reset_index().drop_duplicates(subset=['source', 'target', 'sign'])[['source', 'target', 'sign', 'weight']]
        return net
    @staticmethod
    def collapse_nets(net):
        """
        Collapse the network by clustering the source and target nodes based on their connectivity patterns.
        """

        # Pivot to create source-target matrix
        if 'sign' not in net.columns:
            net['sign'] = np.sign(net['weight'])
            
        adj_matrix = net.pivot_table(index='source', columns='target', values='sign', fill_value=0)

        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist

        # Compute clustering on sources (rows)
        source_dist = pdist(adj_matrix, metric='cosine')
        source_linkage = linkage(source_dist, method='average')
        source_clusters = fcluster(source_linkage, t=0.5, criterion='distance')  # tune `t` to get different granularity

        # Same for targets (columns)
        target_dist = pdist(adj_matrix.T, metric='cosine')
        target_linkage = linkage(target_dist, method='average')
        target_clusters = fcluster(target_linkage, t=0.5, criterion='distance')

        source_module_map = dict(zip(adj_matrix.index, source_clusters))
        target_module_map = dict(zip(adj_matrix.columns, target_clusters))

        net['source_module'] = net['source'].map(source_module_map)
        net['target_module'] = net['target'].map(target_module_map)

        # Updated source module names
        from collections import defaultdict

        # Get source groups
        source_groups = defaultdict(list)
        for gene, mod in source_module_map.items():
            source_groups[mod].append(gene)

        # Same for targets
        target_groups = defaultdict(list)
        for gene, mod in target_module_map.items():
            target_groups[mod].append(gene)


        source_module_names = {
            mod: '/'.join(sorted(genes))
            for mod, genes in source_groups.items()
        }

        # Updated target module names
        target_module_names = {
            mod: '/'.join(sorted(genes))
            for mod, genes in target_groups.items()
        }
        net['source_module_name'] = net['source_module'].map(source_module_names)
        net['target_module_name'] = net['target_module'].map(target_module_names)

        # Collapse
        collapsed_net = net.groupby(['source_module_name', 'target_module_name'])['sign'].mean().reset_index()
        collapsed_net.rename(columns={'source_module_name': 'source', 'target_module_name': 'target', 'sign': 'weight'}, inplace=True)
        return collapsed_net
    @staticmethod
    def add_trend_to_collapsed_net_only_tfs(collapsed_net, data_type, race, cell_type):
        from hiara.src.feature_association.helper import retrieve_sig_stats
        # - sumarize the trends for the collapsed net
        stats_sig = retrieve_sig_stats(race=race, data_type=data_type).drop_duplicates(subset=['cell_type', 'gene'])
        stats_sig_t = stats_sig[stats_sig['cell_type'] == cell_type].set_index(['gene'])
        def summarize_trend(x):
            'Assigns one trend for multiple tfs'
            unique_trends = np.unique([stats_sig_t.loc[tf]['trend'] for tf in x.split('/')])
            if len(unique_trends)==1:
                return unique_trends[0]
            else:
                return 'Mixed'

        source_trends = []
        target_trends = []

        for i, row in collapsed_net.iterrows():
            source = row['source']
            target = row['target']

            source_trend = summarize_trend(source)
            target_trend = summarize_trend(target)

            source_trends.append(source_trend)
            target_trends.append(target_trend)
        collapsed_net['trend_source'] = source_trends
        collapsed_net['trend_target'] = target_trends

        return collapsed_net
    @staticmethod
    def add_trend_to_collapsed_net(collapsed_net, net):
        # - sumarize the trends for the collapsed net
        
        def summarize_trend(x, col='source'):
            'Assigns one trend for multiple tfs'
            df = net.drop_duplicates(subset=[col, f'trend_{col}']).set_index(col).copy()
            # print(df)
    
            unique_trends = np.unique([df.loc[gene][f'trend_{col}'] for gene in x.split('/')])
            if len(unique_trends)==1:
                return unique_trends[0]
            else:
                return 'Mixed'

        source_trends = []
        target_trends = []

        for i, row in collapsed_net.iterrows():
            source = row['source']
            target = row['target']

            source_trend = summarize_trend(source, col='source')
            target_trend = summarize_trend(target, col='target')

            source_trends.append(source_trend)
            target_trends.append(target_trend)
        collapsed_net['trend_source'] = source_trends
        collapsed_net['trend_target'] = target_trends

        return collapsed_net
def dotplot_category_color(df, ax, 
            color_col='trend', 
            size_col='neg_log10_adj_pval', 
            x='cell_type', 
            y='gene', 
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
            size_legend_scale=10,):
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.colors import TwoSlopeNorm

    
    sns.scatterplot(data=df, x=x, y=y, size=size_col, hue=color_col, palette=palette, ax=ax, legend=False, sizes=sizes, alpha=alpha)
    # ax.grid(True, linestyle="--", alpha=0.5)
    # ax.spines[['right']].set_visible(False)
    ax.margins(x=.3, y=.05)
    ax.set_xticks(range(len(df[x].cat.categories)))
    ax.set_xticklabels(df[x].cat.categories, rotation=45, ha='right')
    ax.set_ylabel('')
    ax.set_xlabel('')

    # Create Legends
    if show_size_legend:
        size_legend_values = np.linspace(df[size_col].min(), df[size_col].max(), num=4)
        size_legend_values = np.round(size_legend_values).astype(int)  # force integers
        
        size_legend_handles = [
            plt.scatter([], [], s=s * size_legend_scale, color="black", label=f"{s}")
            for s in size_legend_values
        ]
        size_legend_handle = plt.legend(
            handles=size_legend_handles,
            title=size_legend_title,
            loc=size_legend_loc,
            frameon=False
        )
    if show_color_legend:
        if isinstance(palette, dict):
            # Case 1: dictionary palette
            color_legend = [
                Line2D([0], [0], marker='o', color='none', markerfacecolor=color,
                    markersize=10, label=name, alpha=alpha) 
                for name, color in palette.items()
            ]
            color_legend_handle = plt.legend(
                handles=color_legend, 
                title=color_legend_title, 
                loc=color_legend_loc, 
                frameon=False
            )
        elif palette == "viridis":
            import matplotlib as mpl
            cmap = plt.cm.viridis
            values = df[color_col]
            # values = np.clip(values, a_min=0, a_max=5)
            values = values[np.isfinite(values)]
            vmin = values.min()
            vmax = values.max()
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cax = ax.inset_axes([*color_legend_loc, 0.2, 0.2])  # width and height adjustable

            cbar = plt.colorbar(sm, cax=cax)
            cbar.set_label(color_legend_title, fontsize=10)
            cbar.ax.tick_params(labelsize=9)

        else:
            raise ValueError(f"Unsupported palette type: {palette}")

        ax.spines[['top', 'right']].set_visible(False)
        ax.add_artist(size_legend_handle)
def draw_net_datasets(cell_type, datasets, features, min_degree=3, indivitual_net=True, 
                     draw_evidence=True, draw_collectri=True, 
                     figsize=(4, 4), figsize_collectri=(3,3), 
                     offset_evidence=.11, arc_offset=.05, 
                     offset_evidence_collectri=.1, 
                     promotor_only=False,
                     prior_dir=PRIOR_DIR,
                     grns_dir=GRNS_DIR):
    net_store = []
    for dataset in datasets:
        cell_type_major = mapping_minor_2_major.get(cell_type, cell_type)
        net = retrieve_net(dataset=dataset, cell_type=cell_type_major, promotor_only=promotor_only, grns_dir=grns_dir, prior_dir=prior_dir)
        net['dataset'] = dataset
        net_store.append(net)
    net = pd.concat(net_store, ignore_index=True)
    net = net[(net['source'].isin(features) & net['target'].isin(features))]
    
    # - keep edges with min degree
    net = net.groupby(['source', 'target']).filter(lambda x: len(x) >= min_degree)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_net_nx(net, ax=ax, figsize=figsize, draw_evidence=draw_evidence, palette_evidence=palette_datasets, offset_evidence=offset_evidence, arc_offset=arc_offset)
    plt.title(f"{cell_type_major}", fontsize=14, pad=20, weight='bold')

    if indivitual_net:
        for dataset in datasets+['collectri']:
            net_i = net[net['dataset'] == dataset]
            plot_net_nx(net_i, figsize=figsize, draw_evidence=False, offset_evidence=offset_evidence, arc_offset=arc_offset)
            plt.title(f"{surrogate_names.get(dataset, dataset)}", fontsize=16, pad=20)
    if draw_collectri:
        # - add collectri
        collectri = pd.read_csv(f'{prior_dir}/collectri_with_source.csv')
        collectri['dataset'] = collectri['ref']
        evidence = collectri.copy()
        if False:
            # - add skeleton
            # skeleton = pd.read_csv(f'/home/jnourisa/projs/ongoing/task_grn_inference/resources/grn_benchmark/prior//skeleton.csv')
            skeleton = pd.read_csv(f'{prior_dir}/skeleton_promotor.csv')
            skeleton['weight'] = 1
            skeleton['dataset'] = 'skeleton'
            evidence = pd.concat([evidence, skeleton], ignore_index=True)

        # - evidence
        evidence = evidence[evidence['source'].isin(features) & evidence['target'].isin(features)]
        evidence = evidence[evidence['source'] != evidence['target']]

        refs = evidence['dataset'].unique()
        set2_colors = sns.color_palette("Set2", n_colors=len(refs))
        palette_evidence ={d: color for d, color in zip(refs, set2_colors)}
        plot_net_nx(evidence, figsize=figsize_collectri, draw_evidence=True, palette_evidence=palette_evidence, offset_evidence=offset_evidence_collectri, arc_offset=arc_offset)
        plt.title(f"{cell_type_major} - CollecTRI", fontsize=14, pad=20, weight='bold')
    return fig

def draw_net_datasets_targets(cell_type, datasets, tfs, n_targets=10, promotor_only=False, min_consensus=3,
                             prior_dir=PRIOR_DIR,
                             grns_dir=GRNS_DIR):
    """
    Draw network showing TF target programs - each TF with its top targets based on regulatory weights.
    
    Parameters:
    - cell_type: Cell type to analyze
    - datasets: List of datasets to use
    - tfs: List of transcription factors
    - n_targets: Maximum number of targets to show per TF
    - promotor_only: Whether to use only promoter-based regulations
    """
    # Load networks from all datasets
    net_store = []
    for dataset in datasets:
        cell_type_major = mapping_minor_2_major.get(cell_type, cell_type)
        net = retrieve_net(dataset=dataset, cell_type=cell_type_major, promotor_only=promotor_only, grns_dir=grns_dir, prior_dir=prior_dir)
        # z score for weight
        net['weight'] = (net['weight'] - net['weight'].mean()) / net['weight'].std()
        net['dataset'] = dataset
        net_store.append(net)
    
    if not net_store:
        raise ValueError("No network data found for the given datasets")
    
    net = pd.concat(net_store, ignore_index=True)
    
    # Filter edges based on minimum consensus across datasets
    net = net.groupby(['source', 'target']).filter(lambda x: len(x) >= min_consensus)

    # Filter for TFs as sources only
    net = net[net['source'].isin(tfs)]
    
    if net.empty:
        raise ValueError(f"No regulatory relationships found for TFs: {tfs}")
    
    # For each TF, get top targets based on absolute regulatory weights
    selected_edges = []
    all_nodes = set(tfs)  # Start with TFs
    
    for tf in tfs:
        tf_edges = net[net['source'] == tf].copy()
        if not tf_edges.empty:
            # Calculate absolute weights and get top targets
            tf_edges['abs_weight'] = tf_edges['weight'].abs()
            # Group by target and take mean absolute weight across datasets
            tf_targets = tf_edges.groupby('target')['abs_weight'].mean().sort_values(ascending=False)
            
            # Select top n_targets
            top_targets = tf_targets.head(n_targets).index.tolist()
            all_nodes.update(top_targets)
            
            # Add edges for these top targets
            tf_selected_edges = tf_edges[tf_edges['target'].isin(top_targets)]
            selected_edges.append(tf_selected_edges)
    
    if not selected_edges:
        raise ValueError("No edges found for the selected TFs")
    
    # Combine all selected edges
    filtered_net = pd.concat(selected_edges, ignore_index=True)
    

    # Calculate figure size based on number of nodes
    n_nodes = len(all_nodes)
    figsize = (max(6, n_nodes * 0.4), max(6, n_nodes * 0.4))
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_net_nx(filtered_net, ax=ax, figsize=figsize, draw_evidence=True, 
                palette_evidence=palette_datasets, offset_evidence=0.11, arc_offset=0.05)
    
    # Update title to reflect TF target programs
    tf_names = ", ".join(tfs[:3])  # Show first 3 TFs
    if len(tfs) > 3:
        tf_names += f" and {len(tfs)-3} more"
    
    plt.title(f"{cell_type_major} - Target Programs\n{tf_names}", fontsize=14, pad=20, weight='bold')
    
    return fig

def plot_net_nx(net, figsize=(6, 6), draw_evidence=True, rad_negative=-.3, rad_positive=0, palette_evidence=None, 
                offset_evidence = 0.1, arc_offset = 0.05, ax=None):

    G = nx.DiGraph()
    
    has_dataset = ('dataset' in net.columns) & draw_evidence

    # Aggregate datasets
    net['sign'] = np.sign(net['weight'])
    
    if has_dataset:
        edges = net.groupby(['source', 'target', 'sign'], as_index=False).agg({'dataset':list})
    else:
        edges = net.copy()
        edges['dataset'] = None

    # Add nodes and edges
    for _, row in edges.iterrows():
        source = row['source']
        target = row['target']
        datasets = row['dataset'] if has_dataset else None
        sign = row['sign']
        G.add_edge(source, target, weight=sign, datasets=datasets)

    node_size = 1000

    pos = nx.circular_layout(G)
    scale_factor = 0.3  # adjust this between 0 (tight) and 1 (default)
    pos = {k: v * scale_factor for k, v in pos.items()}
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=G.nodes(),
        node_color='white',
        node_size=node_size,
        edgecolors="black",
        linewidths=0.5,
        alpha=0.8,
        ax=ax
    )

    from matplotlib.patches import ArrowStyle
    rad_store = []
    for _, row in edges.iterrows():
        u = row['source']
        v = row['target']
        datasets = row['dataset'] if has_dataset else None
        sign = row['sign']
        def draw_edge(u, v, arrowstyle, rad):
            G_temp = nx.DiGraph()
            G_temp.add_edge(u, v)
            # Draw the edge with the specified parameters
            nx.draw_networkx_edges(
                G_temp, pos,
                edgelist=[(u, v)],
                # edge_color=[color],
                arrowsize=15,
                min_target_margin=20,
                min_source_margin=20,
                connectionstyle=f"arc3,rad={rad}",
                ax=ax,
                width=1.5,
                arrowstyle=arrowstyle,
            )
        
        # - check if there are multiple signs for the same edge (positive and negative)
        unique_edge = True
        df = edges[(edges['source'] == u) & (edges['target'] == v)]
        if df.groupby(['source', 'target'])['sign'].nunique().max() > 1:
            unique_edge = True

        stimulation = ArrowStyle(stylename="-|>", head_length=0.4, head_width=0.2, widthA=1.0, widthB=1.0, lengthA=0.2, lengthB=0.2, angleA=0, angleB=0, scaleA=None, scaleB=None)
        inhibition = ArrowStyle(stylename="-[", widthB=.3, lengthB=0, angleB=0)
        if unique_edge & (sign > 0):
            rad = 0
            arrowstyle = stimulation
        elif unique_edge & (sign < 0):
            rad = 0
            arrowstyle = inhibition
        elif not unique_edge:
            if sign > 0:
                rad = rad_positive
                arrowstyle = stimulation
            else:
                rad = rad_negative
                arrowstyle = inhibition
        rad_store.append(rad)
        draw_edge(u, v, arrowstyle, rad=rad)  
            
    font_size = 12
    nx.draw_networkx_labels(
        G, pos,
        labels={n: n for n in G.nodes},
        font_size=font_size,
        # font_weight='bold',
        ax=ax
    )

    plt.axis("off")
    plt.margins(x=0.1, y=0.1)

    # If dataset info is available, draw supporting evidence bars
    from matplotlib.patches import Rectangle
    from matplotlib import transforms

    if has_dataset:
        for i, (_, row) in enumerate(edges.iterrows()):
            source = row['source']
            target = row['target']
            sign = row['sign']
            datasets = row['dataset'] if has_dataset else None
            if datasets is None:
                continue

            x0, y0 = pos[source]
            x1, y1 = pos[target]

            # Direction vector
            dx, dy = x1 - x0, y1 - y0
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Normalize direction vector
            length = np.hypot(dx, dy)
            dx /= length
            dy /= length

            # New starting point: near the target
            base_x = x1 - offset_evidence * dx  # 0.1 can be tuned (how close to the target)
            base_y = y1 - offset_evidence * dy


            # Apply perpendicular offset for negative sign (curved edge)
            rad = rad_store[i]
            if rad != 0:
                # Shift along the edge *slightly more* to correct toward target node
                edge_offset_correction = -0.03  # adjust as needed
                base_x = base_x - edge_offset_correction * dx
                base_y = base_y - edge_offset_correction * dy

                # Add perpendicular offset to simulate the arc
                base_x += arc_offset * (-dy)  # perp_dx
                base_y += arc_offset * dx    # perp_dy

                angle += np.degrees(np.arctan(rad))

            # Box properties
            spacing_along_edge = 0.01
            box_width = 0.02
            box_height = 0.01

            for i, dataset in enumerate(datasets):
                offset = i * spacing_along_edge
                offset_x = base_x - offset * dx
                offset_y = base_y - offset * dy

                t = transforms.Affine2D().rotate_deg_around(offset_x, offset_y, angle + 90) + ax.transData

                rect = Rectangle(
                    (offset_x - box_width / 2, offset_y - box_height / 2),
                    box_width,
                    box_height,
                    transform=t,
                    color=palette_evidence[dataset],
                    alpha=0.9
                )
                ax.add_patch(rect)


        handles = [plt.Line2D([0], [0], color=palette_evidence[d], lw=6) for d in net['dataset'].unique()]
        pretty_names = [surrogate_names.get(d, d) for d in net['dataset'].unique()]
        plt.legend(
                handles, 
                pretty_names, 
                title="", 
                handlelength=1, 
                loc='lower left', 
                bbox_to_anchor=(1, 0.1),
                frameon=False, 
                fontsize=10
            )


def plot_net_degrees(net, top_n=10):
    plt.rcParams.update({'font.size': 10})
    out_degree = net.groupby("source").size().sort_values(ascending=False).head(top_n).reset_index(name="out_degree")
    in_degree = net.groupby("target").size().sort_values(ascending=False).head(top_n).reset_index(name="in_degree")

    # Set up the figure
    fig, axes = plt.subplots(1, 2, figsize=(4, 2.5))

    # Define color gradients
    out_colors = sns.color_palette("Blues", len(out_degree))[::-1]
    in_colors = sns.color_palette("Greens", len(in_degree))[::-1]

    # Plot out-degree
    sns.barplot(data=out_degree, x='out_degree', y='source', ax=axes[0], palette=out_colors)
    axes[0].set_title('TFs', pad=10, fontweight='bold', fontsize=10)
    axes[0].set_xlabel('Target genes')
    axes[0].set_ylabel('')
    axes[0].margins(y=0.05, x=.1)
    axes[0].spines[['top', 'right']].set_visible(False)

    # Plot in-degree
    sns.barplot(data=in_degree, x='in_degree', y='target', ax=axes[1], palette=in_colors)
    axes[1].set_title('Target genes', pad=10, fontweight='bold', fontsize=10)
    axes[1].set_xlabel('TFs')
    axes[1].set_ylabel('')
    axes[1].margins(y=0.05, x=.1)
    axes[1].spines[['top', 'right']].set_visible(False)


    plt.tight_layout()