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

from ciim.src.common import surrogate_names, palette_datasets


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
        "Increase": "green",
        "Decrease": "red",
        "Non-sig.": "gray"    
    }

    color_legend_sig = [plt.scatter([], [], s=100, marker='h', color=color, label=name) for name, color in palette_sig.items()]
    color_legend_sig = ax.legend(handles=color_legend_sig, title=" Trend in aging", loc=(0.0, 0.7), frameon=False)

    palette_reg = {
        "Positive": "green",
        "Negative": "red",
        
    }
    color_legend_reg = [plt.scatter([], [], s=100, color=color, label=name) for name, color in palette_reg.items()]
    color_legend_reg = ax.legend(handles=color_legend_reg, title="Regulatory sign", loc=(0.4, 0.76), frameon=False)

    # 2. Regulatory Weight Legend (Circle markers)
    size_legend_values = np.linspace(0, 1, num=6)
    size_legend_handles = [
        plt.scatter([], [], s=s * 200, color="black", label=f"{s:.1f}") for s in size_legend_values
    ]
    size_legend_handle = ax.legend(
        handles=size_legend_handles, title="Regulatory \n   weight", loc=(0.05, 0.1), frameon=False
    )

    # 3. Ageing Significance Legend (Hexagon markers)
    ageing_legend_handles = [
        plt.scatter([], [], s=s * 200, marker="h", color="black", label=f"{s:.1f}") for s in size_legend_values
    ]
    ageing_legend_handle = ax.legend(
        handles=ageing_legend_handles, title="Significance \n   in aging", loc=(0.4, 0.1), frameon=False
    )

    # Add legends manually
    ax.add_artist(color_legend_sig)
    ax.add_artist(color_legend_reg)
    ax.add_artist(size_legend_handle)
    return fig

class DotPlotTFtarget:
    linear_trend_palette = {'Increase': 'green', 'Decrease': 'red', 'Non-Sig': 'gray', 'Inconsistent': 'yellow'}
  
    def __init__(self, tf_all, skeleton):
        self.skeleton = skeleton
        self.tf_all = tf_all
 
    @staticmethod
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min())
    def plot_dotplot(self, 
                    data: pd.DataFrame, 
                    n_top_tfs=20,
                    n_top_targets=40,
                    title='', 
                    figsize=(10, 20), 
                    height_ratios=(4, .2), 
                    width_ratios=[0.3, 4], 
                    ax_main_margin=[0, 0]):
        data=data.sort_values('in_degree_c', ascending=False)
        # data['target'] = pd.Categorical(data['target'], categories=sorted_targets, ordered=True)
        
        # Subset to top TFs and targets
        top_tfs = data.sort_values('meta_p_adj').dropna()['source'].unique()[:n_top_tfs]
        top_targets = data.sort_values('meta_p_adj_target').dropna()['target'].unique()[:n_top_targets]
        data = data[data['source'].isin(top_tfs) & data['target'].isin(top_targets)]
        
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
        sns.scatterplot(data=data, 
                        x=np.zeros(len(data)),  
                        y='source', 
                        size='source_size',  
                        hue='linear_trend', 
                        palette=self.linear_trend_palette,
                        alpha=0.8, 
                        marker='h',
                        sizes=(10, 100),
                        legend=False,
                        ax=ax)

        ax.set_xticks([])
        ax.set_xlabel("")
        ax.set_ylabel("TF", fontsize=11)
        ax.spines[['top', 'right', 'bottom']].set_visible(False)
        ax.margins(y=ax_main_margin[1])

        ## --- MAIN PLOT: TF-Target Interactions ---
        ax = axes[0, 1]

        # Map True/False to edge colors
        edge_colors = data['in_skeleton'].map({True: 'black', False: 'white'})

        # Plot with edge colors
        sns.scatterplot(data=data, 
                        x='target', 
                        y='source', 
                        size=self.normalize(data['weight']),
                        hue='regulation_sig',
                        palette={'Positive': 'green', 'Negative': 'red'}, 
                        alpha=0.8, 
                        legend=False,
                        sizes=(10, 100),
                        edgecolor=edge_colors,  # Add edge coloring
                        linewidth=1,  # Adjust for visibility
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
        ax = axes[1, 1]
        sns.scatterplot(data=data, 
                        x='target', 
                        y=np.zeros(len(data)),  
                        size='target_size',  
                        hue='linear_trend_target', 
                        palette=self.linear_trend_palette,
                        alpha=0.8, 
                        marker='h',
                        sizes=(10, 100),
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
            label.set_color('#CC79A7' if label.get_text() in self.tf_all else '#56B4E9')

        # Hide bottom-left panel (empty)
        axes[1, 0].axis("off")
        
        plt.tight_layout()
        plt.show()
    def plot_legend(self):
        tf_gene_interaction_legend()

    def combine_data(self, stats_source, stats_target, net):
        from ciim.src.tf_activity.helper import compute_linear_trend
        from ciim.src.helper import determine_centrality, determine_indegree_centrality
        stats_source = compute_linear_trend(stats_source, pval_col='meta_p_adj', slope_col='slope', col='source')
        stats_source = stats_source[['source', 'meta_p_adj', 'linear_trend']].drop_duplicates()

        stats_target = compute_linear_trend(stats_target, pval_col='meta_p_adj', slope_col='slope', col='target')
        stats_target = stats_target.rename(columns={'meta_p_adj': 'meta_p_adj_target', 'linear_trend': 'linear_trend_target'})
        stats_target = stats_target[['target', 'meta_p_adj_target', 'linear_trend_target']].drop_duplicates()

        out_degree_c = determine_centrality(net, use_weight=False).reset_index().rename(columns={'index': 'source', 'centrality': 'out_degree_c'})
        in_degree_c = determine_indegree_centrality(net).reset_index().rename(columns={'index': 'target', 'centrality': 'in_degree_c'})

        net = net.merge(out_degree_c, on='source', how='left')
        net = net.merge(in_degree_c, on='target', how='left')

        # Merge precomputed expression significance stats
        
        data = net.merge(stats_target, on='target', how='left')
        data = data.merge(stats_source, on='source', how='left')

        return data
    def wrapper_combine_data(self, stats_source, stats_target, nets_dict):
        cell_types = stats_source['cell_type'].unique()
        data_all = [] 
        # Iterate over cell types and modules
        for cell_type in cell_types:
            print(cell_type)
            net = nets_dict[cell_type]
            
            stats_s = stats_source[stats_source['cell_type'] == cell_type].rename(columns={'tf': 'source'})
            stats_t = stats_target[stats_target['cell_type'] == cell_type]
            data = self.combine_data(stats_s, stats_t, net)

            data['cell_type'] = cell_type
            data_all.append(data)

        # Combine all processed data
        data_all = pd.concat(data_all)
        
        skeleton = self.skeleton

        skeleton['link'] = skeleton['source'] + '_' + skeleton['target']
        data_all['link'] = data_all['source'] + '_' + data_all['target']
        data_all['in_skeleton'] = data_all['link'].isin(skeleton['link'])
        return data_all

def plot_enrich(df, ax):
    sizes=None
    # Compute -log10(Adjusted P-value)
    df["neg_log10_padj"] = -np.log10(df["Adjusted P-value"])

    # Identify term-cell type pairs that have both Increase and Decrease
    multi_trend = df.groupby(["Term", "cell_type"])["linear_trend"].nunique()
    multi_trend_pairs = multi_trend[multi_trend > 1].index

    # Set color mapping
    palette = {"Increase": "green", "Decrease": "red"}

    # Scatter plot for each trend (Increase, Decrease)
    for trend, color in palette.items():
        subset = df[df["linear_trend"] == trend]
        ax.scatter(
            subset["cell_type"], subset["Term"], 
            s=np.log10(subset["neg_log10_padj"])*50,  # Scale size

            color=color, edgecolors=None,
            label=trend, alpha=0.5,
            linewidth=0.8,
            # sizes=(1, 100)
        )


    ax.set_xlabel("Cell Type")
    ax.set_ylabel("Term")
    ax.set_xlabel("")
    ax.margins(x=.2)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    # ax.spines[['right', 'top']].set_visible(False)
    # ax.legend(title="Trend", loc=(1.1, .8), frameon=False)
def compare_nets_plot(datasets, cell_types):
    """Compare and plot GRNs for each cell type between two datasets.

    Parameters:
    - datasets: list containing two datasets.
    - cell_types: list of cell types to analyze.

    Returns:
    - None (displays plots).
    """
    for cell_type in cell_types:
        # Get networks for this cell type from both datasets
        net1 = net_lambda(datasets[0], cell_type)
        net2 = net_lambda(datasets[1], cell_type)

        if net1 is None or net2 is None:
            continue  # Skip if no network exists for this cell type

        # Convert to sets of (source, target) pairs
        edges1 = set(map(tuple, net1[['source', 'target']].values))
        edges2 = set(map(tuple, net2[['source', 'target']].values))

        # Extract TFs (sources) and target genes
        tfs1, tfs2 = set(net1['source']), set(net2['source'])
        targets1, targets2 = set(net1['target']), set(net2['target'])

        # Compute statistics
        stats = {
            "Edges": [len(edges1 & edges2), len(edges1), len(edges2)],
            "TFs": [len(tfs1 & tfs2), len(tfs1 ), len(tfs2 )],
            "Targets": [len(targets1 & targets2), len(targets1), len(targets2)]
        }

        categories = ["Common", surrogate_names[datasets[0]], surrogate_names[datasets[1]]]

        # Create subplots (3 side-by-side)
        fig, axes = plt.subplots(1, 3, figsize=(6, 3), sharey=False)
        plot_titles = ["Edge", "TF", "Target"]
        colors = ['#2E86C1', '#E74C3C', '#27AE60']

        for i, (key, ax) in enumerate(zip(stats.keys(), axes)):
            sns.barplot(x=categories, y=stats[key], ax=ax, palette=colors)
            ax.set_title(plot_titles[i])
            ax.set_ylabel("Count")
            ax.set_xlabel("")
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.spines[['top', 'right']].set_visible(False)
            ax.margins(x=0.1)
        fig.suptitle(f'{cell_type}', fontsize=12)
        plt.tight_layout()
        plt.show()

def plot_trends(top_tfs, datasets, data_dict, datasets_colors, cell_type, surrogate_names={}, stats_df=None, is_expression=False):
    """
    Plots transcription factor (TF) activity/expression trends across datasets.

    Parameters:
    - top_tfs: list of top transcription factors to plot
    - datasets: list of dataset names
    - data_dict: dictionary containing either tf_acts or adata objects
    - datasets_colors: dictionary mapping datasets to colors
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

            ages = adata_sub.obs['age'].values
            expression = adata_sub.X.todense().A.flatten() if scipy.sparse.issparse(adata_sub.X) else adata_sub.X.flatten()
            cell_count = adata_sub.obs['cell_count'].values
            
            # expression = expression / expression[0]

            cell_count_n = cell_count / max(cell_count)

            # Scatter plot
            ax.scatter(
                ages, expression, 
                color=datasets_colors[dataset], 
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
                    p_value_adj = stats_df_sub.loc[:, 'meta_p_value'].values[0]
                    p_value_adj = min([p_value_adj, 1])  # Ensure p-value is not greater than 1
                else:
                    n_tests = len(top_tfs)*len(datasets)
                    p_value_adj = p_value * n_tests
                # Plot fitted line
                fitted_line = slope * age_range + intercept
                ax.plot(age_range, fitted_line, color=datasets_colors[dataset], linestyle='-', linewidth=2)

                

                # Create legend handle with both R² and Spearman ρ
                dataset_name = surrogate_names.get(dataset, dataset)
                legend_label = '     ' + dataset_name + '\n' + r' ($p_{adj}$=' + "{:.2e}".format(p_value_adj) + ")"

                handle = mpatches.Patch(color=datasets_colors[dataset], label=legend_label)
                legend_handles.append(handle)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xlabel('Age')
        ax.set_ylabel('Expression' if is_expression else 'TF Activity score')
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
    datasets = df_combined["dataset"].unique()
    fig, axes = plt.subplots(1, len(cell_types_local), figsize=(12, 3), sharey=False, sharex=False)
    
    for i, cell_type in enumerate(cell_types_local):
        ax = axes[i]
        df_cell_type = df_combined[df_combined["cell_type"] == cell_type]
        
        df_cell_type = add_centrality(df_cell_type, datasets) 
    
        # Plot scatter with correct color mapping
        sns.scatterplot(
            data=df_cell_type,
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
        xmin, xmax = df_cell_type[col_x].min(), df_cell_type[col_x].max()
        ymin, ymax = df_cell_type[col_y].min(), df_cell_type[col_y].max()

        global_min = min(xmin, ymin)
        global_max = max(xmax, ymax)

        placed_positions = []

        if i == 0:
            
            ax.set_ylabel(y_label)
        else:
            ax.set_ylabel("")
        ax.set_xlabel(x_label)
        ax.get_legend().remove()  # Remove legend from individual plot

        ax.set_title(cell_type, pad=20)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        ax.set_aspect("equal", adjustable="datalim")

        # Define padding as a percentage of the total range
        padding = 0.15 * (global_max - global_min)

        # Apply the same limits to both axes
        ax.set_xlim(global_min - padding, global_max + padding)
        ax.set_ylim(global_min - padding, global_max + padding)
        # Add significance threshold lines
        sig_threshold = 1.3
        ax.axvline(sig_threshold, linestyle="--", color="red", alpha=0.6, linewidth=1)  # Vertical
        ax.axvline(-sig_threshold, linestyle="--", color="red", alpha=0.6, linewidth=1)  # Vertical
        ax.axhline(sig_threshold, linestyle="--", color="red", alpha=0.6, linewidth=1)  # Horizontal
        ax.axhline(-sig_threshold, linestyle="--", color="red", alpha=0.6, linewidth=1)  # Horizontal

    handles, labels = ax.get_legend_handles_labels()
    # Create color legend for linear_trend
    color_legend = [mpatches.Patch(color=value, label=surrogate_names[name]) for name, value in palette_datasets.items()]
    color_legend_handle = plt.legend(handles=color_legend, title='Dataset', loc=(1.05, .8), frameon=False)

    # Create size legend for TF centrality
    size_legend_handle = plt.legend(handles=handles[-5:], labels=labels[-5:], title="TF centrality", 
                                    loc=(1.1, -0.3),  frameon=False)

    # Add the color legend manually after the size legend
    plt.gca().add_artist(color_legend_handle)
def compare_stats_across_datasets(df, y_label='TFs'):
    """Compare TFs for each cell type between two datasets.

    Parameters:
    - df: A DataFrame with columns 'cell_type', 'dataset', and 'tf'.

    Returns:
    - None (displays plots).
    """
    cell_types_local = df['cell_type'].unique()  # Get unique cell types
    datasets = df['dataset'].unique()  # Get unique datasets
    # Plot
    palette_datasets_local = {surrogate_names[key]: value for key, value in palette_datasets.items()}
    palette_datasets_local = {**palette_datasets_local, 'Common': 'green'}
    fig, axes = plt.subplots(1, 5, figsize=(9, 2.5), sharey=True)
    for ii, cell_type in enumerate(cell_types_local):
        # Filter the DataFrame for the current cell type
        df_cell_type = df[df['cell_type'] == cell_type]

        # Get the unique TFs for each dataset
        common_tfs = df_cell_type[df_cell_type['meta_p_adj']<0.05]['gene'].unique()
        df_1 = df_cell_type[df_cell_type['dataset'] == datasets[0]]
        df_2 = df_cell_type[df_cell_type['dataset'] == datasets[1]]
        tfs_dataset1 = set(df_1[df_1['p_value_adj']<0.05]['gene'])
        tfs_dataset2 = set(df_2[df_2['p_value_adj']<0.05]['gene'])

        # Compute the common and unique TFs
        
        unique_to_net1 = tfs_dataset1 
        unique_to_net2 = tfs_dataset2 

        # Prepare data for plotting
        data = pd.DataFrame({
            'Category': ['Common', surrogate_names[datasets[0]], surrogate_names[datasets[1]]],
            'Count': [len(common_tfs), len(unique_to_net1), len(unique_to_net2)]
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