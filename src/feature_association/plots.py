import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, linregress
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


from hiara.src.config import CELL_TYPES, OUTPUT_DIR, colors_blind, DISCOVERY_COHORTS ,surrogate_names, palette_datasets, palette_regulation, palette_trend, palette_datasets_pretty, mapping_minor_2_major, palette_trend_2
from hiara.src.feature_association.helper import calculate_tf_activity, bin_feature_values, retrieve_feature_data
from hiara.src.utils.util import retrieve_net, retrieve_adata


def plot_sig_tfs_stats(df, figsize=(3.5, 2), palette=None, ax=None):
    df = df[['gene', 'cell_type', 'trend']]
    df['trend'] = df['trend'].astype(CategoricalDtype(categories=palette.keys(), ordered=True))
    df = df[~df.duplicated()].reset_index(drop=True)
    df_counts = df.groupby(['cell_type', 'trend']).size().reset_index(name='Sig. TFs')
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.barplot(data=df_counts, x='cell_type', y='Sig. TFs', alpha=.8, hue='trend', palette=palette, ax=ax)
    ax.set_ylabel('TF count')
    ax.set_xlabel('')
    ax.margins(x=0.1, y=0.1)
    ax.legend(loc=(1, 0.5), title='Trend', frameon=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    # plt.suptitle('TFs significantly associated with age', fontsize=12)
    # plt.tight_layout()
class ModularizedNetPlot:
    @staticmethod
    def prepare_net_only_tfs(cell_type, race, data_type, min_degree=3):
        from hiara.src.feature_association.helper import retrieve_nets, retrieve_sig_stats

        stats_sig = retrieve_sig_stats(race=race, data_type=data_type).drop_duplicates(subset=['cell_type', 'gene'])
        stats_sig_t = stats_sig[stats_sig['cell_type'] == cell_type].set_index(['gene'])
        sig_tfs = stats_sig_t.index.unique()
        if race=='european':
            datasets = datasets_e
        elif race=='asian':
            datasets = datasets_a
        else:
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

        
        ax.add_artist(size_legend_handle)

def plot_feature_values_per_datasets(cell_type, features, data_type, datasets, feature_type='gene_expression', age_limit=[20, 75], cluster=False, figsize=None):
    import matplotlib.pyplot as plt
    from hiara.src.config import surrogate_names

    n_datasets = len(datasets)
    n_features = len(features)
    if figsize is None:
        figsize = (n_datasets*3, .2*n_features+1)
    fig, axes = plt.subplots(1, n_datasets, figsize=figsize, sharey=False)
    for i, (dataset) in enumerate(datasets):
        if feature_type == 'tf_activity':
            adata = retrieve_feature_data(dataset=dataset, cell_type=cell_type, data_type=data_type, feature_type=feature_type) 
        elif feature_type == 'gene_expression':
            adata = retrieve_adata(dataset=dataset, cell_type=cell_type, data_type=data_type)
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
        adata = adata[:, adata.var_names.isin(features)]
        
        if age_limit is not None:
            adata = adata[(adata.obs['age'] < age_limit[1]) & (adata.obs['age'] > age_limit[0])]
        # - plot target gene expression trend    
        ax = axes[i] if n_datasets > 1 else axes
        mean_expr = bin_feature_values(adata)
        if cluster:
            if i == 0:
                from sklearn.cluster import KMeans
                if len(mean_expr) > 2:
                    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(mean_expr)
                    mean_expr['cluster'] = clusters
                    mean_expr = mean_expr.sort_values('cluster').drop(columns=['cluster'])
                ordered_targets = mean_expr.index
            else:
                mean_expr = mean_expr.reindex(ordered_targets)
        else:
            mean_expr = mean_expr.reindex(features)
        if i == 0:
            show_cbar = True
        else:
            show_cbar = False
        
        heatplot_age_trend(mean_expr, cmap='viridis' if feature_type=='gene_expression' else 'magma', 
                            cbar_title="Gene expression" if feature_type=='gene_expression' else "TF activity", 
                            y_label="Genes" if feature_type=='gene_expression' else "TFs", 
                            ax=ax, 
                            show_cbar=show_cbar)
        if i != 0:
            ax.set_yticklabels([])
            ax.set_ylabel('')
        ax.set_title(surrogate_names[dataset], pad=10, fontsize=10, fontweight='bold')
    # plt.tight_layout()
    # plt.suptitle(cell_type, fontsize=12, fontweight='bold', y=1.05)
    return fig
def plot_feature_values_all_datasets(cell_type, feature, feature_type, datasets, ax=None, show_cbar=True, data_type='bulk', 
                                    age_limit=[20, 80], show_ylabels=True):
    from hiara.src.feature_association.helper import retrieve_feature_data, bin_feature_values
    
    mean_expr_store = []
    for dataset in datasets:
        if feature_type == 'tf_activity':
            adata = retrieve_feature_data(dataset=dataset, cell_type=cell_type, data_type=data_type, feature_type=feature_type) 
        elif feature_type == 'gene_expression':
            adata = retrieve_adata(dataset=dataset, cell_type=cell_type, data_type=data_type)
        else:
            raise ValueError(f"Unsupported feature data_type: {feature_type}")
        adata = adata[(adata.obs['age'] > age_limit[0]) & (adata.obs['age'] < age_limit[1])]
        adata = adata[:, adata.var_names==feature]
        assert adata.shape[1] == 1, f"Feature {feature} not found in dataset {dataset} for cell type {cell_type}"
        if adata.shape[1] == 0:
            continue
        expr = bin_feature_values(adata)
        expr.index = [dataset]
        mean_expr_store.append(expr)
    
    if len(mean_expr_store) == 0:
        return
    mean_expr = pd.concat(mean_expr_store)

    mean_expr.index = mean_expr.index.map(surrogate_names)
    ages = sorted(mean_expr.columns)
    # print(mean_expr)
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 2))

    heatplot_age_trend(mean_expr[ages], cmap='magma' if feature_type=='tf_activity' else 'viridis', 
                        cbar_title = "Gene \n expression" if feature_type == 'gene_expression' else (
                                    "TF \n activity" if feature_type == 'tf_activity' else "Gene score"
                                ),
                        ax=ax, 
                        show_cbar=show_cbar,
                        cbar_kws={
                            "shrink": 1,
                            "aspect": 5,       # Lower values = thicker colorbar (default is ~20)
                            "fraction": 0.1    # Controls the width space the cbar takes in the figure
                        })
    if not show_ylabels:
        ax.set_yticklabels([])
    ax.set_ylabel('')
    ax.set_xlabel('Age')
    # ax.set_title(f'{cell_type}: {feature}', pad=20)

def plot_trend_sle_case(adata, tf='LEF1', cell_type='CD8T'):
    from scipy.stats import linregress

    adata = retrieve_feature_data(dataset='SLE_European', cell_type=cell_type, feature_type='tf_activity', condition=None)
    adata = adata[:, adata.var_names == tf]
    # Extract feature values (flattened, assuming dense matrix)
    feature = adata.X.flatten()  # use .toarray().flatten() if sparse
    age = adata.obs['age'].values

    data = pd.DataFrame({
        'feature': feature,
        'age': age
    })

    # Define masks
    young_mask = data['age'] < 50
    old_mask = data['age'] >= 50

    fig, axes = plt.subplots(1, 2, figsize=(3, 2.2))

    i = 0
    for ax, (mask, color, title) in zip(
        axes,
        [(young_mask, 'royalblue', 'age < 50'), (old_mask, 'darkorange', 'age > 50')]
    ):
        subdata = data[mask]
        ax.scatter(
            subdata['age'],
            subdata['feature'],
            color=color,
            alpha=0.7,
            s=15
        )
        slope, intercept, r_value, p_value, std_err = linregress(subdata['age'], subdata['feature'])
        x_vals = np.linspace(subdata['age'].min(), subdata['age'].max(), 100)
        y_vals = slope * x_vals + intercept
        ax.plot(x_vals, y_vals, color='black', linestyle='--', linewidth=2)
        ax.set_yticks([])
        ax.margins(x=0.1, y=0.2)
        sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
        text = f'p = {p_value:.3g}{sig}' if p_value < 0.05 else f'p = {p_value:.3g}'
        # ax.text(0.5, 1.05, text, transform=ax.transAxes,
        #         ha='center', va='top', fontsize=10, color='black')
        if i == 0:
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_xlabel('Age')
        else:
            ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.set_title(f'{title}\n{text}', pad=15, fontsize=10)
        i+=1
    axes[0].set_ylabel('TF activity')
    plt.suptitle(f'{tf}', fontsize=10, fontweight='bold', y=.9)
    plt.tight_layout()

def plot_analysis_and_centrality(df, all_groups, palette_all, feature_col='gene', figsize=(3.5, 5), plot_centrality=True,
                                ax2_margins={'y': 0.1, 'x': 0.1}, hide_ylabels=False, show_legend=True):
    
    # stats_d_sig = stats_d[stats_d['p_value_adj'] < 0.05]
    # --- Plot ---
    tfs = df[feature_col].unique()
    if plot_centrality:
        fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1.2, .8]})
    else:
        fig, axes = plt.subplots(1, 1, figsize=figsize)

    # - Scatter plot
    ax0 = axes[0] if plot_centrality else axes
    sns.scatterplot(data=df, x='analysis', y=feature_col, hue='trend', ax=ax0, palette=palette_all, s=100, alpha=.8)

    # Overlay black stars
    aging_df = df[df['analysis'] == 'Age-associated']
    assert aging_df.shape[0] > 0, "No Aging TFs found in the data"
    ax0.scatter(aging_df['analysis'], aging_df[feature_col], color='black', marker='*', s=10, zorder=10)
    sig_df = df[df.get('p_value_adj', 1.0) < 0.05]
    ax0.scatter(sig_df['analysis'], sig_df[feature_col], color='black', marker='*', s=10, zorder=10, alpha=.8)

    # Fill in missing x-axis categories
    missing = set(all_groups) - set(df['analysis'].unique())
    for cat in missing:
        ax0.scatter(cat, df[feature_col].iloc[0], color='white', alpha=0)

    ax0.set_xticklabels(ax0.get_xticklabels(), rotation=45, ha="right")
    ax0.set_xlabel('')
    ax0.set_ylabel('TFs' if feature_col == 'gene' else 'Pathways')
    ax0.margins(x=.2, y=.05 if len(tfs) > 10 else 0.2)
    ax0.spines[['top', 'right']].set_visible(False)
    if hide_ylabels:
        ax0.set_yticklabels([])
        ax0.set_ylabel('')
    for spine in ax0.spines.values():
        spine.set_linewidth(0.5)
    ax0.get_legend().remove()
    # - Degree barplot
    if plot_centrality:
        ax1 = axes[1]
        bar_data = df.drop_duplicates(subset=feature_col)
        sns.barplot(data=bar_data, x='degree', y=feature_col, ax=ax1, color='#56B4E9', alpha=0.7, ci=None)
        ax1.set_xlabel('Centrality')
        ax1.set_ylabel('')
        ax1.set_yticks([])
        ax1.margins(**ax2_margins)
        ax1.spines[['top', 'right']].set_visible(False)
        
    if show_legend:
        # - Place legend on the outer right of both subplots
        ordered_labels = ['Decrease in aging', 'Increase in aging', 'Decrease after treatment', 'Increase after treatment', 'Decrease in disease', 'Increase in disease']
        ordered_labels = [label for label in ordered_labels if label in palette_all.keys()]
        handles, labels = ax0.get_legend_handles_labels()

        # Create a dictionary from labels to handles
        label_handle_dict = dict(zip(labels, handles))

        # Reorder handles and labels
        ordered_handles = [label_handle_dict[label] for label in ordered_labels]
        ordered_labels = [label for label in ordered_labels]

        # Add legend
        fig.legend(ordered_handles, ordered_labels, loc='center left', bbox_to_anchor=(1.02, 0.8), frameon=False)

    
def plot_overlap(
        stats_drug_sig, 
        aging_stats_sig, 
        agreement,  # treatment effect should be 'opposite' to aging
        how='left',
        col='cell_type',
        ax=None,
        legend=True,
        figsize=(2.5, 2),
        legend_loc=(1.05, 0.5)
    ):
    

    if not pd.api.types.is_categorical_dtype(stats_drug_sig['cell_type']):
        stats_drug_sig['cell_type'] = stats_drug_sig['cell_type'].astype('category')

    included_celltypes = stats_drug_sig['cell_type'].cat.categories 

    merged = aging_stats_sig.merge(stats_drug_sig, on=['gene', col], how=how)
    merged['slope_sign'] = np.sign(merged['slope'])
    merged['slope_condition_sign'] = np.sign(merged['slope_condition'])
    merged = merged.drop_duplicates(subset=['gene', col, 'slope_sign', 'slope_condition_sign'])
    merged['trend'] = merged['slope_sign'].map({1: 'positive', -1: 'negative'})
    merged = merged[merged['cell_type'].isin(included_celltypes)]

    def compute_agreement(group):
        total = len(group)
        group = group[(~group['slope_sign'].isna()) & (~group['slope_condition_sign'].isna())]
        if agreement == 'opposite':
            agree = (group['slope_sign'] != group['slope_condition_sign']).sum()
        elif agreement == 'same':
            agree = (group['slope_sign'] == group['slope_condition_sign']).sum()
        else:
            raise ValueError("Agreement must be either 'opposite' or 'same'")
        return pd.Series({'n_total_tfs': total, 'n_agreeing_tfs': agree})

    summary = (
        merged.groupby([col, 'trend'])
        .apply(compute_agreement)
        .reset_index()
    )

    if col == 'cell_type':
        cell_types = [t for t in CELL_TYPES if t in summary[col].unique()]
        summary[col] = pd.Categorical(summary[col], categories=cell_types, ordered=True)

    summary['trend'] = ['Increase in aging' if x == 'positive' else 'Decrease in aging' for x in summary['trend']]
    df = summary.copy()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Background bars
    offset = {'Decrease in aging': -0.22, 'Increase in aging': 0.22}
    width=0.35
    colors = {'Decrease in aging': 'tab:blue', 'Increase in aging': 'tab:red'}
    x_locs = {cat: i for i, cat in enumerate(summary[col].cat.categories)}
    for trend, offset_val in offset.items():
        df_trend = df[df['trend'] == trend]
        xpos = [x_locs[val] + offset_val for val in df_trend[col]]
        ax.bar(
            xpos,
            df_trend['n_total_tfs'],
            width=width,
            alpha=0.3,
            color=palette_trend_2[trend],
            edgecolor='black',
            linewidth=0.1,
            label=trend
        )

    for i, row in df.iterrows():
        base_x = x_locs[row[col]]
        xpos = base_x + offset[row['trend']]
        ax.bar(
            xpos,
            row['n_agreeing_tfs'],
            width=width,
            edgecolor=colors[row['trend']],
            facecolor='none',
            hatch='///',
            linewidth=1,
            zorder=1
        )

        y_pos = row['n_total_tfs']
        ax.text(
            xpos,
            y_pos + 2 + y_pos*.1*np.random.rand(),  # Random offset for better visibility,
            f"{int(100*(row['n_agreeing_tfs']/y_pos))}%",
            ha='center',
            va='bottom',
            fontsize=8
        )

    ax.set_xticks(list(x_locs.values()))
    ax.set_xticklabels(list(x_locs.keys()), rotation=45, ha='right')
    ax.set_ylabel("Number of TFs")
    ax.set_xlabel("")
    ax.margins(x=0.1, y=0.2)
    ax.spines[['right', 'top']].set_visible(False)

    # Custom legend
    handles, labels = ax.get_legend_handles_labels()
    if agreement == 'opposite':
        label = 'Rejuvination effect'
    elif agreement == 'same':
        label = 'Age acceleration effect'
    
    hatch_patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label=label)
    handles.append(hatch_patch)
    if legend:
        ax.legend(handles=handles, title='', loc=legend_loc, frameon=False)


def plot_gene_score_association_with_age(cell_type, datasets, data_type, features=None, feature_type='tf_activity', sizes=(50, 100), 
                              top_features=20, min_degree=4, filter_meta_significant=False, race='european', width=3,
                             margins_ax1={'x': 0.1, 'y': 0.1}, margins_ax2={'x': 0.1, 'y': 0.1}, show_size_legend = False, figsize=None,
                             n_top_terms=20):
    from hiara.src.utils.plots import dotplot
    from matplotlib.colors import TwoSlopeNorm
    from hiara.src.config import cmap_trend, palette_trend_2, surrogate_names
    from hiara.src.feature_association.helper import retrieve_features_stats, retrieve_sig_stats
    import matplotlib.gridspec as gridspec
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    if feature_type == 'tf_activity':
        feature_col = 'source'
    elif feature_type == 'gene_expression':
        feature_col = 'target'
    elif feature_type == 'gene_score':
        feature_col = 'pathway'
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    # - format the data
    stats_t = retrieve_features_stats(data_type=data_type, feature_type=feature_type, cell_type=cell_type, datasets=datasets, condition='healthy')
    
    if 'gene' in stats_t.columns:
        stats_t = stats_t.rename(columns={'gene': 'source'})
    if filter_meta_significant:
        stats_sig = retrieve_sig_stats(data_type=data_type, feature_type=feature_type)
        stats_sig = stats_sig[stats_sig['cell_type'] == cell_type]
        sig_tfs = stats_sig[feature_col].unique()
        stats_t = stats_t[stats_t[feature_col].isin(sig_tfs)]
    if features is None:
        features = stats_t[feature_col].unique()
        print(len(features), f'{feature_col}s found in {cell_type} for {feature_type}')
    stats_t = stats_t[stats_t[feature_col].isin(features)]
    
    # - add the number of genes in each pathway
    c_store = []
    for dataset in datasets:
        df = retrieve_feature_data(dataset=dataset, cell_type=cell_type, data_type=data_type, feature_type='gene_score', smoothened=False).var
        df.index.name = 'pathway'
        df = df.reset_index()
        df['dataset'] = dataset
        c_store.append(df)
    c = pd.concat(c_store)
    c_median = c.groupby([feature_col])['n_matching_genes'].median().reset_index()
    c_median = c_median[c_median[feature_col].isin(features)]
    # print(c_median.sort_values(by='n_matching_genes', ascending=False))
    # aa
    c_median = c_median.sort_values(by='n_matching_genes', ascending=False).head(n_top_terms)
    c_std = c.groupby([feature_col])['n_matching_genes'].std().reset_index(name='n_matching_genes_std')
    stats_t = stats_t.merge(c_median, left_on=feature_col, right_on=feature_col, how='inner')
    stats_t = stats_t.merge(c_std, left_on=feature_col, right_on=feature_col, how='inner')
    # sort based on median gene count
    features = c_median[feature_col].tolist()
    stats_t[feature_col] = pd.Categorical(stats_t[feature_col], categories=features, ordered=True)
    stats_t = stats_t.sort_values(by=[feature_col, 'dataset'])
    # print(stats_t[feature_col].nunique(), 'features after merging with gene count')
    

    stats_t['neg_log10_adj_pval'] = -np.log10(stats_t['p_value_adj'])
    stats_t['dataset'] = pd.Categorical(stats_t['dataset'], categories=datasets, ordered=True)
    stats_t['dataset'] = stats_t['dataset'].apply(lambda name: surrogate_names.get(name, name))

    if stats_t.shape[0]==0:
        print(f'No data for {cell_type} {feature_col}')
        raise ValueError(f'No data for {cell_type} {feature_col}')
    if True:
        # - main plot
        df = stats_t.copy()
        
        if figsize is None:
            figsize = (width, .2*len(features)+1.5)
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 0.5, .4])  # middle space reserved for legend
        
        ax = fig.add_subplot(gs[0])
        ax_legend = fig.add_subplot(gs[-1])
        ax_legend.set_axis_off()
        
        if len(features) < 7:
            size_legend_loc = None
            cbar_height='10%'
            cbar_width = "50%"
            bbox_to_anchor_cbar=(1.25, -.5, 1, 1)
        elif len(features) < 15:
            size_legend_loc = None
            cbar_height='10%'
            cbar_width = "50%"
            bbox_to_anchor_cbar=(1.25, -.5, 1, 1)

        else:
            bbox_to_anchor_cbar=(1.2, -.2, 1, 1)
            size_legend_loc=(.95, -.4, 1, 1)
            cbar_height='5%'
            cbar_width="50%"
        
        # Ensure feature_col is a categorical with the desired order
        unique_features = df[feature_col].unique()
        df[feature_col] = pd.Categorical(df[feature_col], categories=unique_features, ordered=True)
        ordered_features = df[feature_col].cat.categories  
        if df['dataset'].nunique() != len(datasets):
            print( f"Only {df['dataset'].nunique()} datasets are available in the stats.")   
        dotplot(df, 
                x='dataset',
                y = feature_col,
                ax=ax, 
                ax_legend=ax_legend,
                color_col='slope', 
                size_col='neg_log10_adj_pval', 
                palette=cmap_trend, 
                show_color_legend=True, 
                show_size_legend=show_size_legend,
                alpha=1,
                size_legend_title='-Log10 p-value',
                color_legend_title='Correlation \nwith aging',
                size_legend_loc=size_legend_loc,
                bbox_to_anchor_cbar=bbox_to_anchor_cbar,
                cbar_height=cbar_height,
                cbar_width=cbar_width,
                linewidth=0.1,  
                sizes=sizes,
                size_legend_scale = 200/max(df['neg_log10_adj_pval']),
                )
        
        ax.margins(**margins_ax1)
        ax.set_ylabel('TF' if feature_col == 'source' else ('Gene' if feature_col == 'target' else 'Pathway'))
        title = 'TF activity' if feature_col == 'source' else ('Gene expression' if feature_col == 'target' else 'Gene score')
        ax.set_title(f'{title} - {cell_type}', pad=10, fontsize=10, fontweight='bold')

    # ------------ centrality
    c = c[c[feature_col].isin(features)]
    c[feature_col] = pd.Categorical(c[feature_col], categories=ordered_features, ordered=True)
    ax = fig.add_subplot(gs[1])
    if True:
        sns.barplot(
            data=df,
            x='n_matching_genes',
            y=feature_col,
            ax=ax,
            color='#56B4E9',
            alpha=0.7,
            ci=None,  # turn off seaborn's built-in error estimation
            errorbar=('sd', df['n_matching_genes_std']),  # pass your own std values
            errwidth=1.2,
            capsize=0.2
        )
    

    ax.spines[['top', 'right']].set_visible(False)
    ax.margins(**margins_ax2)
    ax.set_xlabel('Gene count')
    ax.set_ylabel('')
    ax.set_yticks([])
    plt.subplots_adjust(wspace=0.1)

    return fig
def plot_features_vs_datasets(cell_type, datasets, data_type, features=None, feature_type='tf_activity', sizes=(50, 100), 
                              top_features=20, min_degree=4, filter_meta_significant=False, race='european', 
                              show_size_legend=False):

    from hiara.src.utils.plots import dotplot
    from hiara.src.config import cmap_trend, surrogate_names
    from hiara.src.feature_association.helper import retrieve_features_stats, retrieve_sig_stats
    from hiara.src.utils.util import retrieve_net
    import matplotlib.gridspec as gridspec
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Calculate base dimensions based on data size
    n_datasets = len(datasets)
    # Estimate number of features for initial sizing (will be refined later)
    estimated_n_features = top_features if features is None else len(features) if features is not None else top_features
    
    # Base dimensions calculated from data characteristics - tighter width, looser height
    base_width = max(1.5, min(3.5, 1.0 + n_datasets * 0.2))  # Tighter width range: 1.5-3.5 instead of 1.8-4
    base_height = max(0.12, min(0.25, 0.2 - estimated_n_features * 0.002))  # Smaller height per row for many features

    if feature_type == 'tf_activity':
        feature_col = 'source'
    elif feature_type == 'gene_expression':
        feature_col = 'target'
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    # - format the data
    stats_t = retrieve_features_stats(data_type, feature_type, cell_type=cell_type)
    stats_t = stats_t[stats_t['dataset'].isin(datasets)]
    # print(stats_t)
    
    if 'gene' in stats_t.columns:
        stats_t = stats_t.rename(columns={'gene': 'source'})
    if filter_meta_significant:
        stats_sig = retrieve_sig_stats(data_type=data_type, feature_type=feature_type)
        if 'gene' in stats_sig.columns:
            stats_sig = stats_sig.rename(columns={'gene': 'source'})
        stats_sig = stats_sig[stats_sig['cell_type'] == cell_type]
        sig_tfs = stats_sig[feature_col].unique()
        stats_t = stats_t[stats_t[feature_col].isin(sig_tfs)]
    
    # - get centrality measure
    c_store = []
    for dataset in datasets:
        net = retrieve_net(dataset, cell_type)
        c = net.groupby(feature_col).size()
        c = c.div(c.max())
        c = c.reset_index(name='centrality')
        c['dataset'] = dataset
        c_store.append(c)
    c = pd.concat(c_store)
    c_median = c.groupby([feature_col])['centrality'].median().reset_index()
    c_std = c.groupby([feature_col])['centrality'].std().reset_index(name='centrality_std')

    # print(c_median)
    # print(stats_t)
    stats_t = stats_t.merge(c_median, left_on=feature_col, right_on=feature_col, how='left')
    stats_t = stats_t.merge(c_std, left_on=feature_col, right_on=feature_col, how='left')
    
    # - either find the central features and sort them or sort them based on the given features    
    if features is None:
        # - select the top features: top shared across datasets and top central
        degrees = c.groupby(feature_col).size().sort_values(ascending=False)
        degrees = degrees[degrees >= min_degree]
        c_median_c = c_median[c_median[feature_col].isin(degrees.index)]
        c_median_c = c_median_c[c_median_c[feature_col].isin(stats_t[feature_col].unique())]
        
        features = c_median_c.sort_values('centrality', ascending=False).head(top_features)[feature_col].unique() # subset to top ones
        
        stats_t = stats_t[stats_t[feature_col].isin(features)]
        stats_t = stats_t.sort_values('centrality', ascending=False)
    else:
        if feature_type == 'tf_activity':
            pass
            # - check if the features are in the tf_all list
            # tf_all = np.loadtxt(f"{PRIOR_DIR}/tf_all.csv", dtype=str)
            # features = [tf for tf in features if tf in tf_all]

        # - check if the features are in the stats (remove those that are not present in at least one dataset)
        stats_t = stats_t[stats_t[feature_col].isin(features)]
        features = [tf for tf in features if tf in stats_t[feature_col].unique()]
        features = list(set(features))  # remove duplicates
        stats_t[feature_col] = pd.Categorical(stats_t[feature_col], categories=features, ordered=True)
        stats_t = stats_t.sort_values(feature_col)  
    stats_t['neg_log10_adj_pval'] = -np.log10(stats_t['p_value_adj'])
    stats_t['dataset'] = pd.Categorical(stats_t['dataset'], categories=datasets, ordered=True)
    stats_t['dataset'] = stats_t['dataset'].apply(lambda name: surrogate_names.get(name, name))

    if stats_t.shape[0]==0:
        print(f'No data for {cell_type} {feature_col}')
        raise ValueError(f'No data for {cell_type} {feature_col}')
        
    # Calculate automated layout parameters
    n_features = len(features)
    # Update base_height now that we know the actual number of features
    # Use logarithmic scaling for many features - more generous spacing (looser)
    if n_features <= 10:
        base_height = 0.1  # More generous height for very small number of features
    elif n_features <= 20:
        base_height = 0.16  # More generous height for small number of features
    elif n_features <= 50:
        base_height = 0.12  # More generous for medium number of features
    else:
        # For large feature sets, ensure minimum spacing between dots while keeping reasonable total height
        base_height = max(0.08, 0.20 / np.log10(n_features))  # Increased minimum to prevent dot overlap
    
    
    # Automated figure sizing with better scaling for many features - smaller width scaling
    width_factor = max(1, min(1.5, n_datasets / 8))  # Even smaller width scaling
    
    # Use square root scaling for height to prevent excessive stretching - more conservative
    if n_features <= 10:
        height_factor = 1
    elif n_features <= 50:
        height_factor = max(1, np.sqrt(n_features / 15))  # More conservative scaling
    elif n_features <= 100:
        height_factor = max(1, np.sqrt(n_features / 30))  # Even more conservative for medium-large sets
    else:
        # Very gentle scaling for large feature sets - much less aggressive
        height_factor = max(1, np.log10(n_features) * 1)  # Increased from 0.1 to 0.5 for better spacing
    fig_width = base_width * width_factor + 1.5  # Reduced legend space from 2 to 1.5
    fig_height = base_height * n_features * height_factor + 1.5  # Reduced title/label space from 2 to 1.5
        
    # Automated margin calculation - scalable based on features and datasets
    # X-axis margins: Scale with number of datasets (more datasets need tighter spacing)
    x_margin_base = 0.15  # Reduced base margin for x-axis 
    x_margin_scale = max(0.1, min(0.20, x_margin_base / np.sqrt(n_datasets)))
    
    # Y-axis margins: Scale with number of features - balanced for large sets
    y_margin_base = 0.10  # Reduced base margin for y-axis
    if n_features <= 10:
        y_margin_scale = 1
    elif n_features <= 20:
        y_margin_scale = .5
    elif n_features <= 100:
        y_margin_scale = max(0.02, min(0.10, y_margin_base / np.log10(n_features)))
    else:
        # For large feature sets, use moderate margins since we increased base_height
        y_margin_scale = .01
    
    # Additional scaling based on figure size - more balanced
    width_correction = min(1.3, fig_width / 4)  # Balanced width correction
    height_correction = min(1.3, fig_height / 10)  # Balanced height correction
    
    margins_ax1 = {
        'x': x_margin_scale * width_correction,
        'y': y_margin_scale * height_correction
    }
    margins_ax2 = {
        'x': x_margin_scale * width_correction * 0.8,  # Slightly tighter for centrality plot
        'y': y_margin_scale * height_correction
    }
    
    # Automated legend positioning based on number of features
    if n_features <= 5:
        size_legend_loc = None
        cbar_height = '15%'
        cbar_width = "60%"
        bbox_to_anchor_cbar = (1.25, -0.3, 1, 1)
        wspace = 0.15
    elif n_features <= 10:
        size_legend_loc = None
        cbar_height = '12%'
        cbar_width = "50%"
        bbox_to_anchor_cbar = (1.25, -0.4, 1, 1)
        wspace = 0.12
    elif n_features <= 20:
        size_legend_loc = None
        cbar_height = '8%'
        cbar_width = "45%"
        bbox_to_anchor_cbar = (1.2, -0.5, 1, 1)
        wspace = 0.1
    elif n_features <= 50:
        bbox_to_anchor_cbar = (1.2, -0.4, 1, 1)
        size_legend_loc = (0.95, -0.5, 1, 1)
        cbar_height = '6%'
        cbar_width = "40%"
        wspace = 0.09
    elif n_features <= 100:
        bbox_to_anchor_cbar = (1.2, -0.3, 1, 1)
        size_legend_loc = (0.95, -0.4, 1, 1)
        cbar_height = '5%'
        cbar_width = "40%"
        wspace = 0.08
    else:
        bbox_to_anchor_cbar = (1.2, -0.6, 1, 1)
        size_legend_loc = (0.97, -0.7, 1, 1)
        cbar_height = '1%'
        cbar_width = "40%"
        wspace = 0.1
        
    # Adjust grid width ratios based on data
    centrality_width = min(0.6, max(0.3, 0.4 + n_features * 0.01))
    legend_width = min(0.5, max(0.3, 0.3 + n_features * 0.005))
    width_ratios = [1, centrality_width, legend_width]
        
        
    # - main plot
    df = stats_t.copy()
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    gs = gridspec.GridSpec(1, 3, width_ratios=width_ratios)
    
    ax = fig.add_subplot(gs[0])
    ax_legend = fig.add_subplot(gs[-1])
    ax_legend.set_axis_off()
    
    unique_features = df[feature_col].unique()
    df[feature_col] = pd.Categorical(df[feature_col], categories=unique_features, ordered=True)
    ordered_features = df[feature_col].cat.categories  
    if df['dataset'].nunique() != len(datasets):
        print( f"Only {df['dataset'].nunique()} datasets are available in the stats.")   
    
    dotplot(df, 
            x='dataset',
            y = feature_col,
            ax=ax, 
            ax_legend=ax_legend,
            color_col='slope', 
            size_col='neg_log10_adj_pval', 
            palette=cmap_trend, 
            show_color_legend=True, 
            show_size_legend=show_size_legend,
            alpha=1,
            size_legend_title='-Log10 p-value',
            color_legend_title='Correlation \nwith aging',
            size_legend_loc=size_legend_loc,
            bbox_to_anchor_cbar=bbox_to_anchor_cbar,
            cbar_height=cbar_height,
            cbar_width=cbar_width,
            linewidth=0.1,  
            sizes=sizes,
            size_legend_scale = 200/max(df['neg_log10_adj_pval']),
            )
    ax.spines[['top', 'right']].set_visible(False)
    ax.margins(**margins_ax1)
    ax.set_ylabel('TFs' if feature_col=='source' else 'Genes')
    title = 'TF activity' if feature_col=='source' else 'Gene expression'
    ax.set_title(f'{title} - {cell_type}', pad=10, fontsize=10, fontweight='bold')
    
    # ------------ centrality
    c = c[c[feature_col].isin(features)]
    c[feature_col] = pd.Categorical(c[feature_col], categories=ordered_features, ordered=True)
    ax = fig.add_subplot(gs[1])
    
    sns.barplot(
        data=df,
        x='centrality',
        y=feature_col,
        ax=ax,
        color='#56B4E9',
        alpha=0.7,
        ci=None,  # turn off seaborn's built-in error estimation
        errorbar=('sd', df['centrality_std']),  # pass your own std values
        errwidth=1.2,
        capsize=0.2
    )
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.margins(**margins_ax2)
    ax.set_xlabel('Centrality\n(out-degree)' if feature_col=='source' else 'Centrality\n(in-degree)')
    ax.set_ylabel('')
    ax.set_yticks([])
    
    # Apply scalable figure-level margins and spacing
    # Calculate outer margins based on plot dimensions and content
    left_margin = max(0.08, min(0.2, 0.1 + 0.02 * np.log10(n_features)))  # More space for y-labels
    right_margin = max(0.85, min(0.95, 0.9 - 0.01 * n_datasets))  # Space for legends
    bottom_margin = max(0.1, min(0.25, 0.15 + 0.02 * np.log10(n_datasets)))  # Space for x-labels
    top_margin = max(0.9, min(0.98, 0.95 - 0.005 * n_features))  # Space for title
    
    plt.subplots_adjust(
        left=left_margin,
        right=right_margin, 
        bottom=bottom_margin,
        top=top_margin,
        wspace=wspace
    )

    return fig

def plot_tf_interactions_plus_target_stats_binary(net, ax=None, show_legend=True, sizes=(20, 200), annotate_sig=True, annotate_targets=False):
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Define binary color map
    binary_colors = {'positive': '#2ca02c', 'negative': '#d62728'}  # Green and Red

    # Classify weight into positive or negative
    net = net.copy()
    net['regulation'] = net['weight'].apply(lambda w: 'positive' if w > 0 else 'negative')

    # Optional: Rename datasets using surrogate names if applicable
    net['dataset'] = net['dataset'].apply(lambda name: surrogate_names.get(name, name))

    # Slope direction affects marker shape
    net['slope_direction'] = net['slope'].apply(lambda x: 'Increase in aging' if x > 0 else 'Decrease in aging')

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))

    sns.scatterplot(
        data=net,
        x='target',
        y='dataset',
        hue='regulation',
        palette=binary_colors,
        style='slope_direction',
        markers={'Increase in aging': '^', 'Decrease in aging': 'v'},
        ax=ax,
        s=100,
        alpha=0.7,
        legend=False  # We'll build a custom legend
    )

    ax.set_ylabel('Cohort', fontsize=12, labelpad=10)
    ax.set_xlabel('Targets', fontsize=12, labelpad=10)
    # ax.margins(x=0.05, y=0.4)
    plt.xticks(rotation=90)

    # Optionally annotate targets that are TFs
    if annotate_targets:
        tf_all = np.loadtxt(f"{PRIOR_DIR}/tf_all.csv", dtype=str)
        plt.draw()
        for label in ax.get_xticklabels():
            if label.get_text() in tf_all:
                label.set_color('#56B4E9')  # blue for TFs

    # Build custom legend
    if show_legend:
        style_legend = [
            Line2D([0], [0], marker='^', color='w', label='Increase in aging', markerfacecolor='gray', markersize=8),
            Line2D([0], [0], marker='v', color='w', label='Decrease in aging', markerfacecolor='gray', markersize=8)
        ]

        color_legend = [
            Line2D([0], [0], marker='o', color='w', label='Positive regulation', markerfacecolor=binary_colors['positive'], markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Negative regulation', markerfacecolor=binary_colors['negative'], markersize=10)
        ]

        spacer = Line2D([0], [0], linestyle="none", label="")

        all_handles = style_legend + [spacer] + color_legend

        ax.legend(
            handles=all_handles,
            loc='center left',
            bbox_to_anchor=(1.01, 0.5),
            borderaxespad=0,
            frameon=False
        )
def plot_tf_interactions_plus_target_stats(net, ax=None, show_legend=True, sizes=(20, 200), annotate_sig=True, annotate_targets=False):
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from pandas.api.types import CategoricalDtype
    cmap = plt.cm.RdYlGn  # Red = negative, Green = positive
    norm = TwoSlopeNorm(vmin=-.1, vcenter=0, vmax=.1)
    
    net['dataset'] = net['dataset'].apply(lambda name: surrogate_names.get(name, name))
    net['slope_direction'] = net['slope'].apply(lambda x: 'Increase in aging' if x > 0 else 'Decrease in aging')

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    # Sort by target alphabetically
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
        s = 100,
        alpha=0.7,
        legend=False  # Suppress default legend
        )
    ax.set_ylabel('Cohort', fontsize=12, labelpad=10)
    ax.set_xlabel('Targets', fontsize=12, labelpad=10)
    ax.margins(x=0.05, y=0.2)
    plt.xticks(rotation=90)
    if annotate_targets:
        tf_all = np.loadtxt(f"{PRIOR_DIR}/tf_all.csv", dtype=str)
        # Set tick labels with color
        plt.draw()  # ensures tick labels are populated

        # Loop through the tick labels and modify their color
        for label in ax.get_xticklabels():
            label_text = label.get_text()
            if label_text in tf_all:
                label.set_color('#56B4E9')
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

        size_values = np.percentile(net['neg_log10_adj_pval'], [25, 50, 75])
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

        # all_handles = style_legend + [spacer] + color_legend + [spacer] + size_legend
        all_handles = style_legend + [spacer] + color_legend

        ax.legend(
            handles=all_handles,
            loc='center left',
            bbox_to_anchor=(1.01, 0.5),
            borderaxespad=0,
            title='',
            frameon=False
        )
    
    # - annotate significant targets
    if annotate_sig:
        pval_threshold = 1.4
        x_vals = net['target'].astype('category').cat.codes.values
        y_vals = net['dataset'].astype('category').cat.codes.values

        for (x, y), (_, row) in zip(zip(x_vals, y_vals), net.iterrows()):
            if row['neg_log10_adj_pval'] >= pval_threshold:
                ax.text(x, y + 0.1, '*', ha='center', va='center', alpha=0.9, 
                        fontsize=8, weight='bold', color='black', zorder=10)


def wrapper_flesh_out_tf_interactions(datasets, cell_type, tf, data_type='bulk', n_top=10, keep_sig_only=False, sizes=(20, 100), ax=None, show_legend=True):
    from hiara.src.feature_association.helper import retrieve_features_stats
    from hiara.src.feature_association.plots import plot_tf_interactions_plus_target_stats
    # - get the net for different datasets
    top_targets = []
    net_store = []
    for dataset in datasets:
        net = retrieve_net(dataset, cell_type)[['source', 'target', 'weight', 'cell_type']]
        net = net[net['source'] == tf].copy()
        top_targets.append(net.sort_values(by='weight', ascending=False, key=abs).head(n_top)['target'].tolist())
        net['dataset'] = dataset
        net_store.append(net)
    net = pd.concat(net_store)
    top_targets = np.unique(np.concatenate(top_targets))
        
    # - get the stats of targets per dataset 
    stats_targets = retrieve_features_stats(data_type, feature_type='gene_expression', cell_type=cell_type, condition='healthy')
    stats_targets = stats_targets[stats_targets['dataset'].isin(datasets)]
    net_stats = net.merge(stats_targets[['dataset', 'target', 'p_value_adj', 'slope']], on=['dataset', 'target'], how='left')
    net_stats = net_stats[~net_stats['p_value_adj'].isna()]
    net_stats['neg_log10_adj_pval'] = -np.log10(net_stats['p_value_adj'])

    net_stats = net_stats[net_stats['target'].isin(top_targets)]
    if keep_sig_only:
        net_stats = net_stats[net_stats['neg_log10_adj_pval'] > 1.4]
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(len(top_targets)*.2 + 1, len(datasets)*.2 + 1))

    plot_tf_interactions_plus_target_stats(net_stats.copy(), ax=ax, show_legend=show_legend, sizes=sizes)

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
    adata_all = retrieve_adata(dataset)

    adata = adata_all[adata_all.obs['cell_type'] == cell_type]
    nets = retrieve_net(dataset, cell_type)
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
    obs['donor_age'] = obs['age'].astype(str) + '_' + obs['donor_id'].astype(str)
    # obs = obs[obs.age<=75]
    obs['age'] = pd.to_numeric(obs['age'], errors='coerce')
    min_age = obs.age.min()
    bins = [min_age,45, 100]  
    age_groups = ['45-', '45+']  
    obs['age_group'] = pd.cut(obs['age'], bins=bins, labels=age_groups, right=False)
    return obs
def plot_trend_tfs(cell_type, tfs, data_type='bulk', dataset='data1', ax=None):
    adata = retrieve_adata(dataset, data_type=data_type)
    adata = adata[adata.obs['cell_type'] == cell_type]
    nets = retrieve_net(dataset, cell_type)
    tf_acts = calculate_tf_activity(adata, nets)
    from hiara.src.process_dataset.preprocess.helper import binarize_age
    tf_acts.obs = binarize_age(tf_acts.obs)

    tf_acts_s = tf_acts[:, tf_acts.var_names.isin(tfs)]
    mean_expr = cluster_trends(tf_acts_s)
    

    heatplot_age_trend(mean_expr, cmap='magma', cbar_title="TF activity", y_label="TFs", ax=ax)
    plt.title(f'{cell_type}: TF activity trend', pad=20)

    sorted_tfs = mean_expr.index
    return sorted_tfs
def plot_trend_targets_binarized(cell_type, genes, dataset='data1', ax=None):
    adata = retrieve_adata(dataset)
    adata = adata[adata.obs['cell_type'] == cell_type]
    adata = adata[:, adata.var_names.isin(genes)]
    from hiara.src.process_dataset.preprocess.helper import binarize_age
    adata.obs = binarize_age(adata.obs)
    # print(adata.obs.groupby(['age_group'])['donor_age'].nunique())

    mean_expr = cluster_trends(adata)

    heatplot_age_trend(mean_expr, cmap="viridis", cbar_title="Gene expression", y_label="Targets", ax=ax)
    plt.title(f'{cell_type}: gene expression trend', pad=20)

    sorted_tfs = mean_expr.index
    return sorted_tfs

def plot_overall_heatmap(stats_all, 
                        first_col='cell_type', first_col_palette=None,
                        second_col='dataset', second_col_palette=None,
                        figsize=(6, 8), 
                        sig_dots_y_offset=.8,
                        map_names={},
                        bbox_to_anchor=(1.1, 1),
                        bbox_to_anchor_col2=(1.1, .75),
                        bbox_to_anchor_col1=(1.1, 0.4),
                        trend_colors = ['#B0BF1A', '#E52B50'],
                        trend_names = ['Decrease in aging', 'Increase in aging'],
                        dendrogram_visible=True,
                        draw_legend=True

                        ):
    from hiara.src.config import palette_trend_2
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from scipy.cluster.hierarchy import linkage
    from matplotlib.patches import Patch


    # format the data
    first_col_unique_values = stats_all[first_col].cat.categories
    second_col_unique_values = stats_all[second_col].cat.categories
    
    stats_all = stats_all[stats_all[second_col].isin(second_col_unique_values)]
    stats_all['trend_int'] = stats_all['slope'].map(lambda value: 1 if value > 0 else (-1 if value < 0 else 0))
    multi_index = pd.MultiIndex.from_product([first_col_unique_values, second_col_unique_values], names=[first_col, second_col])
    pivot_df = stats_all.pivot(index='gene', columns=[first_col, second_col], values="trend_int")
    df_plot = pivot_df.reindex(columns=multi_index).fillna(0)

    # - format the is_sig if needed
    if 'is_significant' in stats_all.columns:
        sig_df = stats_all.pivot(index='gene', columns=[first_col, second_col], values="is_significant")
        sig_df = sig_df.reindex(columns=multi_index).fillna(0)
    else:
        sig_df = None

    # prepare for plot
    col_colors = pd.DataFrame({
        map_names.get(second_col, second_col): [second_col_palette.get(analysis, "gray") for _, analysis in df_plot.columns],
        map_names.get(first_col, first_col): [first_col_palette.get(ct, "lightgray") for ct, _ in df_plot.columns]
    }, index=df_plot.columns)

    row_linkage = linkage(df_plot, method='ward')
    # print(palette_trend)
    cmap = ListedColormap([trend_colors[0], 'white', trend_colors[1]])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = BoundaryNorm(bounds, cmap.N)

    g = sns.clustermap(
        df_plot,
        row_linkage=row_linkage,
        col_cluster=False,
        row_cluster=True,
        cmap=cmap,
        norm=norm,
        col_colors=col_colors,
        linewidths=1,
        alpha=.8,
        linecolor=None,
        figsize=figsize,
    )
    g.ax_row_dendrogram.set_visible(dendrogram_visible)
    # if True: # tick labels
    #     g.ax_heatmap.set_xticklabels([])


    g.cax.set_visible(False)
    g.ax_heatmap.set_yticks([])
    g.ax_heatmap.set_ylabel('', fontsize=10, labelpad=5)
    
    g.ax_heatmap.set_xticks([])
    g.ax_heatmap.set_xlabel('', fontsize=12, labelpad=15)

    # - add sig if given
    if sig_df is not None:
        row_order = g.dendrogram_row.reordered_ind
        col_order = list(df_plot.columns)  # Column order stays the same since col_cluster=False

        cell_height = g.ax_heatmap.get_position().height / len(row_order)
        cell_width = g.ax_heatmap.get_position().width / len(col_order)
        # Loop through and add asterisks for significant values
        for i, row_idx in enumerate(row_order):
            for j, col in enumerate(col_order):
                # Extract cell_type and gender from col (tuple format)
                col1, col2 = col
                is_significant = sig_df.loc[sig_df.index[row_idx], (col1, col2)]
                
                # Check if the value is significant
                if is_significant:
                    y_coord = i - sig_dots_y_offset
                    x_coord = (j + 0.5) 
                    
                    g.ax_heatmap.text(
                        x_coord, y_coord,
                        '.',
                        color='black', ha='center', va='center', fontsize=8, fontweight='bold'
                    )
    # Legends
    if draw_legend:
        datasets_legend = [Patch(color=second_col_palette[label], label=map_names.get(label, label)) for label in second_col_unique_values]
        celltype_legend = [Patch(color=first_col_palette[label], label=map_names.get(label, label)) for label in first_col_unique_values]
        trend_legend = [Patch(color=color, label=label, alpha=.8) for label, color in zip(trend_names, trend_colors)]
        legend_datasets = g.ax_heatmap.legend(
            handles=datasets_legend,
            title=map_names.get(second_col, second_col),
            bbox_to_anchor=bbox_to_anchor_col2, 
            loc='upper left',
            fontsize=9,
            title_fontsize=9,
            frameon=False
        )
        legend_datasets.get_title().set_fontweight('bold')  

        legend_celltypes = g.ax_heatmap.legend(
            handles=celltype_legend,
            title=map_names.get(first_col, first_col),
            bbox_to_anchor=bbox_to_anchor_col1,
            loc='upper left',
            fontsize=9,
            title_fontsize=9,
            frameon=False
        )
        legend_celltypes.get_title().set_fontweight('bold') 

        legend_trend = g.ax_heatmap.legend(
            handles=trend_legend,
            title="Trend",
            bbox_to_anchor=bbox_to_anchor,
            loc='upper left',
            fontsize=9,
            title_fontsize=9,
            frameon=False
        )
        legend_trend.get_title().set_fontweight('bold')  


        g.ax_heatmap.add_artist(legend_celltypes)
        g.ax_heatmap.add_artist(legend_datasets) 
    # Tighten layout to reduce whitespace
    # plt.subplots_adjust(top=1.1)
    # plt.show()

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

def plot_joint_scatter(stats_all, col='cell_type', vars=['CD4T', 'CD8T'], annotate=True, figsize=(2.5, 3), ax=None):
    # - plot
    xy_vars = [f'{v}_pval' for v in vars]
    trend_vars = [f'{v}_trend' for v in vars]

    stats_all_table = stats_all.pivot(index='gene', columns=col, values='neg_log10_adj_pval').reset_index().fillna(0)
    stats_all_trend = stats_all.pivot(index='gene', columns=col, values='trend').reset_index()
    stats_all_table = stats_all_table.merge(stats_all_trend, on='gene', suffixes=('_pval', '_trend'))
    stats_all_table[trend_vars] = stats_all_table[trend_vars].fillna('Inconsistent')

    stats_all_table['trend'] = stats_all_table[trend_vars].apply(
                            lambda x: 'Increase in aging' if (x[trend_vars[0]]=='Increase in aging' and x[trend_vars[1]]=='Increase in aging') else ('Decrease in aging' if (x[trend_vars[0]]=='Decrease in aging' and x[trend_vars[1]]=='Decrease in aging') else 'Inconsistent') , axis=1)
    stats_all_table['trend'] = stats_all_table['trend'].astype(CategoricalDtype(categories=['Increase in aging', 'Decrease in aging', 'Inconsistent'], ordered=True))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
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
    ax.set_xlabel(vars[0] + '\n' + r'($- \log_{10} p$)')
    ax.set_ylabel(vars[1] + '\n' + r'($- \log_{10} p$)')
    ax.axvline(x=1.4, color=colors_blind[0], linestyle='--')
    ax.axhline(y=1.4, color=colors_blind[0], linestyle='--')

    ax.margins(x=0.1, y=0.1)
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend(loc=(1.1, 0.2), title='Trend', frameon=False)
    
    if annotate:
        quantile = .8
        high_tf_points = stats_all_table[(stats_all_table[xy_vars[0]] > stats_all_table[xy_vars[0]].quantile(quantile))&
                                                (stats_all_table[xy_vars[1]] > stats_all_table[xy_vars[1]].quantile(quantile))
                                            ]
        
        inconsistent_tfs = stats_all_table[stats_all_table['trend'] == 'Inconsistent']

        df_to_annotate = pd.concat([inconsistent_tfs, high_tf_points], axis=0)
        
        x_offset = 10*np.asarray([-1, -1, 0, -1, 1, .5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0])
        y_offset = 10*np.asarray([ -1,  1, 1, -1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0])

        ii = 0
        for idx, row in df_to_annotate.iterrows():
            
            # Adjust the annotation position slightly away from the point
            ax.annotate(
                row['gene'], 
                xy=(row[xy_vars[0]], row[xy_vars[1]]), 
                xytext=(row[xy_vars[0]]  + x_offset[ii], row[xy_vars[1]] + y_offset[ii]),  # Adjust this value for distance
                textcoords='data',
                color='black', 
                fontsize=8, 
                ha='left', va='top',
                arrowprops=dict(arrowstyle="->", color='black', lw=0.5)  # Arrow pointing to the point
            )
            ii += 1
    # plt.show()
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
def plot_net_nx(net, figsize=(6, 6), draw_evidence=True, rad_negative=-.3, rad_positive=0, palette_evidence=None, 
                offset_evidence = 0.1, arc_offset = 0.05, ax=None):
    import networkx as nx

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
def wrapper_draw_net(cell_type, datasets, features, min_degree=3, indivitual_net=True, draw_evidence=True, draw_collectri=True,figsize=(4, 4), figsize_collectri=(3,3), 
                     offset_evidence=.11, arc_offset=.05, offset_evidence_collectri=.1, promotor_only=False):
    net_store = []
    for dataset in datasets:
        cell_type_major = mapping_minor_2_major.get(cell_type, cell_type)
        net = retrieve_net(dataset=dataset, cell_type=cell_type_major, promotor_only=promotor_only)
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
        collectri = pd.read_csv(f'{PRIOR_DIR}/collectri_with_source.csv')
        collectri['dataset'] = collectri['ref']
        evidence = collectri.copy()
        if False:
            # - add skeleton
            # skeleton = pd.read_csv(f'/home/jnourisa/projs/ongoing/task_grn_inference/resources/grn_benchmark/prior//skeleton.csv')
            skeleton = pd.read_csv(f'{PRIOR_DIR}/skeleton_promotor.csv')
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

def wrapper_draw_tf_target_programs(cell_type, datasets, tfs, n_targets=10, promotor_only=False, min_consensus=3):
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
        net = retrieve_net(dataset=dataset, cell_type=cell_type_major, promotor_only=promotor_only)
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

def heatplot_age_trend(mean_expr, cmap="viridis", cbar_title="Gene expression", y_label="Genes", figsize=(2.5, 3), 
                ax=None, show_cbar=True, cbar_kws={
                                        "shrink": 1,
                                        "aspect": 10,       # Lower values = thicker colorbar (default is ~20)
                                        "fraction": 0.1    # Controls the width space the cbar takes in the figure
                                    }):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    # Plot heatmap
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(mean_expr, cmap=cmap, cbar=show_cbar, 
                cbar_kws=cbar_kws,
                 ax=ax)
    ax.set_yticks(np.arange(mean_expr.shape[0]) + 0.5)
    ax.set_yticklabels(mean_expr.index, rotation=0)

    # Modify the colorbar
    if show_cbar:
        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel(cbar_title, rotation=90, labelpad=5)

        # Set ticks at the min and max values of the colorbar
        vmin, vmax = cbar.vmin, cbar.vmax
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels(['0', '1'], rotation=0, fontsize=7)

    # Labels and formatting
    ax.set_xlabel("Age")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

def heamap_plot_minor_cell_types(stats_all, palette, 
                                            map_names, 
                                            slope_col='slope',
                                            main_col='major_cell_type', 
                                            minor_col='cell_type' ,figsize=(6, 8), sig_dots_y_offset = 0.5, 
                                            annotate_x_ticks=True, dendrogram_visible=True,
                                            show_legend=True):

    from hiara.src.config import palette_cell_types, surrogate_names
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from scipy.cluster.hierarchy import linkage
    from matplotlib.patches import Patch


    major_cell_types = stats_all[main_col].cat.categories
    cell_types = stats_all[minor_col].cat.categories

    stats_all=stats_all[stats_all[minor_col].isin(cell_types)]
    
    stats_all['trend_int'] = stats_all[slope_col].map(lambda value: 1 if value > 0 else (-1 if value < 0 else 0))
    pivot_df = stats_all.pivot(index='gene', columns=minor_col, values='trend_int').fillna(0)
    
    pivot_df = pivot_df.reindex(columns=cell_types).fillna(0)


    # - color map
    cols_names = pivot_df.columns.map(lambda name: mapping_minor_2_major.get(name, name))
    col_colors = [palette_cell_types[name] for name in cols_names]

    palette_values = list(palette.values())
    cmap = ListedColormap([palette_values[0], 'white', palette_values[1]])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = BoundaryNorm(bounds, cmap.N)
    assert not pivot_df.isnull().values.any(), "NaNs found in pivot_df"
    g = sns.clustermap(
        pivot_df,
        row_linkage=linkage(pivot_df, method='ward'),
        col_cluster=False,
        row_cluster=True,
        cmap=cmap,
        # norm=norm,
        col_colors=col_colors,
        linewidths=1,
        alpha=.8,
        linecolor=None,
        figsize=figsize,
    )
    g.ax_row_dendrogram.set_visible(dendrogram_visible)

    g.cax.set_visible(False)
    g.ax_heatmap.set_yticks([])
    g.ax_heatmap.set_ylabel('', fontsize=10, labelpad=5)
    new_labels = [surrogate_names.get(label.get_text(), label.get_text()) for label in g.ax_heatmap.get_xticklabels()]
    if annotate_x_ticks:
        g.ax_heatmap.set_xticklabels(new_labels, rotation=90)  # or any angle you prefer
    else:
        g.ax_heatmap.set_xticks([])

    g.ax_heatmap.set_xlabel('', fontsize=12, labelpad=15)


    # - format the is_sig if needed
    if 'is_significant' in stats_all.columns:
        sig_df = stats_all.pivot(index='gene', columns='cell_type', values="is_significant").fillna(0)
        sig_df = sig_df.reindex(columns=cell_types)
    else:
        sig_df = None
    if sig_df is not None:
            row_order = g.dendrogram_row.reordered_ind
            col_order = list(cell_types)  # Column order stays the same since col_cluster=False

            cell_height = g.ax_heatmap.get_position().height / len(row_order)
            cell_width = g.ax_heatmap.get_position().width / len(col_order)
            # Loop through and add asterisks for significant values
            for i, row_idx in enumerate(row_order):
                for j, col in enumerate(col_order):
                    # Extract cell_type and gender from col (tuple format)
                    is_significant = sig_df.loc[sig_df.index[row_idx], col]
                    
                    # Check if the value is significant
                    if is_significant:
                        y_coord = i - sig_dots_y_offset
                        x_coord = (j + 0.5) 
                        
                        g.ax_heatmap.text(
                            x_coord, y_coord,
                            '.',
                            color='black', ha='center', va='center', fontsize=8, fontweight='bold'
                        )
    if show_legend:
        celltype_legend = [Patch(color=palette_cell_types[label], label=map_names.get(label, label)) for label in major_cell_types]
        trend_legend = [Patch(color=color, label=label, alpha=.8) for label, color in palette.items()]

        legend_celltypes = g.ax_heatmap.legend(
            handles=celltype_legend,
            title=map_names.get(main_col, main_col),
            bbox_to_anchor=(1.1, 0.6),
            loc='upper left',
            fontsize=9,
            title_fontsize=9,
            frameon=False
        )
        legend_celltypes.get_title().set_fontweight('bold') 

        legend_trend = g.ax_heatmap.legend(
            handles=trend_legend,
            title="Trend",
            bbox_to_anchor=(1.1, 1),
            loc='upper left',
            fontsize=9,
            title_fontsize=9,
            frameon=False
        )
        legend_trend.get_title().set_fontweight('bold')  


        g.ax_heatmap.add_artist(legend_celltypes)


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
class DotPlotTFtarget:
    trend_palette = {'Increase in aging': 'green', 'Decrease in aging': 'red', 'Non-Sig': 'gray', 'Inconsistent': 'yellow'}
  
    def __init__(self, tf_all, skeleton):
        self.skeleton = skeleton
        self.tf_all = tf_all
 
    @staticmethod
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min())
    def plot_dotplot(self, 
                    data, 
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
        from hiara.src.feature_association.helper import compute_trend
        from hiara.src.helper import determine_centrality

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
        stats_s = stats_source.rename(columns={'gene': 'source'})
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
                s=cell_count_n * 50
            )

            # Fit linear regression
            if len(ages) > 1:
                age_range = np.linspace(min(ages), max(ages), 100)
                spearman_corr, spearman_p = spearmanr(ages, expression)
                slope, intercept, r_value, p_value, _ = linregress(ages, expression)
                r2 = r_value**2
                # - correct for multiple testing
                if stats_df is not None:
                    stats_df_sub = stats_df[(stats_df['gene'] == tf) & (stats_df['dataset'] == dataset)]
                    p_value_adj = stats_df_sub.loc[:, 'meta_p_adj'].values[0]
                    p_value_adj = min([p_value_adj, 1])  # Ensure p-value is not greater than 1
                else:
                    n_tests = len(top_tfs)*len(datasets)
                    p_value_adj = p_value * n_tests
                # Plot fitted line
                fitted_line = slope * age_range + intercept
                # ax.plot(age_range, fitted_line, color=palette[dataset], linestyle='-', linewidth=2)

                # Create legend handle with both R² and Spearman ρ
                dataset_name = surrogate_names.get(dataset, dataset)
                legend_label = '    ' + dataset_name + f' ({slope.round(2)})'+'\n' + r' ($p-value$=' + "{:.2e}".format(p_value) + ")"

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
                                  x_label = "Expression\nsigned -log10(p adj)",
                                  figsize=(4, 2.7)):
    import matplotlib.patches as mpatches
    cell_types_local = df_combined["cell_type"].unique()
    n_cell_types = len(cell_types_local)
    for i, cell_type in enumerate(cell_types_local):
        df_cell_type = df_combined[df_combined["cell_type"] == cell_type]
        fig, ax = plt.subplots(1, 1, figsize=figsize, sharey=False, sharex=False)
        df_dataset = df_cell_type
        df_dataset['dataset'] = df_dataset['dataset'].map(surrogate_names)
        assert df_dataset.shape[0]>0, f"No data for {cell_type} in {datasets[j]}"
        sns.scatterplot(
            data=df_dataset,
            x=col_x,
            y=col_y,
            palette=palette_datasets_pretty,  # Use the consistent color mapping
            # size="centrality",
            s=10,
            hue="dataset",
            # sizes=(20, 100),
            edgecolor=None,
            alpha=0.7,
            ax=ax,
        )
        top_tfs = df_dataset.sort_values(by='p_value_adj_target', ascending=False).head(2)
        # compute small offsets relative to axis ranges
        x_range = df_dataset[col_x].max() - df_dataset[col_x].min()
        y_range = df_dataset[col_y].max() - df_dataset[col_y].min()
        x_offset = 0.2 * x_range

        for _, row in top_tfs.iterrows():
            ax.text(
                row[col_x] + x_offset,
                row[col_y] + np.random.rand() * .1 *  y_range,
                row['gene'],  # assumes TF names are in column 'gene'
                fontsize=7,
                ha='left',    # anchor text to the left since we shift right
                va='bottom',  # anchor text above since we shift up
                color='black'
            )
        # Annotate top genes with the highest activation significance
        xmin, xmax = df_dataset[col_x].min(), df_dataset[col_x].max()
        ymin, ymax = df_dataset[col_y].min(), df_dataset[col_y].max()
        global_min = min(xmin, ymin)
        global_max = max(xmax, ymax)
        placed_positions = []
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label, labelpad=15)        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_aspect("equal", adjustable="datalim")
        padding = 0.15 * (global_max - global_min)
        sig_threshold = 1.4
        ax.margins(x=0.01, y=0.05)
        linewidth = .5
        alpha = .4
        ax.axvline(sig_threshold, linestyle="--", color="red", alpha=alpha, linewidth=linewidth)  # Vertical
        ax.axvline(-sig_threshold, linestyle="--", color="red", alpha=alpha, linewidth=linewidth)  # Vertical
        ax.axhline(sig_threshold, linestyle="--", color="red", alpha=alpha, linewidth=linewidth)  # Horizontal
        ax.axhline(-sig_threshold, linestyle="--", color="red", alpha=alpha, linewidth=linewidth)  # Horizontal
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, fontsize=10, title='Dataset', title_fontsize=10)
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