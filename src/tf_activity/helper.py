import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from scipy.stats import linregress, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import r2_score
import pandas as pd
import scipy
import shap
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
from ciim.src.helper import determine_centrality
from statsmodels.stats.multitest import multipletests
from pandas.api.types import CategoricalDtype
from ciim.src.common import datasets, datasets_healthy, datasets_disease, surrogate_names, datasets_all
from scipy.stats import mannwhitneyu


tf_all = np.loadtxt(f"/home/jnourisa/projs/ongoing/task_grn_inference/resources/grn_benchmark/prior/tf_all.csv", dtype=str)

def net_lambda(dataset, cell_type):  
    net = pd.read_csv(f"/home/jnourisa/projs/ongoing/ciim/output/grns/{dataset}/net_{cell_type}_all_agegroups_all_batches.csv")
    return net.loc[net['source'].isin(tf_all)]

def adata_lambda(dataset, type='bulk'): 
    base_path = "/vol/projects/jnourisa/datasets/"
    assert type in ['bulk', 'sc', 'bulk_minor']
    file_name =  f"{dataset}_{type}.h5ad"
    return ad.read_h5ad(f"{base_path}{file_name}")

def determine_stats_disease(tf_acts, age_cutoff=50, ctr_group='normal', disease_col='disease'):

    conditions = tf_acts.obs['disease'].unique()
    # case 1: association with age in all samples
    stats_df_1 = linear_association_with_age(tf_acts, genes=tf_acts.var_names)
    stats_df_1['age_group'] = 'all'
    stats_df_1['condition'] = 'all'
    # case 2: association with age for each disease group
    results = []
    for group in conditions:
        tf_acts_sub = tf_acts[tf_acts.obs[disease_col] == group]
        stats_df = linear_association_with_age(tf_acts_sub, genes=tf_acts_sub.var_names)
        stats_df['age_group'] = 'all'
        stats_df['condition'] = group
        results.append(stats_df)
    stats_df_2 = pd.concat(results)
        
    # case 3: disease vs healthy in each age group
    results = []
    mask = tf_acts.obs['age'] > age_cutoff 

    for age_group, mask_age in zip(["young", "old"], [~mask, mask]):  
        tf_acts_age = tf_acts[mask_age]

        for condition in conditions:
            if condition == ctr_group:
                continue
            mask_ctr = tf_acts_age.obs[disease_col] == ctr_group
            mask_condition = tf_acts_age.obs[disease_col] == condition
            
            control_group = tf_acts_age.X[mask_ctr, :].toarray()
            case_group = tf_acts_age.X[mask_condition, :].toarray()

            if (np.sum(mask_condition) < 10) or (np.sum(mask_ctr) < 10):
                continue

            for i, gene in enumerate(tf_acts_age.var_names):
                values_case = case_group[:, i]
                values_control = control_group[:, i]

                if np.sum(values_case) == 0 and np.sum(values_control) == 0:
                    continue  

                stat, pval = mannwhitneyu(values_case, values_control, alternative="two-sided")
                # direction = "Increase in disease" if np.median(values_case) > np.median(values_control) else "Decrease in disease"

                results.append({
                    "tf": gene,
                    "p_value": pval,
                    "slope_disease":  np.median(values_case) > np.median(values_control) ,
                    "values_case": values_case,
                    "values_control": values_control,
                    "age_group": age_group,
                    "condition": condition
                })     


    stats_df_3 = pd.DataFrame(results)
    if stats_df_3.shape[0] > 0:
        stats_df_3["adj_p_value"] = multipletests(stats_df_3["p_value"], method="fdr_bh")[1]
    else:
        print("Warning: No significant results found in stats_df_3.")
    

    # Combine all stats
    stats_df = pd.concat([stats_df_1, stats_df_2, stats_df_3], ignore_index=True)
    return stats_df
        
# def adata_cell_type_lambda(dataset, cell_type):
#     adata = adata_lambda(dataset)
#     if cell_type=='T':
#         adata_sub = adata[adata.obs['cell_type'].isin(['CD4T', 'CD8T'])]
#         cols = adata_sub.obs.columns
#         unique_cols = ['donor_id', 'age']
#         other_cols = [col for col in cols if col not in unique_cols+['cell_type', 'cell_count', 'sum_by']]

#         df = pd.DataFrame(
#             adata_sub.X.toarray() if hasattr(adata_sub.X, "toarray") else adata_sub.X,  # Convert sparse to dense if needed
#             columns=adata_sub.var_names,
#             index=pd.MultiIndex.from_frame(adata_sub.obs[unique_cols])  # Keep MultiIndex
#         ).reset_index()  # Convert MultiIndex back to normal columns
#         df = df.groupby(unique_cols).mean().dropna()
#         adata_mean = ad.AnnData(X=df.values, obs=df.reset_index()[unique_cols], var=pd.DataFrame(index=df.columns))
#         obs = adata_mean.obs.copy()
#         adata_mean.obs = obs.merge(adata_sub.obs[unique_cols+other_cols], on=unique_cols, how='left').drop_duplicates().reset_index(drop=True)
#         adata_mean.obs['cell_count'] = 1000
#         adata_mean.obs['cell_type'] = 'T'
#         adata_sub = adata_mean
#     else:
#         adata_sub = adata[adata.obs['cell_type'] == cell_type]
#     return adata_sub
from ciim.src.common import mapping_major_2_minor

def add_minor_data(df, cell_type):
    stats_minor = pd.read_csv('../output/tf_activation/stats_minor.csv')

    stats_minor = process_rr(stats_minor)
    subcelltypes = mapping_major_2_minor[cell_type]
    stats_minor = stats_minor[stats_minor['cell_type'].isin(subcelltypes)]
    stats_minor['type'] = stats_minor['cell_type']
    unique_subcelltypes = list(stats_minor['cell_type'].unique())

    df = pd.concat([df, stats_minor])
    # df['type'] = df['type'].astype(CategoricalDtype(categories=cats+unique_subcelltypes, ordered=True))

    df['cell_type_o'] = df['cell_type'].copy()
    df['cell_type'] = cell_type
    # df = add_centrality(df)
    df['cell_type'] = df['cell_type_o']

    return df 
def process_rr(df, keep_sig=True, keep_inconsistent=True):
    cols = ['tf', 'cell_type', 'meta_p_adj', 'trend']
    df = df[cols]
    df = df[~df.duplicated()]
    if keep_inconsistent:
        df = df[df['trend']!='Inconsistent']
    if keep_sig:
        df = df[df['meta_p_adj']<0.05]
    # df['trend'] = df['trend'].astype(CategoricalDtype(categories=['Increase in aging', 'Decrease in aging'], ordered=True))
    df['-log10_pval'] = -np.log10(df['meta_p_adj'])
    return df
def retreive_stats(cell_type, types=['bulk','std','M','F'], keep_sig=True, keep_inconsistent=True):
    stats_store = []
    for type in types:
        stats = pd.read_csv(f'../output/tf_activation/stats_{type}.csv')
        stats = stats[stats['cell_type'] == cell_type]
        stats = process_rr(stats, keep_sig, keep_inconsistent)
        print(type)
        stats['type'] = type
        stats_store.append(stats)

    df = pd.concat(stats_store).reset_index(drop=True)
    df['type'] = df['type'].astype(CategoricalDtype(categories=types, ordered=True))
    df = add_centrality(df)

    return df

def binarize_expression(cell_type, genes, dataset='data1'):
    adata = adata_lambda(dataset)
    adata = adata[adata.obs['cell_type'] == cell_type]
    adata = adata[:, adata.var_names.isin(genes)]
    # - binarize 
    if True: 
        expr = adata.to_df()
        expr = expr.merge(adata.obs[['age']], left_index=True, right_index=True, how='left').set_index('age')
        # expr = expr[expr.index>10]
        expr.sort_index(inplace=True)
        expr['age_bin'] = (expr.index.astype(int) // 5) * 5
        expr_mean = expr.groupby('age_bin').mean().T
    # Normalize expression
    min_vals = expr_mean.min(axis=1)
    max_vals = expr_mean.max(axis=1)
    expr_mean = (expr_mean.sub(min_vals, axis=0)).div(max_vals - min_vals, axis=0)
    return expr_mean
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


def find_central_tfs(net, par):
    centrality = net.groupby('source')['weight'].apply(lambda x: x.abs().sum())

    # Compute the 90th percentile threshold
    threshold = centrality.quantile(par['central_tf_q'])

    # Select TFs above the threshold
    top_genes = centrality[centrality >= threshold].index

    return top_genes


def add_root_sample(adata, seed=32):
    '''
    Add a root sample (one of younger age) to the adata object to use as the starting point for pseudotime analysis.
    '''
    root_cells = adata.obs.index[adata.obs['age'] < 30]
    
    if len(root_cells) > 0:
        np.random.seed(seed)
        root_cell = np.random.choice(root_cells, 1)[0]  # Pick a random root cell
        root_cell_idx = np.where(adata.obs.index == root_cell)[0][0]  # Get its index position
        # print(f"Using {root_cell} (index {root_cell_idx}) as the root for pseudotime.")
    else:
        raise ValueError("No samples found with age < 30 to use as root.")
    
    adata.uns['iroot'] = root_cell_idx  # Store the index position
    
def run_dpt(adata, n_neighbors=10, n_comps=10):
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
    sc.tl.diffmap(adata, n_comps=n_comps)
    sc.tl.dpt(adata)

# - pseudotime analysis
def run_pseudotime_analysis(adata, seed=32):
    # - add root age: #TODO: run this multiple times to choose different root cells 
    add_root_sample(adata, seed=seed)
    # - dpt 
    run_dpt(adata)
    if False:
        # - visualization
        sc.tl.umap(adata)
        sc.pl.umap(adata, color=['dpt_pseudotime', 'age'], cmap='viridis', show=True, size=3*(adata.obs['cell_count'] / adata.obs['cell_count'].max() * 100))
        sc.pl.pca(adata, color=['dpt_pseudotime', 'age'], cmap='viridis', show=True, size=3*(adata.obs['cell_count'] / adata.obs['cell_count'].max() * 100))

# def summary_stats(stats_df):
#     """
#     Check if the stats of tfs across different datasets and return the most conservative values."""
#     stats = []
#     for tf in stats_df['tf'].unique():
#         stats_tf = {'tf': tf}
#         df = stats_df[stats_df['tf'] == tf]
        
#         # - get the worst value across dataset: conservative approach 
#         for col in ['p_value_adj']:
#             print(df[col])
#             aa
#             if df[col].shape[0] < df['dataset'].nunique():
#                 worst_value = 1
#             else:
#                 worst_value = df[col].max()  # max for conservative worst value
#                 stats_tf.update({col: worst_value})
        
#         for col in ['slope', 'spearman_corr', 'r2']:
#             if col not in df.columns:
#                 raise ValueError(f'{col} not in df')
#             # - check if all slopes have same sign
#             if col in ['slope', 'spearman_corr']:
#                 if df[col].prod() < 0:
#                     # print(f'warning: inconsistent slopes: {tf}')
#                     stats_tf.update({col: None})
#                     continue
#             # Get the index of the worst value (min absolute value)
#             index_worst = df[col].abs().idxmin()
#             stats_tf.update({col: df[col].loc[index_worst]})
        
#         # Append the dictionary to the list
#         stats.append(stats_tf)
    
#     return pd.DataFrame(stats)
def identify_modules(df):
    import networkx as nx
    import igraph as ig
    import leidenalg as la
    weight_o = df['weight'].copy()
    df['weight'] = df['weight'].abs()
    G = nx.from_pandas_edgelist(df, 'source', 'target', ['weight'])

    # Convert to igraph (better for community detection)
    ig_graph = ig.Graph.TupleList(df.itertuples(index=False), directed=True, edge_attrs=["weight"])

    # Apply Leiden clustering
    partition = la.find_partition(ig_graph, la.RBConfigurationVertexPartition, weights="weight")

    # Extract module assignments
    df['module'] = [partition.membership[ig_graph.vs.find(name=n).index] for n in df['source']]
    df['weight'] = weight_o
    return df
from scipy.stats import spearmanr, linregress
from statsmodels.stats.multitest import multipletests
import pandas as pd

def linear_association_with_age(adata, genes, gene_col='tf'):
    '''
    Calculate p-values for the linear regression of the top tfs across datasets with ageing,
    and apply FDR correction (Benjamini-Hochberg).
    '''
    p_value_store = []

    for gene in genes:
        if gene not in adata.var_names:
            continue
        mask_gene = adata.var_names == gene
        adata_sub = adata[:, mask_gene]

        df = adata_sub.to_df()
        df = df.merge(adata_sub.obs[['age']], left_index=True, right_index=True)

        df.sort_values('age', inplace=True)

        ages = df['age'].values
        expression = df[gene].values

        # Fit linear regression
        if len(ages) > 1:
            spearman_corr, spearman_p = spearmanr(ages, expression)
            
            slope, intercept, r_value, p_value, _ = linregress(ages, expression)

            p_value_store.append({
                gene_col: gene, 
                'p_value': p_value,
                'slope': slope,
                'r2': r_value**2,
                'intercept': intercept,
                'spearman_corr': spearman_corr,
                'spearman_p': spearman_p
            })

    stats_df = pd.DataFrame(p_value_store)

    # # Apply FDR correction to p-values
    # if not stats_df.empty:
    #     stats_df['p_value_adj'] = multipletests(stats_df['p_value'], method='fdr_bh')[1]

    return stats_df
def summary_func(stats_df, col='tf'):
    assert stats_df.shape[0]>0
    def combine_pvalues(df):
        pvals = df["p_value_adj"]
        effect = df["slope"]
        if effect.prod() < 0: # if the effect is in opposite direction, pvalue is none
            return None
        if len(pvals)<2:
            return 1
        else:
            return max(pvals)
    meta_p_values = (
        stats_df.groupby(col)
        .apply(combine_pvalues)  
        .reset_index().rename(columns={0: "meta_p_value"})
        
    )
    if col not in stats_df.columns:
        raise ValueError(f"Column {col} is missing in stats_df")

    if col not in meta_p_values.columns:
        raise ValueError(f"Column {col} is missing in meta_p_values")
    stats_df = stats_df.merge(meta_p_values, on=col)

    return stats_df
        # combine_pvalues(pvals, method="fisher")[1]
def efficient_melting(df):

    # Assuming motif_scores_tf has motifs as index and regions as columns
    index = df.index.to_numpy()
    columns = df.columns.to_numpy()
    values = df.to_numpy()

    # Create a long-form DataFrame using NumPy broadcasting
    df_long = pd.DataFrame(
        {
            "index": np.repeat(index, len(columns)),
            "variable": np.tile(columns, len(index)),
            "values": values.ravel()
        }
    )
    return df_long


def compute_trend(df, pval_col='meta_p_adj', slope_col='slope', col='tf'):
    # Compute -log10(p_value_adj) for dot size
    df["neg_log10_adj_pval"] = -np.log10(df[pval_col])
    if 'trend' in df.columns:
        df.drop('trend', inplace=True, axis=1)
    # Determine color based on slope sign
    def determine_sign(x):
        if x.prod()<0:
            return "Inconsistent" 
        elif x.min() > 0:
            return 'Increase in aging'
        else:
            return 'Decrease in aging'
    trend = df.groupby([col, 'cell_type'])[slope_col].apply(determine_sign).reset_index(name='trend')
    trend = trend.dropna()
 
    df = df.merge(trend, on=[col, 'cell_type'], how='left')
    df["trend"] = pd.Categorical(
        df["trend"], 
        categories=['Increase in aging', 'Decrease in aging', "Inconsistent"], 
        ordered=True
    )
    return df

def stability_selection_booststrap(X, y, n_bootstrap=100, top_k=10):
    """
    Perform stability selection by bootstrapping and selecting important features.
    
    Parameters:
    - X: Feature matrix
    - y: Target vector
    - n_bootstrap: Number of bootstrap iterations.
    - top_k: Number of top features to select.
    
    Returns:
    - top_predictors: List of selected top-k most important features.
    """
    feature_coeffs = np.zeros((n_bootstrap, X.shape[1]))
    
    for i in range(n_bootstrap):
        # Bootstrap sampling
        X_resampled, y_resampled = resample(X, y, random_state=np.random.randint(0, 10000))
        
        # Scale data
        scaler = StandardScaler()
        X_resampled = scaler.fit_transform(X_resampled)

        # Fit Ridge regression
        model = Ridge(alpha=1)
        model.fit(X_resampled, y_resampled)
        
        # Store absolute coefficients
        feature_coeffs[i, :] = np.abs(model.coef_)

    # Compute median coefficient magnitude per feature
    feature_importance = np.median(feature_coeffs, axis=0)
    
    # Get indices of the top-k most important features
    top_features_idx = np.argsort(feature_importance)[-top_k:]
    
    return top_features_idx


def fit_final_model(X, y, top_features_idx):
    """
    Train a final model using only the selected most important features.
    
    Parameters:
    - X: Feature matrix
    - y: Target vector
    - top_features_idx: Indices of the top selected features.
    
    Returns:
    - model_final: Fitted Ridge regression model.
    - y_pred: Predictions of the final model.
    - r2: R² score of the final model.
    - spearman: Spearman correlation of the final model.
    """
    X_selected = X[:, top_features_idx]
    
    # Scale data
    scaler = StandardScaler()
    X_selected = scaler.fit_transform(X_selected)

    # Fit final model
    model_final = Ridge(alpha=1)
    model_final.fit(X_selected, y)
    
    # Predict and evaluate
    y_pred = model_final.predict(X_selected)
    r2 = r2_score(y, y_pred)
    spearman = spearmanr(y_pred, y)[0]
    
    return model_final, y_pred, r2, spearman

def stability_selection_shap(X, y,  top_q=80):
    """
    Perform stability selection using SHAP values for feature importance.

    Parameters:
    - X: Feature matrix
    - y: Target vector
    - n_bootstrap: Number of bootstrap iterations.
    - top_k: Number of top features to select.

    Returns:
    - top_predictors: List of selected top-q most important features.
    """
    
    # Scale data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    # Fit a model (e.g., RandomForest for SHAP)
    # model = RandomForestRegressor(n_estimators=100, random_state=42)
    model = Ridge(random_state=42)

    model.fit(X, y)

    # Compute SHAP values
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Compute mean absolute SHAP values for feature importance
    feature_importances = np.abs(shap_values.values).mean(axis=0)


    # Compute the q percentile threshold
    threshold = np.percentile(feature_importances, top_q)

    # Select features above the threshold
    top_features_idx = np.where(feature_importances >= threshold)[0]


    return top_features_idx
def find_robust_predictors(adata, target, top_q=.9):
    """
    Main function to perform stability selection, feature importance, and model evaluation.
    
    Parameters:
    - adata:  adata  
    - target: Column name for the target variable.
    - top_q: q of top features to select.
    """
    # Prepare data
    X = adata.X
    X = X.toarray() if scipy.sparse.issparse(X) else X
    y = adata.obs[target].values
    feature_names = adata.var_names

    # Stability selection
    top_features_idx = stability_selection_shap(X, y, top_q=top_q)
    top_predictors = feature_names[top_features_idx].values
    # print(f"Selected {len(top_predictors)} most important features.")

    # Fit final model with selected features
    model_final, y_pred, r2, spearman = fit_final_model(X, y, top_features_idx)

    # Print final model evaluation
    print(f"Final Model R²: {r2:.2f}, Spearman: {spearman:.2f}")
    # print("Most important predictors:", list(top_predictors))
    
    return list(top_predictors), r2
def add_centrality(df_all):
    """
        for a given df, compute the centrality of the TFs in the network, averaged over the two datasets, and subsetted to the TFs of interest
    """
    tfs = df_all['tf'].unique()
    cell_types = df_all['cell_type'].unique()
    assert 'centrality' not in df_all.columns, 'centrality already exists'
    consensus_nets_dict = get_consensus_network(cell_types, datasets)

    df_store = []
    for cell_type in cell_types:
        print(cell_type)
        df = df_all[df_all['cell_type'] == cell_type]
        assert df.shape[0]!=0, 'No cells'
        net = consensus_nets_dict[cell_type]
        c = net.groupby('source').size().reset_index(name='centrality')
        c = c[c['source'].isin(tfs)]
        assert c.isna().any().any() == False
        df = df.merge(c, left_on='tf', right_on='source', how='left')
        # assert np.setdiff1d(df['tf'].unique(), c['source'].unique()).size == 0, 'missing tfs' 
        df['centrality'] = df['centrality'].div(df['centrality'].max())
        df['cell_type'] = cell_type
        df_store.append(df)
    df = pd.concat(df_store).reset_index(drop=True)

    return df
def tf_activity_local(net, adata, tf_all=None):
    net = net.pivot(index='source', columns='target', values='weight').fillna(0)
    net = net[[g for g in adata.var_names if g in net.columns]]
    if tf_all is not None:
        tfs_present = np.intersect1d(net.index, tf_all)
    else:
        tfs_present = net.index
    net = net[net.index.isin(tfs_present)]
    # print('ratio of porosity: ', (net==0).sum().sum()/net.size)
    # - subset the adata
    adata = adata[:, adata.var_names.isin(net.columns)]
    # - enrich tfs 
    X = adata.X
    X = X.toarray() if scipy.sparse.issparse(X) else X
    mat = X.T
    
    tf_acts = np.dot(net, mat)
    # - format
    tf_acts = pd.DataFrame(tf_acts, index=net.index, columns=adata.obs.index)
    tf_acts = tf_acts.reset_index().melt(id_vars='source', var_name='sample', value_name='activity')

    # cols = [c for c in adata.obs.columns if c not in ['sample', 'cell_type', 'cell_count']]
    cols = adata.obs.columns

    tf_acts = tf_acts.set_index('sample').merge(adata.obs[cols], left_index=True, right_index=True).reset_index(drop=False)
    # print(f"net: {net.shape}, mat: {mat.shape}, source: {tf_acts['source'].nunique()}")
    
    if 'sample' not in tf_acts.columns:
        tf_acts['sample'] = tf_acts['index']
    
    # Calculate ranks within each sample
    # tf_acts['rank'] = tf_acts.groupby('sample')['activity'].transform(lambda x: x.abs().rank(method='dense', ascending=False)) 

    return tf_acts


def read_tf_acts(dataset, cell_type, type, read_dir='../output/tf_activation/tf_acts/'):
    return ad.read_h5ad(f'{read_dir}/{dataset}_{cell_type}_{type}.h5ad')
def write_tf_acts(adata, dataset, cell_type, type, save_dir='output/tf_activation/tf_acts/'):
    adata.write_h5ad(f'{save_dir}/{dataset}_{cell_type}_{type}.h5ad')

def calculate_tf_activity(adata, net, tf_all=None):    
    # - TFs
    if tf_all is not None:
        net = net[net['source'].isin(tf_all)]

    if False: # run decoupler
        mat = pd.DataFrame(
            data=adata.X.todense(),  
            columns=adata.var_names,  
            index=adata.obs.index  
        )

        tf_acts, tf_pvals = decoupler.run_ulm(mat, net, source='source', target='target', weight='weight', use_raw=False)
        # - formatize
        tf_acts = tf_acts.reset_index().melt(id_vars='index', var_name='source', value_name='activity')
        # cols = [c for c in adata.obs.columns if c not in ['sample', 'cell_type', 'cell_count']]
     
        # obs = adata.obs[cols]
        obs = adata.obs.copy()

        obs = obs.reset_index()
        
        tf_acts['index'] = tf_acts['index'].astype(str)
        obs['index'] = obs['index'].astype(str)
        tf_acts = tf_acts.merge(obs, on='index', how='left').drop('index', axis=1)
        assert tf_acts.shape[0]==tf_acts.shape[0]
    else: # run my implementation
        tf_acts = tf_activity_local(net, adata, tf_all)
    
    if 'index' in tf_acts.columns:
        tf_acts = tf_acts.drop('index', axis=1)
    
    assert tf_acts.shape[0] != 0, 'Empty'
    assert 'sample' in tf_acts.columns, 'sample not in tf_acts' 
    if False:
        tf_acts['sample'] = tf_acts['sample'].astype(int)
    tf_acts['age'] = pd.to_numeric(tf_acts['age'], errors='coerce')

    # - create anndata
    X_df = tf_acts.pivot(index='sample', columns='source', values='activity')
    obs_df = tf_acts.drop_duplicates(subset='sample').set_index('sample')[[c for c in tf_acts.columns if c not in ['source', 'activity', 'sample']]]
    obs_df = obs_df.loc[X_df.index]
    var_df = pd.DataFrame(index=X_df.columns)
    var_df['source'] = var_df.index
    tf_acts_adata = ad.AnnData(X=X_df.values, obs=obs_df, var=var_df)
    return tf_acts_adata

def pathway_analysis_wrapper(df):
    import gseapy as gp
    from gseapy import barplot, dotplot

    res2d_store = []
    for cell_type in df['cell_type'].unique():
    # for cell_type in ['MONO']:
        for trend in df['trend'].unique():
        # for trend in ['Decrease in aging']:
            mask = (df['cell_type'] == cell_type) & (df['trend'] == trend)
            if mask.sum() == 0:
                continue
            stats_df = df[mask]
            # - prepare
            stats_df = stats_df[['tf', 'meta_p_adj']]
            stats_df = stats_df[~stats_df.duplicated()].reset_index(drop=True)
            # ranked_genes = stats_df.set_index('tf')['meta_p_adj'].sort_values(ascending=True)

            # - EA
            genes = stats_df['tf'].unique()
            # genes = np.random.choice(df['tf'].unique(), len(genes))

            print(f"cell_type: {cell_type}, trend: {trend}, n tfs: {len(genes)}")
            np.savetxt('../output/test.csv', genes, fmt='%s', delimiter=',')
            rr = gp.enrichr(gene_list='../output/test.csv',
                            gene_sets=['MSigDB_Hallmark_2020'], #, 'KEGG_2021_Human'
                            organism='human', 
                            outdir=None, 
                            # background=tf_all,
                            )
            res2d = rr.res2d
            res2d = res2d[res2d['Adjusted P-value']<0.05]
            print(res2d.shape)
            res2d['cell_type'] = cell_type
            res2d['trend'] = trend
            res2d_store.append(res2d)
    if len(res2d_store) == 0:
        return None
    
    res2d_all = pd.concat(res2d_store)
    return res2d_all
