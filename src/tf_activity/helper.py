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

tf_all = np.loadtxt(f"/home/jnourisa/projs/ongoing/task_grn_inference/resources/grn_benchmark/prior/tf_all.csv", dtype=str)

def net_lambda(dataset, cell_type):  
    net = pd.read_csv(f"/home/jnourisa/projs/ongoing/ciim/output/grns/{dataset}/net_{cell_type}_all_agegroups_all_batches.csv")
    return net.loc[net['source'].isin(tf_all)]
adata_lambda = lambda dataset: ad.read_h5ad(f"/vol/projects/jnourisa/datasets/{dataset}_bulk.h5ad") 
adata_sc_lambda = lambda dataset: ad.read_h5ad(f"/vol/projects/jnourisa/datasets/{dataset}_sc.h5ad") 
def adata_cell_type_lambda(dataset, cell_type):
    adata = adata_lambda(dataset)
    if cell_type=='T':
        adata_sub = adata[adata.obs['cell_type'].isin(['CD4T', 'CD8T'])]
        cols = adata_sub.obs.columns
        unique_cols = ['donor_id', 'age']
        other_cols = [col for col in cols if col not in unique_cols+['cell_type', 'cell_count', 'sum_by']]

        df = pd.DataFrame(
            adata_sub.X.toarray() if hasattr(adata_sub.X, "toarray") else adata_sub.X,  # Convert sparse to dense if needed
            columns=adata_sub.var_names,
            index=pd.MultiIndex.from_frame(adata_sub.obs[unique_cols])  # Keep MultiIndex
        ).reset_index()  # Convert MultiIndex back to normal columns
        df = df.groupby(unique_cols).mean().dropna()
        adata_mean = ad.AnnData(X=df.values, obs=df.reset_index()[unique_cols], var=pd.DataFrame(index=df.columns))
        obs = adata_mean.obs.copy()
        adata_mean.obs = obs.merge(adata_sub.obs[unique_cols+other_cols], on=unique_cols, how='left').drop_duplicates().reset_index(drop=True)
        adata_mean.obs['cell_count'] = 1000
        adata_mean.obs['cell_type'] = 'T'
        adata_sub = adata_mean
    else:
        adata_sub = adata[adata.obs['cell_type'] == cell_type]
    return adata_sub


def find_central_tfs(net, par):
    centrality = net.groupby('source')['weight'].apply(lambda x: x.abs().sum())

    # Compute the 90th percentile threshold
    threshold = centrality.quantile(par['central_tf_q'])

    # Select TFs above the threshold
    top_genes = centrality[centrality >= threshold].index

    return top_genes


def convert_long_table_2_adata(tf_acts, index_col=["age", "donor_id", "cell_count"]):
        # - convert long table to adata 
    tf_acts_table = tf_acts.pivot(index=index_col, columns='source', values='activity').fillna(0)
    tf_acts_table = tf_acts_table.reset_index()  
    tf_acts_adata = ad.AnnData(obs=tf_acts_table[index_col], X=tf_acts_table[[col for col in tf_acts_table.columns if col not in index_col]])
    tf_acts_adata.obs['age'] = pd.to_numeric(tf_acts_adata.obs['age'], errors='coerce')
    return tf_acts_adata

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
def determine_stats(adata, top_tfs):
    '''
    Calculate p values for the linear regression of the top tfs across datasets with ageing.
    '''
    p_value_store = []
    n_tests = len(top_tfs)
    for tf in top_tfs:
        mask_tf = adata.var_names == tf
        adata_sub = adata[:, mask_tf]

        ages = adata_sub.obs['age'].values
        expression = adata_sub.X.toarray().flatten()
        # cell_count = adata_sub.obs['cell_count'].values
        
        # cell_count_n = cell_count / max(cell_count)

        # Fit linear regression
        if len(ages) > 1:
            # Compute Spearman correlation
            spearman_corr, spearman_p = spearmanr(ages, expression)
            slope, intercept, r_value, p_value, _ = linregress(ages, expression)
            # - correct for multiple testing
            p_value_adj = p_value * n_tests
            p_value_adj = min([p_value_adj, 1])  # Ensure p-value is not greater than 1
            # print(f'{dataset}: {tf} - p-value: {p_value}, p-value_adj: {p_value_adj}')
            p_value_store.append({
                                'tf': tf, 
                                'p_value': p_value, 
                                'p_value_adj': p_value_adj,
                                'slope': slope,
                                'r2': r_value**2,
                                'intercept': intercept,
                                'spearman_corr': spearman_corr,
                                'spearman_p': spearman_p
                                })
    stats_df = pd.DataFrame(p_value_store)
    # stats_df_t = stats_df.pivot(index='dataset', columns='tf', values='p_value_adj')
    # stats_df_t = (stats_df_t<0.05).all(axis=0)
    # sig_ones = list(stats_df_t[stats_df_t].index.values)
    # stats_df = stats_df[stats_df['tf'].isin(sig_ones)]
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


def compute_linear_trend(df, pval_col='meta_p_adj', slope_col='slope', col='tf'):
    # Compute -log10(p_value_adj) for dot size
    df["neg_log10_adj_pval"] = -np.log10(df[pval_col])
    if 'linear_trend' in df.columns:
        df.drop('linear_trend', inplace=True, axis=1)
    # Determine color based on slope sign
    def determine_sign(x):
        if x.prod()<0:
            return "Inconsistent" 
        elif x.min() > 0:
            return "Increase"
        else:
            return "Decrease"
    linear_trend = df.groupby([col, 'cell_type'])[slope_col].apply(determine_sign).reset_index(name='linear_trend')
    linear_trend = linear_trend.dropna()
 
    df = df.merge(linear_trend, on=[col, 'cell_type'], how='left')
    df["linear_trend"] = pd.Categorical(
        df["linear_trend"], 
        categories=["Increase", "Decrease", "Inconsistent"], 
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
def find_robust_predictors(adata, covariates, target, top_q=.9):
    """
    Main function to perform stability selection, feature importance, and model evaluation.
    
    Parameters:
    - adata:  adata  
    - covariates: Columns to drop from the data for feature selection.
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
    
    return list(top_predictors)
def add_centrality(df, datasets):
    """
        for a given df, compute the centrality of the TFs in the network, averaged over the two datasets, and subsetted to the TFs of interest
    """
    tfs = df['tf'].unique()
    cell_type = df['cell_type'].unique()
    assert len(cell_type) == 1
    cell_type = cell_type[0]

    net1 = net_lambda(datasets[0], cell_type)
    net2 = net_lambda(datasets[1], cell_type)

    net1_c = determine_centrality(net1, use_weight=False)
    net2_c = determine_centrality(net2, use_weight=False)

    net1_c['dataset'] = datasets[0]
    net2_c['dataset'] = datasets[1]

    c = pd.concat([net1_c, net2_c], axis=0)
    c = c[c.index.isin(tfs)]
    assert c.isna().any().any() == False
    c = c.reset_index().rename(columns={'index': 'tf'})
    c = c.groupby(['tf'])['centrality'].mean().reset_index()
    df = df.merge(c, on='tf', how='left')
    df['centrality'] = df['centrality'].div(df['centrality'].max())

    return df
def tf_activity_local(net, adata_bulk, tf_all=None):
    net = net.pivot(index='source', columns='target', values='weight').fillna(0)
    net = net[[g for g in adata_bulk.var_names if g in net.columns]]
    if tf_all is not None:
        tfs_present = np.intersect1d(net.index, tf_all)
    else:
        tfs_present = net.index
    net = net[net.index.isin(tfs_present)]
    # print('ratio of porosity: ', (net==0).sum().sum()/net.size)
    # - subset the adata
    adata_bulk = adata_bulk[:, adata_bulk.var_names.isin(net.columns)]
    # - enrich tfs 
    X = adata_bulk.X
    X = X.toarray() if scipy.sparse.issparse(X) else X
    mat = X.T
    
    tf_acts = np.dot(net, mat)
    # - format
    tf_acts = pd.DataFrame(tf_acts, index=net.index, columns=adata_bulk.obs.index)
    tf_acts = tf_acts.reset_index().melt(id_vars='source', var_name='sample', value_name='activity')
    if 'cell_count' in adata_bulk.obs.columns:
        cols = ['cell_type', 'donor_id', 'cell_count', 'age']
    else:
        cols = ['cell_type', 'donor_id', 'age']
    tf_acts = tf_acts.set_index('sample').merge(adata_bulk.obs[cols], left_index=True, right_index=True).reset_index(drop=False)
    # print(f"net: {net.shape}, mat: {mat.shape}, source: {tf_acts['source'].nunique()}")
    
    if 'sample' not in tf_acts.columns:
        tf_acts['sample'] = tf_acts['index']
    # Calculate ranks within each sample
    # tf_acts['rank'] = tf_acts.groupby('sample')['activity'].transform(lambda x: x.abs().rank(method='dense', ascending=False)) 

    return tf_acts



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
        obs = adata.obs[['cell_type', 'donor_id', 'cell_count', 'age']]
        obs = obs.reset_index()
        
        tf_acts['index'] = tf_acts['index'].astype(str)
        obs['index'] = obs['index'].astype(str)
        tf_acts = tf_acts.merge(obs, on='index', how='left').drop('index', axis=1)
        assert tf_acts.shape[0]==tf_acts.shape[0]
    else: # run my implementation
        tf_acts = tf_activity_local(net, adata, tf_all)
    
    if 'index' in tf_acts.columns:
        tf_acts = tf_acts.drop('index', axis=1)
    if 'cell_count' in adata.obs.columns:
        cols = ["sample", "age", "donor_id", "cell_count"]
    else:
        cols = ["sample", "age", "donor_id"]

    tf_acts = convert_long_table_2_adata(tf_acts, index_col=cols)


    return tf_acts
def pathway_analysis_wrapper(df):
    import gseapy as gp
    from gseapy import barplot, dotplot

    res2d_store = []
    for cell_type in df['cell_type'].unique():
    # for cell_type in ['MONO']:
        for trend in df['linear_trend'].unique():
        # for trend in ['Decrease']:
            mask = (df['cell_type'] == cell_type) & (df['linear_trend'] == trend)
            if mask.sum() == 0:
                continue
            stats_df = df[mask]
            # - prepare
            stats_df = stats_df[['tf', 'meta_p_adj']]
            stats_df = stats_df[~stats_df.duplicated()].reset_index(drop=True)
            # ranked_genes = stats_df.set_index('tf')['meta_p_adj'].sort_values(ascending=True)

            # - EA
            print(f"cell_type: {cell_type}, trend: {trend}, n tfs: {stats_df['tf'].nunique()}")
            np.savetxt('../output/test.csv', stats_df['tf'].unique(), fmt='%s', delimiter=',')
            rr = gp.enrichr(gene_list='../output/test.csv',
                            gene_sets=['MSigDB_Hallmark_2020'], #, 'KEGG_2021_Human'
                            organism='human', 
                            outdir=None, 
                            # background=tf_all,
                            )
            res2d = rr.res2d
            res2d = res2d[res2d['Adjusted P-value']<0.05]

            res2d['cell_type'] = cell_type
            res2d['linear_trend'] = trend
            res2d_store.append(res2d)
    if len(res2d_store) == 0:
        return None
    
    res2d_all = pd.concat(res2d_store)
    return res2d_all
