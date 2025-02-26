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

tf_all = np.loadtxt(f"/home/jnourisa/projs/ongoing/task_grn_inference/resources/grn_benchmark/prior/tf_all.csv", dtype=str)

def net_lambda(dataset, cell_type):  
    net = pd.read_csv(f"/home/jnourisa/projs/ongoing/ciim/output/grns/{dataset}/net_{cell_type}_all_agegroups_all_batches.csv")
    return net.loc[net['source'].isin(tf_all)]
adata_lambda = lambda dataset: ad.read_h5ad(f"/vol/projects/jnourisa/datasets/{dataset}_bulk.h5ad") 
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
        cell_count = adata_sub.obs['cell_count'].values
        
        cell_count_n = cell_count / max(cell_count)

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

def plot_trends(top_tfs, datasets, data_dict, datasets_colors, cell_type, stats_df=None, is_expression=False):
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
                legend_label = (f"{dataset}\n"
                                f"β={slope:.2f}, R²={r2:.2f}, p={p_value:.3g}, p_adj={p_value_adj:.3g}\n"
                                f"spear={spearman_corr:.2f}")
                handle = mpatches.Patch(color=datasets_colors[dataset], label=legend_label)
                legend_handles.append(handle)

        ax.set_xlabel('Age')
        ax.set_ylabel('Expression' if is_expression else 'Activity')
        ax.set_title(f'{cell_type}: {tf}', pad=15)

        # Add properly formatted legend
        ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.show()




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

def stability_selection_shap(X, y,  top_q=10):
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

def enrich_tf_local(net, adata_bulk, tf_all=None):
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
    tf_acts = tf_acts.set_index('sample').merge(adata_bulk.obs[['cell_type', 'donor_id', 'cell_count', 'age']], left_index=True, right_index=True).reset_index(drop=False)
    # print(f"net: {net.shape}, mat: {mat.shape}, source: {tf_acts['source'].nunique()}")
    
    if 'sample' not in tf_acts.columns:
        tf_acts['sample'] = tf_acts['index']
    # Calculate ranks within each sample
    tf_acts['rank'] = tf_acts.groupby('sample')['activity'].transform(lambda x: x.abs().rank(method='dense', ascending=False)) 

    return tf_acts



def enrich_tfs(adata_bulk, net, tf_all=None):
    # import decoupler
    # from scipy.stats import zscore

    # - pseudobulk cell type-donor
    # sys.path.insert(0, '../')
    # from task_grn_inference.src.process_data.perturbation.opsca.script import sum_by

    
    # -enrich TFs
    if tf_all is not None:
        net = net[net['source'].isin(tf_all)]

    if False: # run decoupler
        mat = pd.DataFrame(
            data=adata_bulk.X.todense(),  
            columns=adata_bulk.var_names,  
            index=adata_bulk.obs.index  
        )

        tf_acts, tf_pvals = decoupler.run_ulm(mat, net, source='source', target='target', weight='weight', use_raw=False)
        # - formatize
        tf_acts = tf_acts.reset_index().melt(id_vars='index', var_name='source', value_name='activity')
        obs = adata_bulk.obs[['cell_type', 'donor_id', 'cell_count', 'age']]
        obs = obs.reset_index()
        
        tf_acts['index'] = tf_acts['index'].astype(str)
        obs['index'] = obs['index'].astype(str)
        tf_acts = tf_acts.merge(obs, on='index', how='left').drop('index', axis=1)
        assert tf_acts.shape[0]==tf_acts.shape[0]
    else: # run my implementation
        tf_acts = enrich_tf_local(net, adata_bulk, tf_all)
    
    if 'index' in tf_acts.columns:
        tf_acts = tf_acts.drop('index', axis=1)
    tf_acts = convert_long_table_2_adata(tf_acts, index_col=["age", "donor_id", "cell_count"])


    return tf_acts
