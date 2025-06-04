import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

from ciim.src.common import save_dir, surrogate_names, palette_datasets_pretty
from ciim.src.tf_activity.helper import retrieve_sig_stats
from ciim.src.tf_activity.helper import get_consensus_net
from ciim.src.common import datasets_all

def compute_tf_slopes(adata, sig_tfs):
    gene_names = adata.var_names.str.split('//').str[0]
    sig_tfs = [tf for tf in sig_tfs if tf in gene_names]
    mask_genes = gene_names.isin(sig_tfs)
    assert len(mask_genes) == adata.n_vars  # should now pass
    adata.var.index = adata.var.index.astype('category')

    adata_sig = adata[:, mask_genes].copy()
    adata_sig.X = adata_sig.X.toarray() if hasattr(adata_sig.X, 'toarray') else adata_sig.X
    X = pd.DataFrame(adata_sig.X, columns=adata_sig.var_names)
    ages = adata_sig.obs['age'].astype(float).values.reshape(-1, 1)
    slopes = {}
    for tf in adata_sig.var_names:
        tf_values = X[tf].values.reshape(-1, 1)
        if np.all(np.isnan(tf_values)) or np.all(tf_values == tf_values[0]):
            continue
        model = LinearRegression()
        model.fit(ages, tf_values)
        slopes[tf] = model.coef_.item()

    slope_df = pd.DataFrame.from_dict(slopes, orient='index', columns=['slope'])
    return slope_df
def perturb_tf_simulation(adata, net, tfs, slope_df, years=10):
    '''
     - 
    '''
    # print('Number of tfs to perturb:', len(tfs))

    # Extract gene names (first part before //)
    gene_names = adata.var_names.str.split('//').str[0]
    perturb_tfs = slope_df.index.str.split('//').str[0].values
    delta = slope_df.values.flatten()   # scale by years

    # Create expression DataFrame
    expr_df = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
                           columns=adata.var_names,
                           index=adata.obs_names)

    # Construct network matrix: rows = targets, columns = sources
    net_matrix_df = net.pivot(index='target', columns='source', values='weight').fillna(0)

    # Extend net_matrix_df to include all genes from adata.var_names (as targets)
    all_genes = expr_df.columns
    missing_genes = all_genes.difference(net_matrix_df.index)

    if len(missing_genes) > 0:
        zero_df = pd.DataFrame(
            0, index=missing_genes, columns=net_matrix_df.columns
        )
        net_matrix_df = pd.concat([net_matrix_df, zero_df], axis=0)
    # Reorder to match expr_df
    net_matrix_df = net_matrix_df.loc[expr_df.columns]
    

    # Final matrices
    N = net_matrix_df.values             # (num_targets = genes, num_sources = TFs)
    X0s = expr_df.values                 # (num_cells, num_targets = genes)

    # Build perturbation vector p aligned to net_matrix_df.columns (sources)
    tf_list = net_matrix_df.columns.str.split('//').str[0]
    p = np.zeros(len(net_matrix_df.columns))  # initialize

    tf_to_delta = dict(zip(perturb_tfs, delta))
    for i, tf in enumerate(tf_list):
        if tf in tf_to_delta:
            p[i] = tf_to_delta[tf]  # apply perturbation correct because the delta is for 10 years

    # Simulate per cell
    X_store = []
    for X0 in X0s:
        Xs = run_simulation(N, X0, p, n_iter=10)
        X = Xs[-1]
        X_store.append(X)

    # Create perturbed AnnData
    adata_perturb = adata.copy()
    adata_perturb.X = np.array(X_store)

    return adata_perturb

def run_simulation(N, X0, p, n_iter=10):
    X = X0.copy()
    X_store = [X0]
    for it in range(n_iter):
        X = X + np.dot(N, p)
        X_store.append(X)
    return np.array(X_store)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def run_simulation_nonlin(N, X0, p, n_iter=10):
    X = X0.copy()
    X_store = [X0]
    for _ in range(n_iter):
        delta = np.dot(N, p)          # shape: (1422,)
        delta = delta * X             # element-wise modulation by current gene state
        delta = np.tanh(delta)
        X = X + delta
        X_store.append(X)
    return np.array(X_store)
def perturb_tf_activity(adata, tfs, slope_df, years=10):
    # print('Number of tfs to perturb:', len(tfs))

    # Extract gene names (first part before //)
    gene_names = adata.var_names.str.split('//').str[0]
    slope_genes = slope_df.index.str.split('//').str[0]

    # Create a DataFrame from the expression matrix with var_names as columns
    expr_df = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
                           columns=adata.var_names,
                           index=adata.obs_names)

    # Iterate over transcription factors
    for tf in tfs:
        if tf not in gene_names.values or tf not in slope_genes.values:
            continue

        # Get the full var_name matching this TF in adata and slope_df
        tf_varnames = adata.var_names[gene_names == tf]
        tf_slope_names = slope_df.index[slope_genes == tf]

        for varname in tf_varnames.intersection(tf_slope_names):
            delta = slope_df.loc[varname].values
            if pd.isna(delta).any():
                continue
            expr_df[varname] += delta * years

    # Create a copy of the original AnnData and replace X
    adata_perturb = adata.copy()
    adata_perturb.X = expr_df.values

    return adata_perturb
# def experiment_perturb_tfs(dataset, cell_type, data_type, tfs, trends, n_donors=20, 
#                             reg_type='ridge', feature_type='tf_activity', ctr='Unperturbed', treatment='Perturbed'):
#     from ciim.src.clock.helper import prepare_input, predict_age
#     from ciim.src.common import save_dir
#     from ciim.src.common import save_dir
#     import anndata as ad
#     # - calculate slope
#     adata = ad.read_h5ad(f"{save_dir}/tf_activity_smoothed/{dataset}_{cell_type}_{data_type}.h5ad")
    
#     # - get the significant TFs and compute slopes
#     if tfs == 'aging_tfs': # perturb the sig tfs 
#         print('Perturbing the aging TFs')
#         stats_sig = retrieve_sig_stats(type='bulk', race='both', filter_inconsistent=True)
#         stats_sig = stats_sig[stats_sig['cell_type'] == cell_type]
#         tfs = stats_sig['tf'].unique()
#     elif tfs == 'all_tfs': # perturb all tfs
#         print('Perturbing all TFs')
#         tfs = adata.var_names
#     elif isinstance(tfs, list): # perturb the given tfs
#         print('Perturbing the given TFs')
#         tfs = [tf for tf in tfs if tf in adata.var_names]
#     else:
#         raise ValueError("perturb_coverage should be either 'aging_tfs' or 'all_tfs'")
    
    
#     if trends is None: # calculate the slope based on the activity slope
#         slope_df = compute_tf_slopes(adata.copy(), tfs) # slope is for one year
#         trend = 'aging'  # specify the trend for perturbation
#         if trend == 'aging':
#             slope_df = slope_df
#         elif trend == 'anti-aging':
#             slope_df = -slope_df
#         else:
#             raise ValueError("trend should be either 'increase' or 'decrease'")
#     else:  # fixed slope
#         slope_df = pd.DataFrame(trends)
#         slope_df.index = tfs
#         print('Slopes per year: ', slope_df)
#     # - perturb the TFs and create a new adata object
#     if feature_type == 'tf_activity': # here, we just change the TF activity based on the slope
#         adata_perturb = perturb_tf_activity(adata.copy(), tfs, slope_df)
#     elif feature_type == 'gene_expression': # here, we simulate the expression change in response to TF perturbation
#         from ciim.src.tf_activity.helper import get_consensus_net, retrieve_net
#         from ciim.src.common import datasets_all
#         adata = ad.read_h5ad(f"{save_dir}/gene_expression_smoothed/{dataset}_{cell_type}_{data_type}.h5ad")
#         # net = get_consensus_net(datasets=datasets_all, cell_type=cell_type, min_degree=3)
#         net = retrieve_net(dataset, cell_type)
#         adata_perturb = perturb_tf_simulation(adata.copy(), net, tfs, slope_df, years=10)
#     else:
#         raise ValueError("feature_type should be either 'tf_activity' or 'gene_expression'")
#     # - combine the adatas and predict age

#     adata.obs['condition'] = ctr
#     adata_perturb.obs['condition'] = treatment
#     adata_combined = ad.concat([adata, adata_perturb], axis=0)
#     adata_combined = predict_age(adata_combined, cell_type, feature_type=feature_type, data_type=data_type, reg_type=reg_type)
#     obs_combined = adata_combined.obs.copy()
#     # - subset and calculate age acceleration
#     if True:
#         obs_combined['donor_age'] = obs_combined['donor_id'].astype(str) + '_' + obs_combined['age'].astype(str)
#         donors = obs_combined['donor_age'].unique()
#         np.random.seed(0)
#         donors = np.random.choice(donors, n_donors, replace=False)
#         obs_combined = obs_combined[obs_combined['donor_age'].isin(donors)]
#     df_pivot = obs_combined.pivot(index='donor_age', columns='condition', values='predicted_age')
#     df_pivot = df_pivot.dropna(subset=[ctr, treatment])
#     df_pivot['diff'] = df_pivot[treatment] - df_pivot[ctr]
#     df_pivot['cell_type'] = cell_type
#     df_pivot['dataset'] = dataset

#     return df_pivot

def perform_stat_test(df_pivot, ctr='baseline', treatment='perturb'):
    
    t_stat, p_value = stats.ttest_rel(df_pivot[treatment], df_pivot[ctr])
    slope = (df_pivot[treatment] - df_pivot[ctr]).mean()
    return p_value, slope
def plot_age_acceleration_donors(df_pivot, ax=None, ctr='baseline', treatment='perturb'):
    if False: # line plot
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.scatterplot(data=df_pivot, x=ctr, y=treatment, alpha=0.7, ax=ax)
        min_age, max_age = df_pivot[ctr].min(), df_pivot[treatment].max()
        ax.plot([min_age, max_age], [min_age, max_age], color='gray', linestyle='--', label='Ideal')

    if True: # donor plot
        df_plot = df_pivot.reset_index().melt(id_vars='donor_age', 
                                            value_vars=[ctr, treatment],
                                            var_name='condition', 
                                            value_name='predicted_age')

        # plt.figure(figsize=(3, 2.5))
        if ax is None:
            fig, ax = plt.subplots(figsize=(2.5, 2))
        sns.lineplot(data=df_plot, x='condition', y='predicted_age', 
                    hue='donor_age', marker='o', alpha=0.6, legend=False, ax=ax)
 
        ax.margins(x=.1, y=0.1)

def wrapper_plot_age_acceleration_for_tf_perturbation(df_cell, top_n=30, features=None, value_col='signed_neg_log10_pval'):
    from ciim.src.common import save_dir, surrogate_names, palette_datasets_pretty
    
    # Median of absolute mean_diff per TF across datasets
    median_abs = df_cell.groupby('tf')[value_col].apply(lambda x: x.abs().median())
    top_tfs = median_abs.sort_values(ascending=False).head(top_n).index

    # Keep only top TFs
    df_cell = df_cell[df_cell['tf'].isin(top_tfs)].copy()

    # Sort TFs by signed mean_diff for plotting
    tf_order = df_cell.groupby('tf')[value_col].mean().sort_values().index
    df_cell['tf'] = pd.Categorical(df_cell['tf'], categories=tf_order, ordered=True)

    figsize = (3, 5)
    fig, ax = plt.subplots(figsize=figsize)
    df_cell['dataset'] = df_cell['dataset'].apply(lambda x: surrogate_names.get(x, x))
    # Background bars
    sns.barplot(
        data=df_cell,
        y='tf',
        x=value_col,
        color='lightgray',
        edgecolor='black',
        linewidth=0.1,  
        ci=None,
        ax=ax
    )

    # Dataset-colored points
    sns.stripplot(
        data=df_cell,
        y='tf',
        x=value_col,
        hue='dataset',
        palette=palette_datasets_pretty,
        dodge=True,
        alpha=0.8,
        size=5,
        jitter=False,
        orient='h',
        ax=ax
    )

    ax.axvline(0, color='gray', linestyle='--')
    ax.set_title(f'Top {top_n} TFs')
    ax.set_xlabel('Age acceleration significance')
    ax.set_ylabel('TF')
    ax.margins(y=.05)
    ax.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    return fig

from joblib import Parallel, delayed
import os
import anndata as ad
import pandas as pd
import numpy as np

# from ciim.src.insilico_perturbation.helper import compute_tf_slopes, perturb_tf_simulation
from ciim.src.tf_activity.helper import retrieve_sig_stats, retrieve_net
from ciim.src.clock.helper import predict_age
from ciim.src.common import save_dir

import warnings

# Ignore all warnings originating from the anndata module
warnings.filterwarnings("ignore", category=FutureWarning, module=r".*anndata.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r".*anndata.*")
warnings.filterwarnings("ignore", message=".*Observation names are not unique.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def get_perturbation_slopes(adata, all_tfs, mode='overexpression'):


    if mode == 'overexpression':
        direction = 1
        slope_df = pd.DataFrame({'slope': direction}, index=all_tfs)
    elif mode == 'natural_aging':  
        slope_df = compute_tf_slopes(adata, all_tfs)
        slope_df = slope_df.abs()  # take absolute values
    else:
        raise ValueError(f"Unknown perturbation mode: {mode}")
    
    return slope_df
def run_tf_screen_all(
        dataset,
        cell_type,
        tfs = None,
        data_type='bulk',
        years=10,
        version='v1',
        reg_type='ridge',
        feature_type='gene_expression',
        n_donors=20,
        perturbation_mode='natural_aging',
        perturbation_type='single'  # either 'single' or 'multi'
    ):
    print(f"Processing: {cell_type} - {dataset} ({perturbation_mode} | {perturbation_type})")
    
    net = get_consensus_net(datasets=datasets_all, cell_type=cell_type, min_degree=3)
    
    if tfs is None:
        if False:
            stats_sig = retrieve_sig_stats(type='bulk', race='both', filter_inconsistent=True)
            stats_sig = stats_sig[stats_sig['cell_type'] == cell_type]
            tfs = stats_sig['tf'].unique()
        else:
            tfs = net['source'].unique()

    adata = ad.read_h5ad(f"{save_dir}/gene_expression_smoothed/{dataset}_{cell_type}_{data_type}.h5ad")
    slope_df = get_perturbation_slopes(adata, tfs, mode=perturbation_mode)
    tfs = slope_df.index.tolist()

    def process_perturbation(adata_base, perturbed, tfs):
        adata_base.obs['condition'] = 'Baseline'
        perturbed.obs['condition'] = 'Perturbed'
        combined = ad.concat([adata_base, perturbed], axis=0)
        combined = predict_age(combined, cell_type, feature_type, data_type, reg_type, version)
        obs = combined.obs.copy()
        obs['donor_age'] = obs['donor_id'].astype(str) + '_' + obs['age'].astype(str)

        donors = obs['donor_age'].unique()
        if len(donors) < n_donors:
            return None

        donors = np.random.choice(donors, n_donors, replace=False)
        obs = obs[obs['donor_age'].isin(donors)]

        pivot = obs.pivot(index='donor_age', columns='condition', values='predicted_age')
        if 'Perturbed' not in pivot.columns or 'Baseline' not in pivot.columns:
            return None

        pivot['diff'] = pivot['Perturbed'] - pivot['Baseline']
        return {
            'mean_diff': pivot['diff'].mean(),
            'tf': ','.join(tfs),
            'cell_type': cell_type,
            'dataset': dataset,
            'perturbation': perturbation_mode,
            'perturbation_type': perturbation_type
        }

    
    if perturbation_type == 'multi':
        adata_perturb = perturb_tf_simulation(adata.copy(), net, list(tfs), slope_df.loc[tfs], years=years)
        result = process_perturbation(adata, adata_perturb, tfs)
        return pd.DataFrame([result]) 
        
    else:
        tf_results = []
        for tf in tfs:
            slope_tf = slope_df.loc[[tf]]
            adata_perturb = perturb_tf_simulation(adata.copy(), net, [tf], slope_tf, years=years)
            result = process_perturbation(adata, adata_perturb, [tf])
            tf_results.append(result)
        rr = pd.DataFrame(tf_results)

        return rr
def plot_age_acceleration(df_all):
    df_all['median'] = df_all['mean_diff'].median()
    fig, ax = plt.subplots(figsize=(3, 3))
    df_all['dataset'] = df_all['dataset'].apply(lambda name: surrogate_names.get(name, name))
    sns.stripplot(ax=ax, data=df_all, y='mean_diff', x='cell_type', hue='dataset', 
            palette=palette_datasets_pretty, alpha=0.7)
    sns.barplot(ax=ax, data=df_all, y='median', x='cell_type', alpha=0.5, color='gray')
    ax.legend(loc=(1.05, .2), frameon=False, title='Dataset')
    ax.set_ylabel("Age acceleration (years)")
    ax.set_xlabel("")
    return fig
def wrapper_run_tf_screening(par, cell_types, datasets, n_jobs=10):
    from ciim.src.common import save_dir
    # ---- Parallel Execution ----
    from joblib import Parallel, delayed

    os.makedirs(f"{save_dir}/perturbation", exist_ok=True)
    
    tasks = [
        delayed(run_tf_screen_all)(
            dataset, cell_type,
            **par
        )
        for cell_type in cell_types
        for dataset in datasets
    ]

    results = Parallel(n_jobs=n_jobs)(tasks)
    results = [res for res in results if res is not None and not res.empty]
    df_all = pd.concat(results, axis=0)
    from scipy.stats import norm

    df_all['z_score'] = df_all.groupby('cell_type')['mean_diff'].transform(
        lambda x: (x - x.mean()) / x.std()
    )

    df_all['empirical_pval_two_sided'] = 2 * norm.sf(np.abs(df_all['z_score']))
    df_all['signed_neg_log10_pval'] = -np.sign(df_all['mean_diff']) * np.log10(df_all['empirical_pval_two_sided'])
    return df_all