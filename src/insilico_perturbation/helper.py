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
from ciim.src.utils.util import get_genesets
import warnings
from ciim.src.utils.util import calculate_genes_scores
warnings.filterwarnings('ignore')

def compute_slopes(adata, feature_type='gene_expression'):
    def get_linear_slope(ages, gene_values):
        model = LinearRegression()
        model.fit(ages, gene_values)
        return model.coef_.item()
    adata.X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    X = pd.DataFrame(adata.X, columns=adata.var_names)
    ages = adata.obs['age'].astype(float).values.reshape(-1, 1)
    slopes = {}
    if feature_type == 'gene_expression':
        for gene in adata.var_names:
            gene_values = X[gene].values.reshape(-1, 1)
            if np.all(np.isnan(gene_values)) or np.all(gene_values == gene_values[0]):
                continue
            slopes[gene] = get_linear_slope(ages, gene_values)
    elif feature_type == 'gene_score':
        pathways = get_genesets()
        for pathway, genes in pathways.items():
            if len(genes) == 0:
                continue
            try:
                adata = calculate_genes_scores(adata, genes, key='gene_score')
            except ValueError as e:
                continue
            gene_values = adata.obs['gene_score'].values.reshape(-1, 1)
            slopes[pathway] = get_linear_slope(ages, gene_values)
    slope_df = pd.DataFrame.from_dict(slopes, orient='index', columns=['slope'])
    
    return slope_df

def compute_tf_slopes(adata, tfs):
    '''
        Slope in natural aging
    '''
    gene_names = adata.var_names.str.split('//').str[0]
    tfs = [tf for tf in tfs if tf in gene_names]
    mask_genes = gene_names.isin(tfs)
    assert len(mask_genes) == adata.n_vars  # should now pass
    adata.var.index = adata.var.index.astype('category')

    adata_sig = adata[:, mask_genes].copy()
    slope_df = compute_slopes(adata_sig, feature_type='gene_expression')
    return slope_df
def perturb_tf_simulation(adata, net, tfs, slope_df, simulation_iteration=10):
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
def perturb_tf_activity(adata, tfs, slope_df, simulation_iteration=10):
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
            expr_df[varname] += delta * simulation_iteration

    # Create a copy of the original AnnData and replace X
    adata_perturb = adata.copy()
    adata_perturb.X = expr_df.values

    return adata_perturb

def perform_stat_test(df_pivot, ctr='baseline', treatment='perturb'):
    
    t_stat, p_value = stats.ttest_rel(df_pivot[treatment], df_pivot[ctr])
    slope = (df_pivot[treatment] - df_pivot[ctr]).mean()
    return p_value, slope

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
    return fig, top_tfs

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
        slope_df = slope_df*10
    else:
        raise ValueError(f"Unknown perturbation mode: {mode}")
    return slope_df
def compute_age_shift(adata_base, perturbed, **kwargs):
        adata_base.obs['condition'] = 'Baseline'
        perturbed.obs['condition'] = 'Perturbed'
        combined = ad.concat([adata_base, perturbed], axis=0)

        combined = predict_age(combined, **kwargs)
        obs = combined.obs.copy()
        obs['donor_age'] = obs['donor_id'].astype(str) + '_' + obs['age'].astype(str)

        donors = obs['donor_age'].unique()

        donors = np.random.choice(donors, len(donors), replace=False)
        obs = obs[obs['donor_age'].isin(donors)]

        pivot = obs.pivot(index='donor_age', columns='condition', values='predicted_age')
        if 'Perturbed' not in pivot.columns or 'Baseline' not in pivot.columns:
            return None
        
        pivot['age_shift'] = pivot['Perturbed'] - pivot['Baseline']

        return pivot
def compute_gene_score_shift(adata_base, adata_perturbed, genes):
    import scanpy as sc
    # Compute gene scores
    try:
        adata_base = calculate_genes_scores(adata_base, genes, key='gene_score_orig')
        adata_perturbed = calculate_genes_scores(adata_perturbed, genes, key='gene_score_perturbed')
    except ValueError as e:
        raise ValueError(f"Error calculating gene scores: {e}. Ensure that the genes are present in the dataset.")

    # Compute expression shift
    score_shift = adata_perturbed.obs['gene_score_perturbed'] - adata_base.obs['gene_score_orig']
    assert adata_base.obs['gene_score_orig'].isna().any()==False, "Baseline gene scores contain only zeros, cannot compute shift."
    score_shift_n = score_shift.abs()/adata_base.obs['gene_score_orig'].abs()
    
    adata_base.obs['donor_age'] = adata_base.obs['donor_id'].astype(str) + '_' + adata_base.obs['age'].astype(str)
    result_df = pd.DataFrame({
        'donor_age': adata_base.obs['donor_age'],
        'Baseline_gene_score': adata_base.obs['gene_score_orig'].values,
        'Perturbed_gene_score': adata_perturbed.obs['gene_score_perturbed'].values,
        'gene_score_shift': score_shift.values,
        'gene_score_shift_n': score_shift_n,
    }, index=adata_base.obs_names)
    result_df.set_index('donor_age', inplace=True)
    return result_df

def run_in_silico(
        dataset,
        cell_type,
        slope_df=None,
        tfs = None,
        data_type='bulk',
        simulation_iteration=10,
        version='v1',
        reg_type='ridge',
        feature_type='gene_expression',
        n_donors=20,
        perturbation_mode='natural_aging', # None, 'overexpression'
        perturbation_type='single'  # either 'single' or 'multi'
    ):
    print(f"Processing: {cell_type} - {dataset} ({perturbation_mode} | {perturbation_type})")
    
    net = get_consensus_net(datasets=datasets_all, cell_type=cell_type)
    # net = retrieve_net(dataset=dataset, cell_type=cell_type)
    if tfs is None:
        tfs = net['source'].unique()
    elif tfs=='aging':
        stats_aging = retrieve_sig_stats(type='bulk', race='both', filter_inconsistent=True)
        stats_aging = stats_aging[stats_aging['cell_type'] == cell_type]
        tfs = stats_aging['tf'].unique()
    else:
        pass

    adata = ad.read_h5ad(f"{save_dir}/gene_expression_smoothed/{dataset}_{cell_type}_{data_type}.h5ad")
    if 'disease' in adata.obs.columns:
        adata = adata[adata.obs['disease'] == 'normal'].copy()
    if slope_df is None:
        slope_df = get_perturbation_slopes(adata, tfs, mode=perturbation_mode)
    if 'cell_type' in slope_df.columns:
        slope_df = slope_df[slope_df['cell_type']==cell_type]
        slope_df = slope_df[['slope']]
    tfs = slope_df.index.tolist()

   
    if perturbation_type == 'multi':
        slope_df = slope_df.loc[tfs]

        adata_perturb = perturb_tf_simulation(adata.copy(), net, list(tfs), slope_df, simulation_iteration=simulation_iteration)
        result = compute_age_shift(adata, adata_perturb, 
                                    cell_type=cell_type, 
                                    data_type=data_type,
                                    version=version,
                                    reg_type=reg_type,
                                    feature_type=feature_type)
        
        result.columns = pd.MultiIndex.from_product([['age_shift'], result.columns])
        if True:
            pathways = get_genesets()
            for key, genes in pathways.items():
                try:
                    result_2 = compute_gene_score_shift(adata, adata_perturb, genes)
                except ValueError as e:
                    print(f"Error calculating gene scores for {key}: {e}")
                    continue
                result_2.columns = pd.MultiIndex.from_product([[key], result_2.columns])
                result = result.join(result_2, how='left')
        
        result['tf'] = ','.join(tfs)
        result['cell_type'] = cell_type
        result['dataset'] = dataset
        result['perturbation'] = perturbation_mode
        result['perturbation_type'] = perturbation_type

        return result
        
    else:
        slope_df = slope_df.abs()  # take absolute values
        tf_results = []
        for tf in tfs:
            slope_tf = slope_df.loc[[tf]]
            adata_perturb = perturb_tf_simulation(adata.copy(), net, [tf], slope_tf, simulation_iteration=simulation_iteration)
            result = compute_age_shift(adata, adata_perturb, 
                                    cell_type=cell_type, 
                                    data_type=data_type,
                                    version=version,
                                    reg_type=reg_type,
                                    feature_type=feature_type)
            result['tf'] = ','.join([tf])
            result['cell_type'] = cell_type
            result['dataset'] = dataset
            result['perturbation'] = perturbation_mode
            result['perturbation_type'] = perturbation_type
            tf_results.append(result)
        rr = pd.concat(tf_results).reset_index()
        
        return rr


def plot_age_acceleration(df_all, x_col='cell_type', log_y=False, margins=(0.1, 0.2)):
    # If a specific order is given, enforce it
    order = df_all[x_col].unique()
    print(df_all['cell_type'].unique())
    # Calculate median per category
    df_median = df_all.groupby(x_col)['mean_diff'].median().reset_index()

    # Plotting
    fig, ax = plt.subplots(figsize=(3, 3))
    df_all['dataset'] = df_all['dataset'].apply(lambda name: surrogate_names.get(name, name))
    sns.stripplot(ax=ax, data=df_all, y='mean_diff', x=x_col, hue='dataset',
                  palette=palette_datasets_pretty, alpha=0.7, order=order)
    sns.barplot(ax=ax, data=df_median, y='mean_diff', x=x_col, alpha=0.5, color='gray', order=order)

    # Axes and labels
    ax.legend(loc=(1.05, .2), frameon=False, title='Dataset')
    ax.margins(x=margins[0], y=margins[1])
    ax.set_ylabel("Age shift (years)")
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    if log_y:
        ax.set_yscale('symlog')

    return fig

def wrapper_in_silico_perturbation(par, cell_types, datasets, n_jobs=10):
    from ciim.src.common import save_dir
    # ---- Parallel Execution ----
    from joblib import Parallel, delayed

    os.makedirs(f"{save_dir}/perturbation", exist_ok=True)
    
    tasks = [
        delayed(run_in_silico)(
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

    # df_all['z_score'] = df_all.groupby('cell_type')['mean_diff'].transform(
    #     lambda x: (x - x.mean()) / x.std()
    # )

    # df_all['empirical_pval_two_sided'] = 2 * norm.sf(np.abs(df_all['z_score']))
    # df_all['signed_neg_log10_pval'] = -np.sign(df_all['mean_diff']) * np.log10(df_all['empirical_pval_two_sided'])
    return df_all