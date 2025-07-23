import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

from ciim.src.common import save_dir, surrogate_names, palette_datasets_pretty
from ciim.src.feature_association.helper import retrieve_sig_stats
from ciim.src.feature_association.helper import get_consensus_net
from ciim.src.common import datasets_all
from ciim.src.utils.util import get_genesets
import warnings
from ciim.src.feature_association.helper import retrieve_feature_data 

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
def perturb_tf_simulation(adata, net, tfs, slope_df, simulation_iteration=3):
    '''
     - 
    '''

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
        Xs = run_simulation(N, X0, p, n_iter=simulation_iteration)
        X = Xs[-1]
        X = np.clip(X, 0, None)  # ensure non-negative expression
        X_store.append(X)

    # Create perturbed AnnData
    adata_perturb = adata.copy()
    adata_perturb.X = np.array(X_store)

    return adata_perturb


def run_simulation(N, X0, p, n_iter=3, decay=.7):
    X = X0.copy()
    X_store = [X0]
    signal = p.copy()
    for _ in range(n_iter):
        delta = np.dot(N, signal)
        X = X + np.tanh(delta)  # non-linear update
        X_store.append(X)
        signal *= decay
    return np.array(X_store)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def perturb_tf_activity(adata, tfs, slope_df, simulation_iteration=10):
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


from joblib import Parallel, delayed
import os
import anndata as ad
import pandas as pd
import numpy as np

from ciim.src.feature_association.helper import retrieve_sig_stats, retrieve_net
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

    adata_base.obs['donor_age'] = adata_base.obs['donor_id'].astype(str) + '_' + adata_base.obs['age'].astype(str)
    result_df = pd.DataFrame({
        'donor_age': adata_base.obs['donor_age'],
        'baseline_gene_score': adata_base.obs['gene_score_orig'].values,
        'perturbed_gene_score': adata_perturbed.obs['gene_score_perturbed'].values,
        'gene_score_shift': score_shift.values,
    }, index=adata_base.obs_names)
    result_df.set_index('donor_age', inplace=True)
    return result_df
import numpy as np
import pandas as pd

def compute_log2fc_genewise(adata_base, adata_perturbed):
    # Get normalized expression matrices
    X_base = adata_base.X
    X_perturbed = adata_perturbed.X

    # Convert to dense if sparse
    if not isinstance(X_base, np.ndarray):
        X_base = X_base.toarray()
    if not isinstance(X_perturbed, np.ndarray):
        X_perturbed = X_perturbed.toarray()

    # Log2 fold change per cell per gene (no averaging)
    log2fc = np.log2((X_perturbed + 1e-6) / (X_base + 1e-6))

    # Return as DataFrame: rows = obs names, columns = gene names
    return pd.DataFrame(log2fc, index=adata_perturbed.obs_names, columns=adata_perturbed.var_names)
def run_in_silico(
        dataset,
        cell_type,
        tfs,
        adata_dict, 
        net_dict,
        pathways, 
        slope_df=None,
        data_type='bulk',
        simulation_iteration=3,
        version='v1',
        reg_type='ridge',
        feature_type='gene_expression',
        n_donors=20,
        perturbation_mode='natural_aging', # None, 'overexpression'
        verbose=-1
    ):
    if verbose == 0:
        print(f"Processing: {cell_type} - {dataset} ({perturbation_mode})")
    # - prepare inputs
    adata = adata_dict[(dataset, cell_type)].copy()
    net = net_dict[cell_type]
    if slope_df is None:
        slope_df = get_perturbation_slopes(adata, tfs, mode=perturbation_mode)
    if 'cell_type' in slope_df.columns:
        slope_df = slope_df[slope_df['cell_type']==cell_type]
        slope_df = slope_df[['slope']]
    tfs = slope_df.index.tolist()

    adata_perturb = perturb_tf_simulation(adata.copy(), net, list(tfs), slope_df, simulation_iteration=simulation_iteration)
    result = compute_age_shift(adata, adata_perturb, 
                                cell_type=cell_type, 
                                data_type=data_type,
                                version=version,
                                reg_type=reg_type,
                                feature_type=feature_type)
    
    result.columns = pd.MultiIndex.from_product([['age_shift'], result.columns])
    if True:
        result_list = [result]
        for key, genes in pathways.items():
            try:
                result_2 = compute_gene_score_shift(adata, adata_perturb, genes)
            except ValueError as e:
                if verbose > 0:
                    print(f"Error calculating gene scores for {key}: {e}")
                continue
            result_2.columns = pd.MultiIndex.from_product([[key], result_2.columns])
            result_list.append(result_2)
        result = pd.concat(result_list, axis=1)
    
    result['tf'] = ','.join(tfs)
    result['cell_type'] = cell_type
    result['dataset'] = dataset
    result['perturbation'] = perturbation_mode

    log2f_gene_wise_df = compute_log2fc_genewise(adata, adata_perturb)
    print(log2f_gene_wise_df)
    aaa

    return result

def wrapper_in_silico_perturbation(par, cell_types, datasets, n_jobs=10):
    from ciim.src.common import save_dir
    # ---- Parallel Execution ----
    from joblib import Parallel, delayed

    os.makedirs(f"{save_dir}/perturbation", exist_ok=True)
    
    adata_dict = {
        (ds, ct): retrieve_feature_data(dataset=ds, cell_type=ct, smoothened=True, feature_type='gene_expression')
        for ct in cell_types
        for ds in datasets
    }
    net_dict = {
        ct: get_consensus_net(datasets=datasets_all, cell_type=ct)
        for ct in cell_types
    }
    pathways = get_genesets()

    tasks = [
        delayed(run_in_silico)(
            dataset=dataset, cell_type=cell_type, adata_dict=adata_dict, net_dict=net_dict, pathways=pathways,
            **par
        )
        for cell_type in cell_types
        for dataset in datasets
    ]

    results = Parallel(n_jobs=n_jobs)(tasks)
    results = [res for res in results if res is not None and not res.empty]
    df_all = pd.concat(results, axis=0)
    
    return df_all
def wrapper_in_silico_single_perturbation(tfs, par, cell_types, datasets, n_jobs=10):
    from ciim.src.common import save_dir
    # ---- Parallel Execution ----
    from joblib import Parallel, delayed

    os.makedirs(f"{save_dir}/perturbation", exist_ok=True)
    adata_dict = {
        (ds, ct): retrieve_feature_data(dataset=ds, cell_type=ct, smoothened=True, feature_type='gene_expression')
        for ct in cell_types
        for ds in datasets
    }
    net_dict = {
        ct: get_consensus_net(datasets=datasets_all, cell_type=ct)
        for ct in cell_types
    }
    pathways = get_genesets()

    tasks = [
        delayed(run_in_silico)(
            dataset=dataset, cell_type=cell_type, adata_dict=adata_dict, net_dict=net_dict, tfs=[tf], pathways=pathways,
            **par
        )
        for cell_type in cell_types
        for dataset in datasets for tf in tfs
    ]

    results = Parallel(n_jobs=n_jobs)(tasks)
    results = [res for res in results if res is not None and not res.empty]
    df_all = pd.concat(results, axis=0)
    
    return df_all
def identify_top_tfs(age_shift_mean_t, value_col='signed_neg_log10_pval', top_n=20):
    # Median of absolute mean_diff per TF across datasets
    median_abs = age_shift_mean_t.groupby('tf')[value_col].apply(lambda x: x.abs().median())
    top_tfs = median_abs.sort_values(ascending=False).head(top_n).index
    age_shift_mean_t = age_shift_mean_t[age_shift_mean_t['tf'].isin(top_tfs)]
    tf_order = age_shift_mean_t.groupby('tf')[value_col].mean().sort_values().index
    return tf_order
