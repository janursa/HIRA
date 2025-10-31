
import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import sys
import os
import json
import scanpy as sc
import matplotlib.pyplot as plt
from collections import defaultdict
from ciim.src.common import base_dir, SAVE_DIR, PRIOR_DIR, datasets_all, mapping_minor_2_major

def read_gmt(file_path: str) -> dict[str, list[str]]:
    """Reas gmt file and returns a dict of gene"""
    gene_sets = {}
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split("\t")
            gene_set_name = parts[0]
            gene_set_description = parts[1]
            genes = parts[2:]
            gene_sets[gene_set_name] = {
                "description": gene_set_description,
                "genes": genes,
            }
    return gene_sets
def retrieve_adata(dataset, type='bulk', cell_type=None, age_limit=20): 
    base_path = f"{base_dir}/datasets/"
    if 'bulk' in type:
        base_path = f"{base_path}/bulk/"
    elif 'sc' in type:
        base_path = f"{base_path}/sc/"
       
    gene_names = np.loadtxt(f'{base_dir}/prior/gene_names.txt', dtype=str)

    assert type in ['sc', 'bulk', 'bulk_minor', 'bulk_M', 'bulk_F', 'bulk_minor_M', 'bulk_minor_F', 'metacell'], f'Unknown type {type}'
    
    if (type == 'bulk_M') | (type == 'bulk_F'):
        gender = type.split('_')[1]
        type = 'bulk'
        adata = ad.read_h5ad(f"{base_path}/{dataset}_{type}.h5ad")
        adata = adata[adata.obs['sex']==gender]
    elif (type == 'bulk_minor_M') | (type == 'bulk_minor_F'):
        gender = type.split('_')[-1]
        type = 'bulk_minor'
        adata = ad.read_h5ad(f"{base_path}/{dataset}_{type}.h5ad")
        adata = adata[adata.obs['sex']==gender]
    else:
        adata = ad.read_h5ad(f"{base_path}/{dataset}_{type}.h5ad")

    if 'age' not in adata.obs.columns:
        print('Warning: "age" column not found in adata.obs. Setting to default age of 20.')
        adata.obs['age'] = 20
    
    if ('lognorm' in adata.layers) | ('X_norm' in adata.layers):
        print(f'Using layer {("lognorm" if "lognorm" in adata.layers else "X_norm")}')
        adata.X = adata.layers['lognorm'] if 'lognorm' in adata.layers else adata.layers['X_norm']

    adata.obs['dataset'] = dataset
    adata = adata[:, adata.var_names.isin(gene_names)]
    # print('\n', adata.obs['cell_type'].unique())

    if cell_type is not None:
        if cell_type not in adata.obs['cell_type'].unique():
            raise ValueError(f'Given cell type "{cell_type}" not in {adata.obs["cell_type"].unique()}')
        adata = adata[adata.obs['cell_type'] == cell_type]
    if 'age' in adata.obs.columns:
        adata = adata[~adata.obs['age'].isna()].copy()
        adata.obs['age'] = adata.obs['age'].astype(float).astype(int)
        adata = adata[adata.obs['age'] >= age_limit].copy()  
    if 'sex' in adata.obs.columns:
        adata.obs['sex'] = adata.obs['sex'].apply(lambda name: {'F': 'Female', 'M':'Male'}.get(name, name))

    if dataset not in ['ibd']:
        adata.obs.rename({'perturbation': 'condition', 'disease': 'condition', 'treatment': 'condition', 'Max_WHO_Group': 'condition'}, axis=1, inplace=True)
    
    if 'condition' not in adata.obs.columns:
        adata.obs['condition'] = 'normal'

    unique_conditions = adata.obs['condition'].unique()

    ctr_key = None
    if 'normal' in unique_conditions:
        ctr_key = 'normal'
    elif 'healthy' in unique_conditions:
        ctr_key = 'healthy'
    elif 'PBS' in unique_conditions:
        ctr_key = 'PBS'
    elif 'Dimethyl Sulfoxide' in unique_conditions:
        ctr_key = 'Dimethyl Sulfoxide'
    elif '24 h RPMI' in unique_conditions:
        ctr_key = '24 h RPMI'
    elif dataset == 'ibd':
        ctr_key = None
    elif dataset == 'Covid_50MHH':
        ctr_key = None
    else:
        raise ValueError(f'No control condition found for {dataset}.')

    adata.obs.loc[:, 'is_control'] = adata.obs['condition'] == ctr_key
    

    adata.obs['donor_age'] = adata.obs['donor_id'].astype(str) + adata.obs['age'].astype(str)
        
    return adata

def retrieve_net(dataset, cell_type, only_promotor_based=False, c_t=5):  
    from ciim.src.common import SAVE_DIR
    cell_type_major = mapping_minor_2_major.get(cell_type, cell_type)
    assert cell_type_major in ['CD4T', 'CD8T', 'NK', 'B', 'MONO'], f'Unknown cell type {cell_type_major}'
    net = pd.read_csv(f"{SAVE_DIR}/grns/{dataset}/net_{cell_type_major}_all_agegroups_all_batches.csv")
    gene_names = np.loadtxt(f'{base_dir}/prior/gene_names.txt', dtype=str)
    net = net[net['target'].isin(gene_names)]
    if False:
        skeleton = pd.read_csv(f'{base_dir}/prior/skeleton_promotor.csv')
        net['edge'] = net['source'] + '_' + net['target']
        net = net[net['edge'].isin(skeleton['edge'])]
        net = net.drop('edge', axis=1)
    if only_promotor_based:
        net = net[net['promotor_based']]
    
    centrality_df = net.groupby(['source']).size()
    tfs = centrality_df[centrality_df>c_t].index
    net = net[net['source'].isin(tfs)]
    return net[['source', 'target', 'weight', 'cell_type']]

def retrieve_nets(datasets, cell_type, only_promotor_based=False):
    net_store = []
    for dataset in datasets:
        net = retrieve_net(dataset=dataset, cell_type=cell_type, only_promotor_based=only_promotor_based)
        net['dataset'] = dataset
        net_store.append(net)
    nets = pd.concat(net_store, ignore_index=True)
    return nets

def retrieve_net_consensus(datasets=datasets_all, cell_type='CD8T', min_degree=3, only_promotor_based=False):
    from scipy.stats import zscore
    net_store = []
    for dataset in datasets:
        net = retrieve_net(dataset, cell_type, only_promotor_based=only_promotor_based)
        net['dataset'] = dataset
        net_store.append(net)

    nets = pd.concat(net_store)

    nets['link'] = nets['source'] + '_' + nets['target']

    # Filter out inconsistent links (keep those with consistent sign)
    sign_info = nets.groupby('link')['weight'].apply(lambda x: set(np.sign(x)))
    consistent_links = sign_info[sign_info.apply(lambda x: len(x) == 1)].index
    nets = nets[nets['link'].isin(consistent_links)]

    # Filter links shared by at least min_degree datasets
    degrees = nets.groupby('link')['dataset'].nunique()
    shared_links = degrees[degrees >= min_degree].index
    nets = nets[nets['link'].isin(shared_links)]

    # Compute z-score per link (within each link group)
    nets['zscore'] = nets.groupby('dataset')['weight'].transform(zscore)

    # Compute mean z-score across datasets per (source, target)
    net_mean_z = (
        nets.groupby(['source', 'target'])['zscore']
        .mean()
        .reset_index()
        .rename(columns={'zscore': 'weight'})
    )
    
    return net_mean_z


def add_root_sample(adata):
    '''
    Add the root sample (the one with the minimum age) to the adata object for pseudotime analysis.
    '''
    if 'age' not in adata.obs.columns:
        raise ValueError("Column 'age' not found in adata.obs.")

    # Find the cell with the minimum age
    min_age = adata.obs['age'].min()
    root_cells = adata.obs.index[adata.obs['age'] == min_age]

    if len(root_cells) == 0:
        raise ValueError("No samples found to use as root.")

    # Choose the first one (or could sort and choose consistently)
    root_cell = root_cells[0]
    root_cell_idx = np.where(adata.obs.index == root_cell)[0][0]

    adata.uns['iroot'] = root_cell_idx
def run_dpt(adata, n_neighbors=10, n_comps=10):
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
    sc.tl.diffmap(adata, n_comps=n_comps)
    sc.tl.dpt(adata)
def flesh_out_collectri():
    import omnipath as op
    CollecTRI = op.interactions.CollecTRI.get(genesymbols=True, organism='human', loops=False)
    def extract_sources(source_str):
        # Split and clean each source tag
        parts = source_str.split(';')
        sources = []
        for s in parts:
            s = s.strip().replace('CollecTRI', '').replace('_', '').strip(';').strip()
            if s:
                sources.append(s)
        return sources

    # Expand the DataFrame
    expanded_rows = []
    for _, row in CollecTRI.iterrows():
        cleaned_sources = extract_sources(row['sources'])
        weight = 1 if row['is_stimulation'] else -1
        for ref in cleaned_sources:
            if ref in ['NTNU.Curated']:
                # Skip this source
                continue
            expanded_rows.append({
                'source': row['source_genesymbol'],
                'target': row['target_genesymbol'],
                'weight': weight,
                'ref': ref  # this is now a single cleaned source per row
            })

    # Create the new curated DataFrame
    curated_net = pd.DataFrame(expanded_rows)
    curated_net.to_csv(f'{base_dir}/prior/collectri_with_source.csv', index=False)
# - pseudotime analysis
def run_pseudotime_analysis(adata, seed=32):
    # - add root age: #TODO: run this multiple times to choose different root cells 
    add_root_sample(adata)
    # - dpt 
    run_dpt(adata)
    if True:
        # - visualization
        sc.tl.umap(adata)
        # sc.pl.umap(adata, color=['dpt_pseudotime', 'age'], cmap='viridis', show=True, size=3*(adata.obs['cell_count'] / adata.obs['cell_count'].max() * 100))
        # sc.pl.pca(adata, color=['dpt_pseudotime', 'age'], cmap='viridis', show=True, size=3*(adata.obs['cell_count'] / adata.obs['cell_count'].max() * 100))
    return adata

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
def find_robust_predictors(adata, target, top_q=90):
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

def basic_qc(adata, min_genes_per_cell = 200, max_genes_per_cell = 5000, min_cells_per_gene = 10):
    mt = adata.var_names.str.startswith('MT-')
    print('shape before ', adata.shape)
    # 1. stats
    total_counts = adata.X.sum(axis=1)
    n_genes_by_counts = (adata.X > 0).sum(axis=1)
    # mt_frac = adata[:, mt].X.sum(axis=1) / total_counts
    
    low_gene_filter = (n_genes_by_counts < min_genes_per_cell)
    high_gene_filter = (n_genes_by_counts > max_genes_per_cell)
    # mt_filter = mt_frac > max_mt_frac

    # 2. Filter cells
    # print(f'Number of cells removed: below min gene {low_gene_filter.sum()}, exceed max gene {high_gene_filter.sum()}')
    mask_cells=  (~low_gene_filter)& \
                 (~high_gene_filter)
                #  (~mt_filter)
    # 3. Filter genes
    n_cells = (adata.X!=0).sum(axis=0)
    mask_genes = n_cells>min_cells_per_gene
    adata_f = adata[mask_cells, mask_genes]
    print('shape after ', adata_f.shape)
    return adata_f

import pandas as pd
from scipy.stats import hypergeom
import numpy as np


def test_mixed_effects(dataset, df, ctr, treatment, target_variable='predicted_age'):
    import warnings
    warnings.filterwarnings("ignore")
    if dataset == 'op':
        fixed_effects=['condition']
        group_key='plate_name'
    elif dataset == 'parsebioscience':
        fixed_effects=['condition', 'cell_type_minor']
        group_key='donor_id'
    elif dataset == 'CXCL9':
        fixed_effects=['condition']
        group_key='donor_id'
    else:
        raise ValueError('Unknown dataset for mixed effects')
    import statsmodels.formula.api as smf

    df = df[df['condition'].isin([ctr, treatment])].copy()
    df['condition'] = pd.Categorical(df['condition'], categories=[ctr, treatment], ordered=True)
    df['condition'] = df['condition'].cat.codes  # 0 for ctr, 1 for treatment
    # df['donor_age'] = df['donor_age'].astype('category')
    # df['donor_age'] = df['donor_age'].cat.codes
    # Construct formula dynamically
    fixed_effects_s = ' + '.join(fixed_effects)
    formula = f"{target_variable} ~ {fixed_effects_s}"
    if group_key is None:
        model = smf.mixedlm(formula, df)
    else:
        model = smf.mixedlm(formula, df, groups=df[group_key])
    result = model.fit()
    pval = result.pvalues['condition']
    coef = result.params['condition']
    
    return pval, coef

def test_paired(df, ctr, treatment):
    import scipy.stats as stats
    df_pivot = df.pivot(index=['donor_age'], columns='condition', values='predicted_age').dropna()
    assert df_pivot.shape[1] == 2, f"Expected exactly two conditions, found {df_pivot.shape[1]}: {df_pivot.columns.tolist()}"
    
    t_stat, p_value = stats.ttest_rel(df_pivot[treatment], df_pivot[ctr])
    slope = (df_pivot[treatment] - df_pivot[ctr]).mean()
    return p_value, slope

def test_unpaired(df, ctr, treatment):
    # Extract samples
    ctr_samples = df[df['condition'] == ctr]['predicted_age'].dropna()
    treatment_samples = df[df['condition'] == treatment]['predicted_age'].dropna()
    
    # Check if we have enough samples
    if len(ctr_samples) < 2 or len(treatment_samples) < 2:
        raise ValueError(f"Not enough samples for {ctr} vs {treatment}. ")
    
    # Check for zero variance or non-numeric data
    if ctr_samples.var() == 0 or treatment_samples.var() == 0:
        raise ValueError(f"Zero variance in samples for {ctr} vs {treatment}. "
                         "Ensure there are multiple unique values in each group.")
    # Perform the t-test
    t_stat, p_value = stats.ttest_ind(treatment_samples, ctr_samples, equal_var=False)
    slope = treatment_samples.mean() - ctr_samples.mean()

    return p_value, slope