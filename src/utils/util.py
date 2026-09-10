
import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import scanpy as sc
from pathlib import Path
from scipy import stats
from hira.src.config import DATA_TYPES, MAJOR_CT_LABEL, SUB_CTS, DISCOVERY_COHORTS, mapping_minor_2_major, CONSENSUS_MIN_DEGREE, \
     SUB_CT_LABEL, get_config, PRIOR_DIR, DATA_DIR, GRNS_DIR, NET_WEIGHT_THRESHOLD, NET_MAX_SIZE, NET_SKELETON

# increase width of output display
pd.set_option('display.max_columns', None)

def coarsen(series):
    """Strip a trailing batch index, e.g. 'IN_NIB_B001'->'IN_NIB', 'Data1_6'->'Data1'."""
    return series.astype(str).str.replace(r'_[A-Za-z]*\d+$', '', regex=True)

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

def retrieve_adata(dataset, 
                   data_type='bulk', 
                   cell_type=None, 
                   age_limit=20, 
                   condition=None, 
                   only_net_genes=False,
                   only_sig_genes=False,
                   granularity=MAJOR_CT_LABEL,
                   mask_condition_col='condition', 
                   test_mode=False,
                   only_obs=False):    

    
    assert data_type in DATA_TYPES, f'Unknown type {data_type}'
    if test_mode:
        adata = ad.read_h5ad(f"{DATA_DIR}/{data_type}/zhang.h5ad", backed='r')
        adata.obs['dataset'] = dataset
    else:
        adata = ad.read_h5ad(f"{DATA_DIR}/{data_type}/{dataset}.h5ad", backed='r')

    obs = adata.obs.copy()
    if dataset not in ['ibd']:
        obs.rename({'perturbation': 'condition', 'disease': 'condition', 'treatment': 'condition', 'Max_WHO_Group': 'condition'}, axis=1, inplace=True)
    if 'condition' in obs.columns:
        config = get_config(dataset)
        name_mapping = config.name_mapping
        if name_mapping is not None:
                obs['condition'] = obs['condition'].map(lambda x: name_mapping.get(x, x))
        obs['condition'] = obs['condition'].astype(str)
    else:
        obs['condition'] = 'healthy'
    # make donor names pretties
    donors_all = sorted(obs['donor_id'].unique())
    donor_map = {d: f"Donor {i+1}" for i, d in enumerate(donors_all)}  # Pretty names
    obs['donor_id_old'] = obs['donor_id'].copy()
    obs['donor_id'] = obs['donor_id'].map(lambda x: donor_map.get(x, x))
    obs['donor_age'] = obs['donor_id'].astype(str) + '- age: ' + obs['age'].astype(str)

    cfg = get_config(dataset)
    bulk_group = cfg.bulk_group
    assert bulk_group is not None, f'Bulk grouping not defined for dataset {dataset}'
    obs['bulk_group'] = obs[bulk_group].astype(str).agg('_'.join, axis=1)

    gene_names = np.loadtxt(f'{PRIOR_DIR}/gene_names.txt', dtype=str)
    mask_genes = adata.var_names.isin(gene_names)
    if only_net_genes:
        print('Filtering to only genes in the GRN network...')
        net = retrieve_net_consensus(cell_type=cell_type)
        mask_genes &= adata.var_names.isin(net['target'].unique())
    if only_sig_genes:
        from hira.src.feature_association.helper import retrieve_sig_stats  # local: avoids circular import
        print('Filtering to only genes significantly associated with aging...')
        sig_genes = retrieve_sig_stats(analysis_name='ge_major_b', cell_type=cell_type)['gene'].unique()
        mask_genes &= adata.var_names.isin(sig_genes)
    # For datasets with pre-mapped labels, use them instead of CellTypist-assigned Major_CT
    # if dataset in ['soundlife', 'parsebioscience'] and 'Major_CT_original' in obs.columns:
    #     print('Remove meeee - using original major cell type labels for soundlife and parsebioscience')
    #     obs[MAJOR_CT_LABEL] = obs['Major_CT_original']

    # Cell type filtering
    if data_type in ['sc', 'bulk_minor']:
        obs = obs[obs[SUB_CT_LABEL].isin(SUB_CTS)].copy()
        obs[SUB_CT_LABEL] = pd.Categorical(obs[SUB_CT_LABEL], categories=SUB_CTS, ordered=True)
    
    mask = np.ones(len(obs), dtype=bool)
    if (cell_type is not None and cell_type != 'all'):
        if (granularity == 'Major_CT') & ('Major_CT' not in obs.columns) &  ('cell_type' in obs.columns):
            obs['Major_CT'] = obs['cell_type']
        if cell_type not in obs[granularity].unique():
            raise ValueError(f'Given cell type "{cell_type}" not in {obs[granularity].unique()} dataset: {dataset}')
        cell_type_mask = obs[granularity] == cell_type
        mask &= cell_type_mask.values


    if condition is not None:
        if isinstance(condition, str):
            condition = [condition]
        for c in condition:
            if c not in obs[mask_condition_col].unique():
                raise ValueError(f'Given condition "{c}" not in {obs[mask_condition_col].unique()} dataset: {dataset}')
        mask_condition = obs[mask_condition_col].isin(condition)
        mask &= mask_condition.values

    if test_mode: #filter by donors
        print('Get ride of meeeee')
        # select 10 young and 10 old
        obs['age'] = obs['age'].astype(float).astype(int)
        old_donors = obs[obs['age']>60]['donor_age'].unique()
        young_donors = obs[obs['age']<40]['donor_age'].unique()
        selected_donors = list(young_donors[:5]) + list(old_donors[:5])
        mask_donors = obs['donor_age'].isin(selected_donors)
        mask &= mask_donors.values
    
    # Apply mask to obs and apply all transformations
    obs = obs.iloc[np.where(mask)[0]].copy()
    obs['dataset'] = dataset
    
    # Age processing
    if 'age' in obs.columns:
        nan_age = obs['age'].isna()
        if nan_age.sum() > 0:
            print(f'Warning: {nan_age.sum()} cells with NaN age found. ')
            raise ValueError('Cells with NaN age found.')
        obs['age'] = obs['age'].astype(float).astype(int)
        obs = obs[obs['age'] >= age_limit].copy()  
    else:
        print('Warning: "age" column not found in obs. Setting to default age of 20.')
        obs['age'] = 20
    
    # Sex mapping
    if 'sex' in obs.columns:
        obs['sex'] = obs['sex'].apply(lambda name: {'F': 'Female', 'M':'Male'}.get(name, name))
    # Age group
    age_t = 50
    obs['age_group'] = obs['age'].apply(lambda x: 'Young' if x < age_t else 'Old')
    
    # If only_obs is True, return the processed obs dataframe without loading .X
    if only_obs:
        return obs
    
    # Otherwise, create AnnData object with .X
    if True: # experimental. new way to subset backed anndata
        obs_indices = obs.index
        adata_backed = adata  # Keep reference to backed version
        
        # Find positions in the original adata that match the filtered obs indices
        original_index = adata_backed.obs.index
        # Use index.get_indexer for efficient position lookup (requires unique index)
        if original_index.is_unique:
            obs_positions = original_index.get_indexer(obs_indices)
            if (obs_positions == -1).any():
                raise ValueError("Some obs indices not found in original adata")
        else:
            # Fall back to positional lookup when the index has duplicates
            index_to_pos = {idx: pos for pos, idx in enumerate(original_index)}
            obs_positions = np.array([index_to_pos[idx] for idx in obs_indices])
        
        var_indices = np.where(mask_genes)[0]
        X_subset = adata_backed.X[obs_positions, :][:, var_indices]
        var_subset = adata_backed.var.iloc[var_indices].copy()
        adata = ad.AnnData(
            X=X_subset,
            obs=obs,
            var=var_subset,
            uns=adata_backed.uns.copy() if hasattr(adata_backed, 'uns') else {},
        )
        for layer_name in adata_backed.layers.keys():
            adata.layers[layer_name] = adata_backed.layers[layer_name][obs_positions, :][:, var_indices]
    else:
        adata = adata[mask, mask_genes].to_memory()    
        obs = obs[obs.index.isin(adata.obs.index)].copy()  # Ensure obs is in the same order as adata
        for c in obs.columns:
            adata.obs[c] = obs[c]
    
    if ('lognorm' in adata.layers) | ('X_norm' in adata.layers):
        print(f'Using layer {("lognorm" if "lognorm" in adata.layers else "X_norm")}')
        adata.X = adata.layers['lognorm'] if 'lognorm' in adata.layers else adata.layers['X_norm']
    else:
        if data_type == 'sc':
            if 'log1p' in adata.uns:
                del adata.uns['log1p']
            one_value = adata.X.data[0]
            # res should be close enough
            res = np.isclose(one_value, np.int64(one_value))
            if not res:
                raise ValueError(f'Expected count data for sc dataset, but found non-integer value: {one_value}')
            adata.layers['counts'] = adata.X.copy()
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
    
    adata.uns['dataset'] = dataset

    if True:
        if data_type == 'sc':
            n_cell_t = 10
            # Use the granularity parameter passed to the function
            group = bulk_group + [granularity]
            adata.obs['group'] = adata.obs[group].astype(str).agg('_'.join, axis=1)
            sample_size = adata.obs.groupby('group', as_index=False).size()
            sample_size_p = sample_size[sample_size['size']>n_cell_t]
            mask = adata.obs['group'].isin(sample_size_p['group'])
            adata = adata[mask]
    mask_non = adata.obs[granularity].isna()
    adata = adata[~mask_non]

    # if (dataset == 'soundlife') & (data_type=='sc'):
    #     adata = adata[adata.obs['visitName']=='Flu Year 1 Day 0']

    return adata

def retrieve_net(dataset, cell_type, skeleton=NET_SKELETON, data_type='sc', grns_dir=GRNS_DIR, prior_dir=PRIOR_DIR):
    cell_type_major = mapping_minor_2_major.get(cell_type, cell_type)
    assert cell_type_major in ['CD4T', 'CD8T', 'NK', 'B', 'MONO', 'all'], f'Unknown cell type {cell_type_major}'
    assert skeleton in ('skeleton', 'promotor', None), f'Unknown skeleton {skeleton}'
    folder = f"{grns_dir}/{dataset}/{data_type}/"
    
    net = pd.read_csv(f"{folder}/net_{cell_type_major}.csv")
    gene_names = np.loadtxt(f'{prior_dir}/gene_names.txt', dtype=str)
    net = net[net['target'].isin(gene_names)]
    if NET_WEIGHT_THRESHOLD is not None:
        net = net[net['weight'] > NET_WEIGHT_THRESHOLD]
    if NET_MAX_SIZE is not None:
        net = net.sort_values(by='weight', ascending=False, key=abs).head(NET_MAX_SIZE)
    if skeleton is not None:
        net = net[net[f'{skeleton}_based']]
    return net[['source', 'target', 'weight', 'cell_type']]

# def retrieve_nets(datasets, cell_type, promotor_only=False):
#     net_store = []
#     for dataset in datasets:
#         net = retrieve_net(dataset=dataset, cell_type=cell_type, promotor_only=promotor_only)
#         net['dataset'] = dataset
#         net_store.append(net)
#     nets = pd.concat(net_store, ignore_index=True)
#     return nets

def retrieve_net_consensus(cell_type, datasets=DISCOVERY_COHORTS, min_degree=CONSENSUS_MIN_DEGREE, skeleton=NET_SKELETON, force=False, grns_dir=None):
    # One cache file per cell type. Its contents follow the config (NET_SKELETON,
    # NET_MAX_SIZE, ...) in force at build time -- rerun consensus_nets.py after
    # changing any of them.
    if grns_dir is None:
        grns_dir = GRNS_DIR
    save_name = f"{grns_dir}/consensus_net_{cell_type}.csv"
    if Path(save_name).exists() and not force:
        # print('Loading existing consensus GRN for', cell_type, 'with min degree', min_degree)
        net_mean = pd.read_csv(save_name)
        return net_mean
    print('Retrieving consensus GRN for', cell_type, 'with min degree', min_degree)
    from scipy.stats import zscore
    net_store = []
    for dataset in datasets:
        net = retrieve_net(dataset, cell_type, skeleton=skeleton, grns_dir=grns_dir)
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
    if False:
        # Compute z-score per link (within each link group)
        nets['zscore'] = nets.groupby('dataset')['weight'].transform(zscore)

        # Compute mean z-score across datasets per (source, target)
        net_mean = (
            nets.groupby(['source', 'target'])['zscore']
            .mean()
            .reset_index()
            .rename(columns={'zscore': 'weight'})
        )
    else:
        # just take the mean weight
        net_mean = (
            nets.groupby(['source', 'target'])['weight']
            .mean()
            .reset_index()
        )
    net_mean.to_csv(save_name, index=False)
    
    return net_mean


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
    curated_net.to_csv(f'{PRIOR_DIR}/collectri_with_source.csv', index=False)
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

def retrieve_sig_net(data_type='bulk', cell_type=None):
    df = pd.read_csv(f'{OUTPUT_DIR}/sig_nets/sig_nets_{type}.csv')
    if cell_type is not None:
        df = df[df['cell_type'] == cell_type]
    return df

# def determine_sig_network(data_type,  min_degree=3):
#     os.makedirs(f'{OUTPUT_DIR}/sig_nets', exist_ok=True)
#     stats_tfs = retrieve_sig_stats(data_type, feature_type='tf_activity')
#     stats_targets = retrieve_sig_stats(data_type, feature_type='gene_expression')
    
#     datasets = DISCOVERY_COHORTS
    

#     nets_stats_store = []
#     for cell_type in MAJOR_CTS:
#         stats_tfs_t = stats_tfs[stats_tfs['cell_type'] == cell_type].drop_duplicates(subset=['cell_type', 'gene'])[['gene', 'meta_p_adj', 'slope', 'trend']]
#         stats_targets_t = stats_targets[stats_targets['cell_type'] == cell_type].drop_duplicates(subset=['cell_type', 'target'])[['target', 'meta_p_adj', 'slope', 'trend']]
        
#         if len(stats_tfs_t) == 0:
#             print('No source for', cell_type, ' skipping it')
#             continue
#         if len(stats_targets_t) == 0:
#             print('No target for', cell_type, ' skipping it')
#             continue
        
#         # - get the nets
#         net = retrieve_net_consensus(datasets, cell_type=cell_type, min_degree=min_degree)
#         sig_tfs = stats_tfs_t['gene'].unique()
#         sig_targets = stats_targets_t['target'].unique()
#         net = net[(net['source'].isin(sig_tfs)) & (net['target'].isin(sig_targets))]
#         net = net.groupby(['source', 'target'])['weight'].mean().reset_index() # probably not necessary
#         # - get the stats
#         nets_stats = pd.merge(net, stats_tfs_t, left_on='source', right_on='gene', how='left')
#         nets_stats = pd.merge(nets_stats, stats_targets_t, left_on='target', right_on='target', how='left', suffixes=('_source', '_target'))
#         nets_stats = nets_stats[['source', 'target', 'weight', 'slope_source', 'slope_target', 'meta_p_adj_source', 'meta_p_adj_target', 'trend_source', 'trend_target']]
#         nets_stats['cell_type'] = cell_type
#         nets_stats_store.append(nets_stats)
#     nets_stats = pd.concat(nets_stats_store)

#     os.makedirs(f'{OUTPUT_DIR}/sig_nets', exist_ok=True)
#     nets_stats.to_csv(f'{OUTPUT_DIR}/sig_nets/sig_nets_{type}.csv')


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

# def basic_qc(adata, min_genes_per_cell = 200, max_genes_per_cell = 5000, min_cells_per_gene = 10):
#     mt = adata.var_names.str.startswith('MT-')
#     print('shape before ', adata.shape)
#     # 1. stats
#     total_counts = adata.X.sum(axis=1)
#     n_genes_by_counts = (adata.X > 0).sum(axis=1)
#     # mt_frac = adata[:, mt].X.sum(axis=1) / total_counts
    
#     low_gene_filter = (n_genes_by_counts < min_genes_per_cell)
#     high_gene_filter = (n_genes_by_counts > max_genes_per_cell)
#     # mt_filter = mt_frac > max_mt_frac

#     # 2. Filter cells
#     # print(f'Number of cells removed: below min gene {low_gene_filter.sum()}, exceed max gene {high_gene_filter.sum()}')
#     mask_cells=  (~low_gene_filter)& \
#                  (~high_gene_filter)
#                 #  (~mt_filter)
#     # 3. Filter genes
#     n_cells = (adata.X!=0).sum(axis=0)
#     mask_genes = n_cells>min_cells_per_gene
#     adata_f = adata[mask_cells, mask_genes]
#     print('shape after ', adata_f.shape)
#     return adata_f

import pandas as pd
from scipy.stats import hypergeom
import numpy as np


def test_mixed_effects(df, ctr, treatment, target_variable='predicted_age', config=None, group_key=None):
    import warnings
    warnings.filterwarnings("ignore")
    import statsmodels.formula.api as smf
    
    formula = config.mixed_effects_formula
    if group_key is None:
        group_key = config.mixed_effects_group
    
    # Determine the actual condition column name from config
    condition_col = config.condition_column if (config is not None and hasattr(config, 'condition_column')) else 'condition'
    
    # For interaction models (formula contains *), use ALL data without filtering
    df = df[df[condition_col].isin([ctr, treatment])].copy()
    
    n_groups = df[group_key].nunique()
    n_ctr = df[df[condition_col] == ctr].shape[0]
    n_treat = df[df[condition_col] == treatment].shape[0]
    
    if n_groups < 2:
        print(f"DIAGNOSTIC: Insufficient groups for {treatment} vs {ctr}: only {n_groups} {group_key}(s)")
        return (np.nan, np.nan) 
    
    if n_ctr < 2 or n_treat < 2:
        print(f"DIAGNOSTIC: Insufficient samples for {treatment} vs {ctr}: ctr={n_ctr}, treat={n_treat}")
        return (np.nan, np.nan)
    
    # Check variance
    var_ctr = df[df[condition_col] == ctr][target_variable].var()
    var_treat = df[df[condition_col] == treatment][target_variable].var()
    
    if var_ctr == 0 or var_treat == 0:
        print(f"DIAGNOSTIC: Zero variance for {treatment} vs {ctr}: var_ctr={var_ctr}, var_treat={var_treat}")
        return (np.nan, np.nan) 

    # Only convert if there are no NAs (conversion would fail otherwise)
    if hasattr(df[condition_col].dtype, 'name') and 'Int' in str(df[condition_col].dtype):
        if not df[condition_col].isna().any():
            df[condition_col] = df[condition_col].astype(int)
    
    # Convert other columns that might be nullable integers
    for col in df.columns:
        if hasattr(df[col].dtype, 'name') and 'Int' in str(df[col].dtype):
            if not df[col].isna().any():
                df[col] = df[col].astype(int)
    
    if df[condition_col].dtype == 'object' or df[condition_col].dtype.name == 'category':
        df[condition_col] = pd.Categorical(df[condition_col], categories=[ctr, treatment], ordered=True)
        df[condition_col] = df[condition_col].cat.codes  # 0 for ctr, 1 for treatment - ensures coefficient is (treatment - ctr)

    # replace feature_values with target_variable
    formula = formula.replace('feature_values', target_variable)
    # Fit mixed model
    try:
        model = smf.mixedlm(formula, df, groups=df[group_key])
        result = model.fit(method='bfgs', maxiter=100, warn_convergence=False)
    except Exception as e:
        print(f"DIAGNOSTIC: Model fitting failed with error: {str(e)}")
        print(formula, df[[group_key, condition_col, target_variable]])
        aaa
    
    # Extract p-value and coefficient for main condition effect
    # The coefficient name depends on the formula and encoding
    coef_names = result.params.index.tolist()
    
    # Try to find the condition effect coefficient
    # It could be named as the condition_col or C(condition_col)[T.treatment]
    condition_coef_name = None
    for name in coef_names:
        if condition_col in name and name != 'Intercept':
            condition_coef_name = name
            break
    
    if condition_coef_name is None:
        # Fallback: use the first non-intercept coefficient
        condition_coef_name = [n for n in coef_names if n != 'Intercept'][0]
    
    pval = result.pvalues[condition_coef_name]
    coef = result.params[condition_coef_name]
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


# HGNC gene groups 728 ("S ribosomal proteins") + 729 ("L ribosomal proteins"), fetched from
# https://www.genenames.org/cgi-bin/genegroup/download?id=728&type=branch (and id=729).
# A prefix regex like RP[SL] also catches RPS6KA1/3/4/5, RPS6KB1, RPS6KC1, RPS6KL1, RPS19BP1
# (S6-kinase family, not ribosomal proteins) -- use the curated symbol list instead.
RB_GENES = [
    'FAU', 'RPL10', 'RPL10A', 'RPL10L', 'RPL11', 'RPL12', 'RPL13', 'RPL13A', 'RPL14', 'RPL15',
    'RPL17', 'RPL18', 'RPL18A', 'RPL19', 'RPL21', 'RPL22', 'RPL22L1', 'RPL23', 'RPL23A', 'RPL24',
    'RPL26', 'RPL26L1', 'RPL27', 'RPL27A', 'RPL28', 'RPL29', 'RPL3', 'RPL30', 'RPL31', 'RPL32',
    'RPL34', 'RPL35', 'RPL35A', 'RPL36', 'RPL36A', 'RPL36AL', 'RPL37', 'RPL37A', 'RPL38', 'RPL39',
    'RPL39L', 'RPL3L', 'RPL4', 'RPL41', 'RPL5', 'RPL6', 'RPL7', 'RPL7A', 'RPL7L1', 'RPL8', 'RPL9',
    'RPLP0', 'RPLP1', 'RPLP2', 'RPS10', 'RPS11', 'RPS12', 'RPS13', 'RPS14', 'RPS15', 'RPS15A',
    'RPS16', 'RPS17', 'RPS18', 'RPS19', 'RPS2', 'RPS20', 'RPS21', 'RPS23', 'RPS24', 'RPS25',
    'RPS26', 'RPS27', 'RPS27A', 'RPS27L', 'RPS28', 'RPS29', 'RPS3', 'RPS3A', 'RPS4X', 'RPS4Y1',
    'RPS4Y2', 'RPS5', 'RPS6', 'RPS7', 'RPS8', 'RPS9', 'RPSA', 'UBA52',
]

def filter_rb_mt_genes(adata):
    """Drop mitochondrial (MT-*) and ribosomal protein (RB_GENES) genes."""
    drop = adata.var_names.str.startswith('MT-') | adata.var_names.isin(RB_GENES)
    print(f'Dropping {drop.sum()} RB/MT genes', flush=True)
    return adata[:, ~drop].copy()

def basic_qc(adata, min_genes_per_cell=200, max_genes_per_cell=5000, min_cells_per_gene=10):
    mt = adata.var_names.str.startswith("MT-")
    print("shape before ", adata.shape)
    total_counts = adata.X.sum(axis=1)
    n_genes_by_counts = (adata.X > 0).sum(axis=1)

    low_gene_filter = n_genes_by_counts < min_genes_per_cell
    high_gene_filter = n_genes_by_counts > max_genes_per_cell

    mask_cells = (~low_gene_filter) & (~high_gene_filter)
    n_cells = (adata.X != 0).sum(axis=0)
    mask_genes = n_cells > min_cells_per_gene
    adata_f = adata[mask_cells, mask_genes]
    print("shape after ", adata_f.shape)
    return adata_f


def read_gene_annotation(annotation_file):
    """Read a GTF gene annotation file and extract TSS per gene."""
    gtf_columns = [
        "chromosome", "source", "feature", "start", "end",
        "score", "strand", "frame", "attributes",
    ]
    gtf_df = pd.read_csv(
        annotation_file, sep="\t", comment="#", names=gtf_columns, low_memory=False
    )
    gtf_df["gene_name"] = gtf_df["attributes"].str.extract(r'gene_name "([^"]+)"')

    annotation_df = gtf_df[gtf_df["feature"] == "gene"][
        ["chromosome", "start", "end", "strand", "gene_name"]
    ]
    annotation_df["TSS"] = annotation_df.apply(
        lambda x: x["start"] if x["strand"] == "+" else x["end"], axis=1
    )
    return annotation_df[["chromosome", "start", "end", "TSS", "strand", "gene_name"]]


def sum_by(adata: ad.AnnData, col: str, unique_mapping: bool = True) -> ad.AnnData:
    """Sum `.X` entries per unique value of `col`.

    Adapted from https://discourse.scverse.org/t/group-sum-rows-based-on-jobs-feature/371/4
    """
    from scipy import sparse

    assert pd.api.types.is_categorical_dtype(adata.obs[col])
    cat = adata.obs[col].values

    indicator = sparse.coo_matrix(
        (np.broadcast_to(True, adata.n_obs), (cat.codes, np.arange(adata.n_obs))),
        shape=(len(cat.categories), adata.n_obs),
    )

    sum_adata = ad.AnnData(
        indicator @ adata.X,
        var=adata.var,
        obs=pd.DataFrame(index=cat.categories),
    )

    # copy over `.obs` values that have a one-to-one-mapping with `.obs[col]`
    obs_cols = list(set(adata.obs.columns) - set([col]))
    if unique_mapping:
        one_to_one_mapped_obs_cols = []
        nunique_in_col = adata.obs[col].nunique()
        for other_col in obs_cols:
            if len(adata.obs[[col, other_col]].drop_duplicates()) == nunique_in_col:
                one_to_one_mapped_obs_cols.append(other_col)
    else:
        one_to_one_mapped_obs_cols = obs_cols

    joining_df = (
        adata.obs[[col] + one_to_one_mapped_obs_cols].drop_duplicates().set_index(col)
    )
    assert (sum_adata.obs.index == sum_adata.obs.join(joining_df).index).all()
    sum_adata.obs = sum_adata.obs.join(joining_df)
    sum_adata.obs.index.name = col
    sum_adata.obs = sum_adata.obs.reset_index()
    sum_adata.obs.index = sum_adata.obs.index.astype("str")

    cell_count_df = adata.obs.groupby(col).size().reset_index(name='cell_count')
    sum_adata.obs = sum_adata.obs.merge(cell_count_df, on=col, how='left')
    return sum_adata


def bulkify_func(adata, cell_count_t=10, covariates=['cell_type', 'donor_id', 'age']):
    """Aggregate single-cell counts into pseudobulk per covariate group."""
    adata.obs['sum_by'] = ''
    for covariate in covariates:
        adata.obs['sum_by'] += '_' + adata.obs[covariate].astype(str)
    adata.obs['sum_by'] = adata.obs['sum_by'].astype('category')
    adata_bulk = sum_by(adata, 'sum_by', unique_mapping=True)
    cell_count_df = adata.obs.groupby('sum_by').size().reset_index(name='cell_count')
    if 'cell_count' in adata_bulk.obs:
        adata_bulk.obs.drop('cell_count', axis=1, inplace=True)
    adata_bulk.obs = adata_bulk.obs.merge(cell_count_df, on='sum_by')
    adata_bulk = adata_bulk[adata_bulk.obs['cell_count'] >= cell_count_t]
    return adata_bulk


def metacellify_func(adata, target_size=15, cell_count_t=5, n_pcs=15, random_state=0,
                      covariates=['cell_type', 'donor_id', 'age']):
    """Aggregate single-cell counts into metacells (~target_size cells each).

    Within each covariate group (e.g. donor x cell type), cells are split into
    sub-clusters of ~target_size cells via MiniBatchKMeans on a PCA embedding
    of the group's normalized expression, then raw counts are summed per
    sub-cluster. This controls metacell size directly (n_clusters = n_cells /
    target_size) rather than via Leiden resolution, which doesn't map
    predictably to cluster size.
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA
    import scipy.sparse as sp
    import gc

    adata.obs['group'] = ''
    for covariate in covariates:
        adata.obs['group'] += '_' + adata.obs[covariate].astype(str)

    metacell_id = np.empty(adata.n_obs, dtype=object)
    for group, idx in adata.obs.groupby('group').indices.items():
        n_cells = len(idx)
        n_clusters = max(1, round(n_cells / target_size))
        if n_clusters >= n_cells:
            # ponytail: too few cells to cluster meaningfully, one metacell per cell
            labels = np.arange(n_cells)
        else:
            X = adata.X[idx]
            if sp.issparse(X):
                X = X.toarray()
            # normalize_total + log1p are per-cell, so doing them on the group slice is
            # identical to normalizing the whole matrix, without the full-size copy
            totals = X.sum(axis=1, keepdims=True)
            totals[totals == 0] = 1
            X = np.log1p(X * (1e4 / totals))
            n_comp = min(n_pcs, X.shape[0] - 1, X.shape[1])
            if n_comp >= 2:
                X = PCA(n_components=n_comp, random_state=random_state).fit_transform(X)
            labels = MiniBatchKMeans(
                n_clusters=n_clusters, random_state=random_state, n_init=3
            ).fit_predict(X)
        metacell_id[idx] = [f'{group}_mc{l}' for l in labels]
    gc.collect()

    adata.obs['metacell_id'] = pd.Categorical(metacell_id)
    adata_mc = sum_by(adata, 'metacell_id', unique_mapping=True)
    cell_count_df = adata.obs.groupby('metacell_id').size().reset_index(name='cell_count')
    if 'cell_count' in adata_mc.obs:
        adata_mc.obs.drop('cell_count', axis=1, inplace=True)
    adata_mc.obs = adata_mc.obs.merge(cell_count_df, on='metacell_id')
    adata_mc = adata_mc[adata_mc.obs['cell_count'] >= cell_count_t]
    return adata_mc