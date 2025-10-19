
from ciim.src.common import CLOCKS_DIR
import pandas as pd
import numpy as np
import anndata as ad
import joblib
import numpy as np
from scipy import sparse
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import issparse
from anndata import AnnData

def save_function(model, gene_names, cell_type, data_type, feature_type, reg_type, version):
    import os
    import joblib
    import numpy as np
    os.makedirs(CLOCKS_DIR, exist_ok=True)
    if reg_type == 'NN':
        import os
        import shutil
        model_dir = os.path.join(CLOCKS_DIR, f"{cell_type}_{data_type}_{feature_type}_{reg_type}_{version}")
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            shutil.rmtree(model_dir)
        model.save(model_dir, save_anndata=True)
    else:
        joblib.dump(model, os.path.join(CLOCKS_DIR, f"{cell_type}_{data_type}_{feature_type}_{reg_type}_{version}_model.pkl"))
    np.savetxt(f'{CLOCKS_DIR}/feature_names_{cell_type}_{data_type}_{feature_type}_{version}.txt', gene_names, fmt='%s')

def retrieve_function(reg_type, cell_type, data_type, feature_type, version):
    import os
    import joblib
    import numpy as np
    if reg_type == 'NN':
        from ciim.src.clock.NN.helper import save_path_train
        import cpa
        model = cpa.CPA.load(dir_path=save_path_train)
        gene_names = model.adata.var_names
    else:
        model = joblib.load(os.path.join(CLOCKS_DIR, f"{cell_type}_{data_type}_{feature_type}_{reg_type}_{version}_model.pkl"))
        gene_names = np.loadtxt(f'{CLOCKS_DIR}/feature_names_{cell_type}_{data_type}_{feature_type}_{version}.txt', dtype=str)
    return model, gene_names

def evaluate_groupwise_median(obs):
    import pandas as pd
    from scipy.stats import spearmanr
    from sklearn.metrics import r2_score
    import numpy as np
    df = obs.copy()
    df['age'] = df['age'].astype(float)
    df['donor_age'] = df['donor_age'].astype(str)
    grouping_cols = ['donor_age', 'age']
    predicted_age = df.groupby(grouping_cols)['predicted_age'].median().values
    actual_age = df.groupby(grouping_cols)['age'].median().values

    # print(np.isnan(actual_age).sum(), np.isnan(predicted_age).sum())
    sp = spearmanr(actual_age, predicted_age)[0]
    r2 = r2_score(actual_age, predicted_age)
    
    scores = {
        'Spearman': sp,
        'R2': r2
    }
    return scores

def _df_to_adata(df):
    import anndata as ad
    import pandas as pd
    adata = ad.AnnData(
        X=df.values,
        obs=pd.DataFrame(index=df.index).reset_index(),
        var=pd.DataFrame(index=df.columns)
    )
    return adata
# def prepare_input(dataset, cell_type, feature_type='tf_activity', data_type='bulk'):
#     '''
#     For tf_activity, it gets the consensus net and calculates the tf activity'''
#     from ciim.src.feature_association.helper import calculate_tf_activity
#     from ciim.src.utils.util import retrieve_adata, retrieve_net_consensus
#     from ciim.src.common import mapping_minor_2_major, datasets_all, minor_cell_types
#     from scipy.sparse import issparse

#     cell_type_major = mapping_minor_2_major.get(cell_type, cell_type)
#     net = retrieve_net_consensus(datasets_all, cell_type_major)
    
#     adata = retrieve_adata(dataset=dataset, cell_type=cell_type_major, type=data_type)
    
#     if feature_type == 'tf_activity':
#         adata = calculate_tf_activity(adata, net)
#     elif feature_type == 'gene_expression':
#         pass #TODO: here the feature space is set to target gene
#         # adata = adata[:, adata.var_names.isin(net['target'].unique())].copy()
#     else:
#         raise ValueError('Unknown feature type')
#     adata.obs['donor_age'] = adata.obs['donor_id'].astype(str) + '_' + adata.obs['age'].astype(str)
#     print(dataset, adata.X.shape)
#     if 'minor' in data_type:
#         adata = adata[adata.obs['Sub_CT'].isin(minor_cell_types)].copy()
#         adata = pivot_adata_minor(adata)
#         print('pivoted', adata.X.shape)
#     try:
#         adata.X = adata.layers['X_norm']
#     except:
#         pass
#     return adata

def format_data(datasets, cell_type=None, data_type='bulk', only_ctr=False, only_targets=False):
    from ciim.src.utils.util import retrieve_adata, retrieve_net_consensus
    import anndata as ad
    
    adata_store = []
    for d in datasets:
        adata = retrieve_adata(dataset=d, type=data_type)
        adata_store.append(adata)

    # adata_train = ad.concat(adata_store, join='inner', axis=0)
    adata_train = ad.concat(
        adata_store,
        axis=0,          # concatenate cells
        join='outer',    # keep all obs columns (outer join on .obs)
        merge='first'    # if duplicated obs keys, take the first
    )

    # Now restrict to common genes
    common_genes = set.intersection(*(set(a.var_names) for a in adata_store))
    adata_train = adata_train[:, list(common_genes)]
    if cell_type is not None:
        # net = retrieve_net_consensus(cell_type=cell_type)
        # adata_train = adata_train[adata_train.obs['cell_type'] == cell_type, adata_train.var_names.isin(net['target'].unique())].copy()
        adata_train = adata_train[adata_train.obs['cell_type'] == cell_type, :].copy()

    adata_train.obs_names_make_unique()

    if only_ctr:
        adata_train = adata_train[adata_train.obs['is_control']]
    
    print('Datasets in merged adata:', adata_train.obs['dataset'].unique())
    # print('Conditions in merged adata:', adata_train.obs['condition'].unique())
    print('Cell types in merged adata:', adata_train.obs['cell_type'].unique())

    for col in adata_train.obs.columns:
        if adata_train.obs[col].dtype == "object":
            adata_train.obs[col] = adata_train.obs[col].astype(str)

    if only_targets:
        from ciim.src.common import datasets_all
        assert cell_type is not None, "cell_type must be specified to filter for target genes"
        # min_degree = min(len(aging_clock_train_datasets), 4)
        net = retrieve_net_consensus(datasets_all, cell_type, min_degree=3)
        target_genes = net['target'].unique()
        adata_train = adata_train[:, adata_train.var_names.isin(target_genes)].copy()

    return adata_train
def align_feature_space(adata, gene_names):

    var_names = np.array(adata.var.index.tolist())
    var_index = {gene: i for i, gene in enumerate(var_names)}

    # Collect indices or mark as -1 for missing
    idxs = np.array([var_index.get(gene, -1) for gene in gene_names])

    # Create a matrix with correct shape
    rows = adata.obs.shape[0]
    cols = len(gene_names)
    X_aligned = sparse.lil_matrix((rows, cols))

    # Fill in available gene columns
    present = idxs != -1
    if present.sum() > 0:
        X_aligned[:, present] = adata[:].X[:, idxs[present]]

    # Convert to CSR for efficiency
    X_aligned = X_aligned.tocsr()

    # Create new AnnData object
    new_adata = AnnData(
        X=X_aligned,
        obs=adata.obs.copy(),
        var={"gene_symbols": gene_names},
    )
    new_adata.var_names = gene_names

    return new_adata
def predict_age(adata, cell_type, feature_type='tf_activity', data_type='bulk', reg_type='ridge', version='v1.0'):
    # try:
    #     adata.X = adata.layers['X_norm'].copy()  # Ensure we use the normalized data
    # except:
    #     pass
    
    # - load the model
    model, gene_names = retrieve_function(reg_type, cell_type, data_type, feature_type, version)
    
    # - align the genes
    adata = align_feature_space(adata, gene_names)
    if reg_type == 'NN':
        import cpa 
        from ciim.src.clock.NN.helper import cell_type_train
        cpa.CPA.wrapper_setup_data(adata, data_type=data_type, batch_key='dataset', cell_type=cell_type_train)
        model.predict_age(adata)
        adata.obs.rename(columns={'age_predicted': 'predicted_age'}, inplace=True)
    else:
        X = adata.X.copy()
        if issparse(X):
            X = X.toarray()
        predicted_age = model.predict(X)
        adata.obs['predicted_age'] = predicted_age.copy()
    return adata
def merge_adata(datasets, feature_type, cell_type, data_type, age_limit=0):
    from ciim.src.common import SAVE_DIR
    adata_store = []
    for dataset in datasets:
        # adata = prepare_input(dataset, cell_type, feature_type=feature_type, data_type=data_type)
        adata = ad.read_h5ad(f"{SAVE_DIR}/{feature_type}_smoothed/{dataset}_{cell_type}_{data_type}.h5ad")
        adata = adata[(adata.obs['age']>=age_limit)].copy()

        adata = adata[adata.obs['is_control']]
        adata_store.append(adata)
    adata_all = ad.concat(adata_store, join='inner', axis=0)
    print(adata_all.obs['dataset'].value_counts())
    return adata_all

def stability_selection_shap(features, model, X, y, top_q=80, top_features=50):
    """
    Perform stability selection using SHAP values for feature importance (linear model version).

    Parameters:
    - model: A sklearn pipeline with a StandardScaler and a linear model (e.g., Ridge).
    - X: Feature matrix (DataFrame or ndarray)
    - y: Target vector
    - top_q: Percentile threshold to select top features

    Returns:
    - top_features_idx: Indices of selected top-q most important features
    """
    import shap
    import numpy as np
    import pandas as pd

    # Fit the pipeline model
    model.fit(X, y)

    # Extract components from the pipeline
    scaler = model.named_steps['standardscaler']
    linear_model = model.named_steps['ridge']

    # Apply the same transformation used in training
    X_scaled = scaler.transform(X)

    # Use SHAP's LinearExplainer for efficiency
    explainer = shap.LinearExplainer(linear_model, X_scaled)
    shap_values = explainer(X_scaled)

    # Compute mean SHAP values for each feature
    feature_importances = shap_values.values.mean(axis=0)

    # Compute the q percentile threshold
    if top_q is None:
        top_features_idx = np.argsort(np.abs(feature_importances))[-top_features:]
    else:
        threshold = np.percentile(feature_importances, top_q)
        # Select features above the threshold
        top_features_idx = np.where(feature_importances >= threshold)[0]
    feature_importances = feature_importances[top_features_idx]
    features = features[top_features_idx]

    rr = {'importance': feature_importances, 'feature': features}
     
    return pd.DataFrame(rr)
