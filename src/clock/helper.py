
import pandas as pd
import numpy as np
import anndata as ad
import numpy as np
from scipy import sparse
import pandas as pd
from anndata import AnnData
from hiara.src.utils.util import retrieve_adata
from hiara.src.config import (
    CLOCKS_DIR,
    use_local_clocks, 
    clock_version
)


# def evaluate_groupwise_median(obs):
#     import pandas as pd
#     from scipy.stats import spearmanr
#     from sklearn.metrics import r2_score
#     import numpy as np
#     df = obs.copy()
#     df['age'] = df['age'].astype(float)
#     df['donor_age'] = df['donor_age'].astype(str)
#     grouping_cols = ['donor_age', 'age']
#     predicted_age = df.groupby(grouping_cols)['predicted_age'].median().values
#     actual_age = df.groupby(grouping_cols)['age'].median().values

#     # print(np.isnan(actual_age).sum(), np.isnan(predicted_age).sum())
#     sp = spearmanr(actual_age, predicted_age)[0]
#     r2 = r2_score(actual_age, predicted_age)
    
#     scores = {
#         'Spearman': sp,
#         'R2': r2
#     }
#     return scores

# def _df_to_adata(df):
#     import anndata as ad
#     import pandas as pd
#     adata = ad.AnnData(
#         X=df.values,
#         obs=pd.DataFrame(index=df.index).reset_index(),
#         var=pd.DataFrame(index=df.columns)
#     )
#     return adata


# def format_data(datasets, cell_type=None, data_type='bulk', only_ctr=False, only_targets=False):
#     from hiara.src.utils.util import retrieve_adata, retrieve_net_consensus
#     import anndata as ad
    
#     adata_store = []
#     for d in datasets:
#         adata = retrieve_adata(dataset=d, type=data_type)
#         adata_store.append(adata)

#     # adata_train = ad.concat(adata_store, join='inner', axis=0)
#     adata_train = ad.concat(
#         adata_store,
#         axis=0,          # concatenate cells
#         join='outer',    # keep all obs columns (outer join on .obs)
#         merge='first'    # if duplicated obs keys, take the first
#     )

#     # Now restrict to common genes
#     common_genes = set.intersection(*(set(a.var_names) for a in adata_store))
#     adata_train = adata_train[:, list(common_genes)]
#     if cell_type is not None:
#         # net = retrieve_net_consensus(cell_type=cell_type)
#         # adata_train = adata_train[adata_train.obs['cell_type'] == cell_type, adata_train.var_names.isin(net['target'].unique())].copy()
#         adata_train = adata_train[adata_train.obs['cell_type'] == cell_type, :].copy()

#     adata_train.obs_names_make_unique()

#     if only_ctr:
#         adata_train = adata_train[adata_train.obs['is_control']]
    
#     print('Datasets in merged adata:', adata_train.obs['dataset'].unique())
#     # print('Conditions in merged adata:', adata_train.obs['condition'].unique())
#     print('Cell types in merged adata:', adata_train.obs['cell_type'].unique())

#     for col in adata_train.obs.columns:
#         if adata_train.obs[col].dtype == "object":
#             adata_train.obs[col] = adata_train.obs[col].astype(str)

#     if only_targets:
#         assert False, 'Fix me'
#         from hiara.src.config import DISCOVERY_COHORTS
#         assert cell_type is not None, "cell_type must be specified to filter for target genes"
#         # min_degree = min(len(aging_clock_train_datasets), 4)
#         net = retrieve_net_consensus(DISCOVERY_COHORTS, cell_type, min_degree=3)
#         target_genes = net['target'].unique()
#         adata_train = adata_train[:, adata_train.var_names.isin(target_genes)].copy()

#     return adata_train
# def align_feature_space(adata, gene_names):

#     var_names = np.array(adata.var.index.tolist())
#     var_index = {gene: i for i, gene in enumerate(var_names)}

#     # Collect indices or mark as -1 for missing
#     idxs = np.array([var_index.get(gene, -1) for gene in gene_names])

#     # Create a matrix with correct shape
#     rows = adata.obs.shape[0]
#     cols = len(gene_names)
#     X_aligned = sparse.lil_matrix((rows, cols))

#     # Fill in available gene columns
#     present = idxs != -1
#     if present.sum() > 0:
#         X_aligned[:, present] = adata[:].X[:, idxs[present]]

#     # Convert to CSR for efficiency
#     X_aligned = X_aligned.tocsr()

#     # Create new AnnData object
#     new_adata = AnnData(
#         X=X_aligned,
#         obs=adata.obs.copy(),
#         var={"gene_symbols": gene_names},
#     )
#     new_adata.var_names = gene_names

#     return new_adata

# def merge_adata(datasets, feature_type, cell_type, data_type, age_limit=0):
#     from hiara.src.config import OUTPUT_DIR
#     adata_store = []
#     for dataset in datasets:
#         # adata = prepare_input(dataset, cell_type, feature_type=feature_type, data_type=data_type)
#         adata = ad.read_h5ad(f"{OUTPUT_DIR}/{feature_type}_smoothed/{dataset}_{cell_type}_{data_type}.h5ad")
#         adata = adata[(adata.obs['age']>=age_limit)].copy()

#         adata = adata[adata.obs['is_control']]
#         adata_store.append(adata)
#     adata_all = ad.concat(adata_store, join='inner', axis=0)
#     print(adata_all.obs['dataset'].value_counts())
#     return adata_all
def wrapper_predict_age(adata, cell_type, use_local_clocks=use_local_clocks):
    import sys
    sys.path.insert(0, '../GRNimmuneClock')
    from grnimmuneclock import predict_age
    adata = predict_age(adata, cell_type=cell_type, use_local_clocks=use_local_clocks)
    return adata
def wrapper_clock_predictions(cell_types, evaluate_datasets, data_type='bulk', condition=None):
    obs_store = []
    for cell_type in cell_types:
        for dataset in evaluate_datasets:
            adata = retrieve_adata(dataset=dataset, data_type=data_type, cell_type=cell_type, only_net_genes=True, condition=condition)
            conds = adata.obs['condition'].unique()
            for cond in conds:
                adata_c = adata[adata.obs['condition'] == cond]
                wrapper_predict_age(adata=adata_c, cell_type=cell_type)
                obs = adata_c.obs
                obs['dataset'] = dataset
                obs['cell_type'] = cell_type
                obs['condition'] = cond
                obs_store.append(obs)
    obs = pd.concat(obs_store, axis=0)
    return obs

# def stability_selection_shap(features, model, X, y, top_q=80, top_features=50):
#     """
#     Perform stability selection using SHAP values for feature importance (linear model version).

#     Parameters:
#     - model: A sklearn pipeline with a StandardScaler and a linear model (e.g., Ridge).
#     - X: Feature matrix (DataFrame or ndarray)
#     - y: Target vector
#     - top_q: Percentile threshold to select top features

#     Returns:
#     - top_features_idx: Indices of selected top-q most important features
#     """
#     import shap
#     import numpy as np
#     import pandas as pd

#     # Fit the pipeline model
#     model.fit(X, y)

#     # Extract components from the pipeline
#     scaler = model.named_steps['standardscaler']
#     linear_model = model.named_steps['ridge']

#     # Apply the same transformation used in training
#     X_scaled = scaler.transform(X)

#     # Use SHAP's LinearExplainer for efficiency
#     explainer = shap.LinearExplainer(linear_model, X_scaled)
#     shap_values = explainer(X_scaled)

#     # Compute mean SHAP values for each feature
#     feature_importances = shap_values.values.mean(axis=0)

#     # Compute the q percentile threshold
#     if top_q is None:
#         top_features_idx = np.argsort(np.abs(feature_importances))[-top_features:]
#     else:
#         threshold = np.percentile(feature_importances, top_q)
#         # Select features above the threshold
#         top_features_idx = np.where(feature_importances >= threshold)[0]
#     feature_importances = feature_importances[top_features_idx]
#     features = features[top_features_idx]

#     rr = {'importance': feature_importances, 'feature': features}
     
#     return pd.DataFrame(rr)
