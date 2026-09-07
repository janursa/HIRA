
import pandas as pd
import numpy as np
import anndata as ad
import numpy as np
from scipy import sparse
import pandas as pd
from anndata import AnnData
from hira.src.utils.util import retrieve_adata
from hira.src.config import (
    CLOCKS_DIR,
    CLOCK_STATS_DIR,
    USE_LOCAL_CLOCK, 
    CLOCK_V
)


def save_clock_stats(df, name):
    """Persist a clock summary table. Figures and tests_code must read the same numbers,
    so every statistic a clock figure annotates is written here."""
    path = f'{CLOCK_STATS_DIR}/{name}.csv'
    df.to_csv(path, index=False)
    print(f'  Saved stats: {path}')
    return path

def wrapper_predict_age(adata, cell_type, USE_LOCAL_CLOCK=USE_LOCAL_CLOCK, version=CLOCK_V, model_dir=CLOCKS_DIR):
    import sys
    sys.path.insert(0, '../GRNimmuneClock')
    from grnimmuneclock import retrieve_function
    # ponytail: AgingClock/predict_age only support CD4T/CD8T (published models); local
    # retrained clocks (CLOCKS_DIR) cover all MAJOR_CTS, so predict directly instead of
    # going through AgingClock's restricted public API.
    model, gene_names = retrieve_function(
        cell_type=cell_type,
        model_dir=model_dir if USE_LOCAL_CLOCK else None,
        version=version,
    )
    adata_aligned = ad.AnnData(
        X=pd.DataFrame(adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X),
                        columns=adata.var_names, index=adata.obs_names)
          .reindex(columns=gene_names, fill_value=0).values,
        obs=adata.obs,
    )
    adata.obs['predicted_age'] = model.predict(adata_aligned.X)
    if 'age' in adata.obs.columns:
        adata.obs['age_acceleration'] = adata.obs['predicted_age'] - adata.obs['age']
    return adata
def wrapper_clock_predictions(cell_types, evaluate_datasets, data_type='bulk', condition=None, version=CLOCK_V,
                              model_dir=CLOCKS_DIR, only_net_genes=False, only_sig_genes=False):
    obs_store = []
    for cell_type in cell_types:
        for dataset in evaluate_datasets:
            adata = retrieve_adata(dataset=dataset, data_type=data_type, cell_type=cell_type,
                                    only_net_genes=only_net_genes, only_sig_genes=only_sig_genes, condition=condition)
            conds = adata.obs['condition'].unique()
            for cond in conds:
                adata_c = adata[adata.obs['condition'] == cond]
                wrapper_predict_age(adata=adata_c, cell_type=cell_type, version=version, model_dir=model_dir)
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
