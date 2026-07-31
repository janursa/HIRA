
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
    USE_LOCAL_CLOCK, 
    CLOCK_V
)

def wrapper_predict_age(adata, cell_type, USE_LOCAL_CLOCK=USE_LOCAL_CLOCK, version=CLOCK_V):
    import sys
    sys.path.insert(0, '../GRNimmuneClock')
    from grnimmuneclock import predict_age
    adata = predict_age(adata, cell_type=cell_type, version=version)
    return adata
def wrapper_clock_predictions(cell_types, evaluate_datasets, data_type='bulk', condition=None, version=CLOCK_V):
    obs_store = []
    for cell_type in cell_types:
        for dataset in evaluate_datasets:
            adata = retrieve_adata(dataset=dataset, data_type=data_type, cell_type=cell_type, only_net_genes=True, condition=condition)
            conds = adata.obs['condition'].unique()
            for cond in conds:
                adata_c = adata[adata.obs['condition'] == cond]
                wrapper_predict_age(adata=adata_c, cell_type=cell_type, version=version)
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
