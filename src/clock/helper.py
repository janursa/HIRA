
from ciim.src.common import clock_save_dir
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


def save_function(model, gene_names, cell_type, data_type, feature_type, reg_type, version, model_args=None, model_kwargs=None):
    import os
    import joblib
    import numpy as np
    os.makedirs(clock_save_dir, exist_ok=True)
    if reg_type == 'NN':
        import torch
        model_path = os.path.join(clock_save_dir, f"{cell_type}_{data_type}_{feature_type}_{reg_type}_{version}_model.pt")
        torch.save({
            'state_dict': model.state_dict(),
            'model_args': model_args,
            'model_kwargs': model_kwargs
        }, model_path)
    else:
        joblib.dump(model, os.path.join(clock_save_dir, f"{cell_type}_{data_type}_{feature_type}_{reg_type}_{version}_model.pkl"))
    np.savetxt(f'{clock_save_dir}/feature_names_{cell_type}_{data_type}_{feature_type}_{version}.txt', gene_names, fmt='%s')

def retrieve_function(reg_type, cell_type, data_type, feature_type, version):
    import os
    import joblib
    import numpy as np
    if reg_type == 'NN':
        from ciim.src.clock.NN.NN import VAEAgeModel
        import torch
        model_path = os.path.join(clock_save_dir, f"{cell_type}_{data_type}_{feature_type}_{reg_type}_{version}_model.pt")
        model_class = VAEAgeModel
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model_args = checkpoint['model_args']
        model_kwargs = checkpoint['model_kwargs']

        model = model_class(*model_args, **model_kwargs)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
    else:
        model = joblib.load(os.path.join(clock_save_dir, f"{cell_type}_{data_type}_{feature_type}_{reg_type}_{version}_model.pkl"))
    gene_names = np.loadtxt(f'{clock_save_dir}/feature_names_{cell_type}_{data_type}_{feature_type}_{version}.txt', dtype=str)
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
def prepare_input(dataset, cell_type, feature_type='tf_activity', data_type='bulk'):
    '''
    For tf_activity, it gets the consensus net and calculates the tf activity'''
    from ciim.src.feature_association.helper import retrieve_adata_bulk, get_consensus_net, calculate_tf_activity
    from ciim.src.common import mapping_minor_2_major, datasets_all, minor_cell_types
    from scipy.sparse import issparse

    cell_type_major = mapping_minor_2_major.get(cell_type, cell_type)
    net = get_consensus_net(datasets_all, cell_type_major)
    
    adata = retrieve_adata_bulk(dataset, cell_type=cell_type_major, type=data_type)
    if feature_type == 'tf_activity':
        adata = calculate_tf_activity(adata, net)
    elif feature_type == 'gene_expression':
        targets = net['target'].unique()
        adata = adata[:, adata.var_names.isin(targets)].copy()
    else:
        raise ValueError('Unknown feature type')
    adata.obs['donor_age'] = adata.obs['donor_id'].astype(str) + '_' + adata.obs['age'].astype(str)
    print(dataset, adata.X.shape)
    if 'minor' in data_type:
        adata = adata[adata.obs['Sub_CT'].isin(minor_cell_types)].copy()
        adata = pivot_adata_minor(adata)
        print('pivoted', adata.X.shape)
    return adata
def align_feature_space(adata, gene_names):
    var_names = np.array(adata.var_names)
    var_index = {gene: i for i, gene in enumerate(var_names)}

    # Collect indices or mark as -1 for missing
    idxs = np.array([var_index.get(gene, -1) for gene in gene_names])

    # Create a matrix with correct shape
    rows = adata.shape[0]
    cols = len(gene_names)
    X_aligned = sparse.lil_matrix((rows, cols))

    # Fill in available gene columns
    present = idxs != -1
    if present.sum() > 0:
        X_aligned[:, present] = adata.X[:, idxs[present]]

    # Convert to CSR for efficient prediction
    X = X_aligned.tocsr()
    if issparse(X):
        X = X.toarray()  # convert sparse to dense
    return X
def predict_age(adata, cell_type, feature_type='tf_activity', data_type='bulk', reg_type='ridge', version='v1.0'):
    
    # - load the model
    model, gene_names = retrieve_function(reg_type, cell_type, data_type, feature_type, version)
    
    # - align the genes
    X = align_feature_space(adata, gene_names)
    predicted_age = model.predict(X)
    adata.obs['predicted_age'] = predicted_age.copy()
    # # - show the score
    # if False:
    #     age = adata.obs['age']
    #     predicted_age = adata.obs['predicted_age']
    #     # print(predicted_age.shape, age.shape)
    #     rr_dict = {'Spearman': spearmanr(age, predicted_age)[0], 'R2': r2_score(age, predicted_age)}
    #     print(rr_dict)
    # else:
    #     scores = evaluate_groupwise_median(adata.obs)
    return adata

# def stability_selection_shap(model, X, y,  top_q=80):
#     """
#     Perform stability selection using SHAP values for feature importance.

#     Parameters:
#     - X: Feature matrix
#     - y: Target vector
#     - n_bootstrap: Number of bootstrap iterations.
#     - top_k: Number of top features to select.

#     Returns:
#     - top_predictors: List of selected top-q most important features.
#     """
#     import shap
#     import numpy as np

#     model_function = lambda X: model.predict(X)

#     model.fit(X, y)

#     # Compute SHAP values
#     explainer = shap.Explainer(model_function, X)
#     required_evals = 2 * X.shape[1] + 1
#     shap_values = explainer(X, max_evals=required_evals)

#     # Compute mean absolute SHAP values for feature importance
#     feature_importances = np.abs(shap_values.values).mean(axis=0)


#     # Compute the q percentile threshold
#     threshold = np.percentile(feature_importances, top_q)

#     # Select features above the threshold
#     top_features_idx = np.where(feature_importances >= threshold)[0]


#     return top_features_idx

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
