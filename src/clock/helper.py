
from ciim.src.common import save_dir
import pandas as pd
import numpy as np
import anndata as ad
clock_save_dir = f"{save_dir}/clock/"


def save_function(model, gene_names, cell_type, data_type, feature_type, reg_type, version, model_args=None, model_kwargs=None):
    import os
    import joblib
    import numpy as np
    import torch

    if reg_type == 'NN':
        model_path = os.path.join(clock_save_dir, f"{cell_type}_{data_type}_{feature_type}_{reg_type}_model.pt")
    
        torch.save({
            'state_dict': model.state_dict(),
            'model_args': model_args,
            'model_kwargs': model_kwargs
        }, model_path)
    else:
        joblib.dump(model, os.path.join(clock_save_dir, f"{cell_type}_{data_type}_{feature_type}_{reg_type}_model.pkl"))
    np.savetxt(f'{clock_save_dir}/feature_names_{cell_type}_{data_type}_{feature_type}_{version}.txt', gene_names, fmt='%s')

def retrieve_function(reg_type, cell_type, data_type, feature_type, version):
    import os
    import joblib
    import numpy as np
    
    
    if reg_type == 'NN':
        from ciim.src.clock.NN import AgePredictionModel
        import torch
        model_path = os.path.join(clock_save_dir, f"{cell_type}_{data_type}_{feature_type}_{reg_type}_model.pt")
        model_class = AgePredictionModel
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model_args = checkpoint['model_args']
        model_kwargs = checkpoint['model_kwargs']

        model = model_class(*model_args, **model_kwargs)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
    else:
        model = joblib.load(os.path.join(clock_save_dir, f"{cell_type}_{data_type}_{feature_type}_{reg_type}_model.pkl"))
    gene_names = np.loadtxt(f'{clock_save_dir}/feature_names_{cell_type}_{data_type}_{feature_type}_{version}.txt', dtype=str)
    return model, gene_names

def evaluate_groupwise_median(obs):
    import pandas as pd
    from scipy.stats import spearmanr
    from sklearn.metrics import r2_score
    import numpy as np
    df = obs.copy()
    df['age'] = df['age'].astype(float)
    df['donor_id'] = df['donor_id'].astype(str)

    predicted_age = df.groupby(['donor_id', 'age'])['predicted_age'].median().values
    actual_age = df.groupby(['donor_id', 'age'])['age'].median().values

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
    from ciim.src.tf_activity.helper import retrieve_adata_bulk, get_consensus_net, calculate_tf_activity
    from ciim.src.common import mapping_minor_2_major, datasets_all, minor_cell_types
    from scipy.sparse import issparse

    cell_type_major = mapping_minor_2_major.get(cell_type, cell_type)
    adata = retrieve_adata_bulk(dataset, cell_type=cell_type_major, type=data_type)
    if feature_type == 'tf_activity':
        net = get_consensus_net(datasets_all, cell_type_major, min_degree=3)
        adata = calculate_tf_activity(adata, net)
    elif feature_type == 'gene_expression':
        pass
    else:
        raise ValueError('Unknown feature type')
    print(dataset, adata.X.shape)
    if 'minor' in data_type:
        adata = adata[adata.obs['Sub_CT'].isin(minor_cell_types)].copy()
        adata = pivot_adata_minor(adata)
        print('pivoted', adata.X.shape)
    return adata

def predict_age(X, cell_type, feature_type='tf_activity', data_type='bulk', reg_type='ridge', version='v1.0'):
    import joblib
    import numpy as np
    from scipy import sparse
    import pandas as pd
    from scipy.stats import spearmanr
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler
    from scipy.sparse import issparse
    

    if isinstance(X, pd.DataFrame):
        adata = _df_to_adata(X)
    else:
        adata = X
    # - load the model
    model, gene_names = retrieve_function(reg_type, cell_type, data_type, feature_type, version)
        
    if True:
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
    else:
        X = adata.X
    
    if issparse(X):
        X = X.toarray()  # convert sparse to dense
    
    if reg_type == 'NN':
        import torch
        from ciim.src.clock.NN import predict
        X = torch.tensor(X, dtype=torch.float32)
        predicted_age = predict(model, X)
        predicted_age = predicted_age.detach().numpy()
    else:
        predicted_age = model.predict(X)


    adata.obs['predicted_age'] = predicted_age.copy()

    # - show the score
    if False:
        age = adata.obs['age']
        predicted_age = adata.obs['predicted_age']
        # print(predicted_age.shape, age.shape)
        rr_dict = {'Spearman': spearmanr(age, predicted_age)[0], 'R2': r2_score(age, predicted_age)}
        print(rr_dict)
    else:
        scores = evaluate_groupwise_median(adata.obs)
        print(scores)
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

def tune_params_gbm(model, X, y, groups, scoring, n_trials=50, random_state=42):
    import optuna
    import lightgbm as lgb
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
    import numpy as np
    # Silence Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Silence LightGBM logs
    model.set_params(verbosity=-1)

    logo = LeaveOneGroupOut()
    cv_splits = list(logo.split(X, y, groups=groups))

    def objective(trial):
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 7, 20),  # Must be int
            'max_depth': trial.suggest_int('max_depth', 3, 10),    # Must be int
            # 'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            # 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),  # Add log for efficiency
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            # 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            # 'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            # 'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
            # 'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'random_state': random_state,
            'n_jobs': -1,
            'verbosity': -1 
        }

        model.set_params(**params)
        scores = cross_val_score(model, X, y, cv=cv_splits, scoring=scoring)
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print("Best parameters:")
    print(study.best_params)
    print("Best CV score:")
    print(study.best_value)

    # Return trained model with best parameters
    model.set_params(**study.best_params)
    return model

def spearman_corr(y_true, y_pred):
    from scipy.stats import spearmanr
    return spearmanr(y_true, y_pred).correlation
def tune_params_ridge(model, X, y, groups, scoring):
    from sklearn.metrics import make_scorer, r2_score
    from sklearn.model_selection import cross_val_score
    import numpy as np
    from sklearn.model_selection import LeaveOneGroupOut

    # Define candidate alphas
    alphas = [.1, 1, 10, 100]

    best_alpha = None
    best_score = -np.inf

    for alpha in alphas:
        model = model.set_params(ridge__alpha=alpha)
        cv = LeaveOneGroupOut().split(X, y, groups=groups)
        scores = cross_val_score(model, X, y, cv=list(cv), scoring=scoring)
        mean_score = np.mean(scores)
        print(scores)
        print(f"Alpha: {alpha}, Mean performance: {mean_score:.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = alpha

    print(f"\nBest alpha: {best_alpha}, Best CV performance: {best_score:.4f}")
    model.set_params(ridge__alpha=alpha)
    return model

def pivot_adata_minor(adata):
    from scipy.sparse import issparse
    import pandas as pd
    import anndata as ad
    adata.obs['Major_CT'] = adata.obs['Major_CT'].astype('str')
    adata.obs['Sub_CT'] = adata.obs['Sub_CT'].astype('str')
    adata.obs['donor_age'] = adata.obs['donor_id'].astype('str') + '_' + adata.obs['age'].astype('str')
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    expr_df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)

    # Attach metadata
    cols = ['Major_CT', 'Sub_CT', 'donor_id', 'age']
    if 'disease' in adata.obs.columns:
        cols.append('disease')
    if 'treatment' in adata.obs.columns:
        cols.append('treatment')

    for col in cols:
        expr_df[col] = adata.obs[col].values

    # Melt into long format
    
    melted = expr_df.melt(id_vars=cols, var_name='gene', value_name='expression')

    # Create gene_Sub_CT column
    melted['gene_sub'] = melted['gene'] + '//' + melted['Sub_CT']

    # - pivot and reformat
    unique_samples = [col for col in cols if col not in ['Sub_CT']]
    pivot = melted.pivot_table(index=unique_samples, columns='gene_sub', values='expression').fillna(0)
    
    obs = pd.DataFrame(pivot.index.tolist(), columns=pivot.index.names)
    adata_pivot = ad.AnnData(X=pivot.values, obs=obs, var=pd.DataFrame(index=pivot.columns))

    
    adata_pivot.obs = adata_pivot.obs.merge(adata.obs.drop_duplicates(subset=unique_samples), on=unique_samples, how='left')
    return adata_pivot


def build_model(reg_type, X, y, batch_labels, par):
    # - choose the model
    if reg_type == 'tabpfn':
        from tabpfn import TabPFNRegressor 
        model = TabPFNRegressor()  
    elif reg_type == 'NN':
        from ciim.src.clock.NN import train, AgePredictionModel, seed_all, predict
        import torch
        import numpy as np
        from torch.utils.data import DataLoader, TensorDataset

        # - format the inputs
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        # - train
        n_genes = X.shape[1]
        n_batches = int(batch_labels.max().item()) + 1

        model_args = (n_genes, n_batches)
        model_kwargs = {'latent_dim': 32, 'hidden_dim': 128, 'dropout': .2}

        seed_all(42)
        model = AgePredictionModel(*model_args, **model_kwargs)

        model = train(model, X, y, batch_labels, epochs=500, lr=1e-3, batch_size=64, tmp_dir=par['temp_dir'])

        y_trained = predict(model, X, batch_labels).detach().numpy()

        return model, model_args, model_kwargs, y_trained 
    elif reg_type == 'ridge':
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge


        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=1, random_state=42)
        )
    elif reg_type == 'elasticnet':
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import ElasticNet
        model = make_pipeline(
            StandardScaler(),
            ElasticNet(alpha=1.0, l1_ratio=0.1, random_state=42)
        )
    elif reg_type == 'GBM':
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
                random_state=42,
                n_jobs=-1
            )        
    else:
        raise ValueError('Unknown reg_type')
    if reg_type != 'NN':
        # - feature selection
        # select_features()
        # - tune the model
        if  par['tune_model']:
            groups = adata_all.obs['dataset'].values
            if reg_type == 'GBM':
                model = tune_params_gbm(model, X, y, groups, scoring=make_scorer(r2_score, greater_is_better=False))
            elif reg_type == 'ridge':
                model = tune_params_ridge(model, X, y, groups, scoring=make_scorer(spearman_corr, greater_is_better=True))
        # - fit the model
        model.fit(X, y)
        y_trained = model.predict(X)
        # - save
        return model, None, None, y_trained    

def wrapper_build_model_cell_type(cell_type, par):
    import anndata as ad
    from scipy.sparse import issparse
    # - prepare the data
    reg_type = par['reg_type']
    feature_type = par['feature_type']
    data_type = par['data_type']
    datasets_training = par['datasets_training']
    adata_store = []
    for dataset in datasets_training:
        # adata = prepare_input(dataset, cell_type, feature_type=feature_type, data_type=data_type)
        adata = ad.read_h5ad(f"{save_dir}/tf_activity_smoothed/{dataset}_{cell_type}_{data_type}.h5ad")
        if 'SLE' in dataset:
            adata = adata[adata.obs['disease']=='normal'].copy()
        adata_store.append(adata)
    adata_all = ad.concat(adata_store, join='inner', axis=0)
    gene_names = adata_all.var_names.values

    batch_labels = adata_all.obs['dataset'].astype('category').cat.codes.values # is used for NN training
    
    X = adata_all.X
    y = adata_all.obs['age']
    if issparse(X):
        X = X.toarray()  
    # - train the model

    def remove_collinear_features(X, threshold=0.9):
        """
        Remove collinear features from X using a correlation threshold.
        Returns reduced X and list of retained feature names.
        """
        corr_matrix = X.corr().abs()
        upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        to_drop = set()

        for i in range(corr_matrix.shape[0]):
            for j in range(i+1, corr_matrix.shape[1]):
                if corr_matrix.iloc[i, j] > threshold:
                    to_drop.add(corr_matrix.columns[j])  # drop the second TF

        retained_features = [f for f in X.columns if f not in to_drop]
        return X[retained_features], retained_features
    X, gene_names = remove_collinear_features(pd.DataFrame(X, columns=gene_names), threshold=0.95)
    print(cell_type, f"Number of features after removing collinear features: {X.shape[1]}")
    model, model_args, model_kwargs, y_trained = build_model(reg_type, X, y, batch_labels, par)
    # - save the training performance
    adata_all.obs['predicted_age'] = y_trained.copy()
    adata_all.write(f"{par['temp_dir']}/{cell_type}_{data_type}_{feature_type}_{reg_type}_adata.h5ad")
    # - save the model and feature space
    save_function(model, gene_names, cell_type, data_type, feature_type, reg_type, version=par['version'], model_args=model_args, model_kwargs=model_kwargs)
        
    