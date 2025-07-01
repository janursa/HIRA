

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import optuna
from sklearn.metrics import r2_score, make_scorer, get_scorer
import warnings
warnings.filterwarnings('ignore')

def spearman_corr(y_true, y_pred):
    from scipy.stats import spearmanr
    return spearmanr(y_true, y_pred).correlation



loss_function = make_scorer(spearman_corr, greater_is_better=True)  # make_scorer(spearman_corr, greater_is_better=True) 


def tune_params_gbm(model, X, y, cv_groups, scoring, n_trials=50, random_state=42):
    import optuna
    import lightgbm as lgb
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
    import numpy as np
    # Silence Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Silence LightGBM logs
    model.set_params(verbosity=-1)

    logo = LeaveOneGroupOut()
    cv_splits = list(logo.split(X, y, groups=cv_groups))

    def objective(trial):
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 4, 10),  # Must be int
            'max_depth': trial.suggest_int('max_depth', 3, 7),    # Must be int
            # 'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            # 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),  # Add log for efficiency
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            # 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            # 'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            # 'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
            # 'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'random_state': random_state,
            'n_jobs': 4,
            'verbosity': -2 
        }

        model.set_params(**params)
        # Perform cross-validation
        _, cv = get_custom_cv(cv_groups)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
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

    


def get_custom_cv(groups, main_code=0):
    """
    Leave-one-group-out CV, where each fold leaves out one non-main group for testing,
    and always includes the main_code group in training.
    """
    unique_groups = np.unique(groups)
    ordered_test_groups = [g for g in unique_groups if g != main_code]

    custom_splits = []
    for test_group in ordered_test_groups:
        test_mask = (groups == test_group)
        train_mask = (groups != test_group)  # includes main_code and all others
        custom_splits.append((np.where(train_mask)[0], np.where(test_mask)[0]))
    return ordered_test_groups, custom_splits

def tune_params_ridge(X, y, cv_groups=None, scoring='r2'):
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.01, 1000.0, log=True)
        model = Pipeline([
            ('standardscaler', StandardScaler()),
            ('ridge', Ridge(alpha=alpha, random_state=42))
        ])
        if cv_groups is not None:
            _, cv = get_custom_cv(cv_groups)
        else:
            raise ValueError("cv_groups must be provided to ensure main dataset is always in training.")
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    best_alpha = study.best_params['alpha']
    best_score = study.best_value

    print(f"\nBest alpha: {best_alpha:.4f}, Best CV performance: {best_score:.4f}")

    best_model = Pipeline([
        ('standardscaler', StandardScaler()),
        ('ridge', Ridge(alpha=best_alpha, random_state=42))
    ])

    return best_model

def build_model(reg_type, X, y, batch_labels, tune_model, temp_dir):
    import anndata as ad
    
    print(X.shape, y.shape)
    # - choose the model
    if reg_type == 'tabpfn':
        from tabpfn import TabPFNRegressor 
        model = TabPFNRegressor()  
    elif reg_type == 'NN':
        from ciim.src.clock.NN.NN import VAEAgeModel, seed_all
        import torch

        # - format the inputs
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        # - train
        n_genes = X.shape[1]
        n_batches = int(batch_labels.max().item()) + 1

        model_args = (n_genes, n_batches)
        model_kwargs = {'latent_dim': 32, 'hidden_dim': 128, 'dropout': .2}
        train_kwargs = {'batch_size': 64, 'epochs': 200, 'lr': 1e-3, 
                        'alpha':1, # reconstruction loss weight
                        'beta': 1.0, # age prediction loss weight
                        'gamma':0, # KL divergence weight
                        }

        seed_all(42)
        model = VAEAgeModel(*model_args, **model_kwargs)

        from ciim.src.clock.NN.train import train as train_NN
        model = train_NN(model, X, y, batch_labels, tmp_dir=temp_dir, **train_kwargs)
        y_trained = model.predict(X, batch_labels)
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
        
        if tune_model:
            if reg_type == 'GBM':
                model = tune_params_gbm(model, X, y, batch_labels, scoring=loss_function)
            elif reg_type == 'ridge':
                model = tune_params_ridge(X, y, batch_labels, scoring=loss_function)
            elif reg_type == 'elasticnet':
                model = tune_params_elasticnet(X, y, batch_labels, scoring=loss_function)

        # - fit the model
        model.fit(X, y)
        y_trained = model.predict(X)
        # - save
        return model, None, None, y_trained    

def wrapper_build_model_cell_type(cell_type, par):
    import anndata as ad
    from scipy.sparse import issparse
    from ciim.src.clock.helper import save_function
    from ciim.src.common import save_dir
    import pandas as pd
    import numpy as np
    # - prepare the data
    reg_type = par['reg_type']
    feature_type = par['feature_type']
    data_type = par['data_type']
    datasets_training = par['datasets_training']
    age_limit = par['age_limit']
    main_dataset = 'data1'

    adata_store = []
    for dataset in datasets_training:
        # adata = prepare_input(dataset, cell_type, feature_type=feature_type, data_type=data_type)
        adata = ad.read_h5ad(f"{save_dir}/{feature_type}_smoothed/{dataset}_{cell_type}_{data_type}.h5ad")
        adata = adata[(adata.obs['age']>age_limit)].copy()
        # print(adata.obs['age'].min())
        if 'SLE' in dataset:
            adata = adata[adata.obs['disease']=='normal'].copy()
        adata_store.append(adata)
    adata_all = ad.concat(adata_store, join='inner', axis=0)
    
    
    # explicitly order categories with main_dataset first
    all_datasets = adata_all.obs['dataset'].unique().tolist()
    # Move main_dataset to the front
    ordered_datasets = [main_dataset] + [d for d in all_datasets if d != main_dataset]
    adata_all.obs['dataset'] = adata_all.obs['dataset'].astype(pd.CategoricalDtype(categories=ordered_datasets, ordered=True))

    dataset_code_map = dict(zip(
        range(len(adata_all.obs['dataset'].cat.categories)),
        adata_all.obs['dataset'].cat.categories
    ))
    # Now this will give main_dataset code 0
    batch_labels = adata_all.obs['dataset'].cat.codes.values

    adata_all.write(f"{par['temp_dir']}/{cell_type}_{data_type}_{feature_type}_{reg_type}_adata.h5ad")
    gene_names = adata_all.var_names.values
    
    X = adata_all.X
    y = adata_all.obs['age']
    if issparse(X):
        X = X.toarray()  
    # - train the model
    if True: # use dataset labels for cross validation (or training in NN)
        batch_labels=batch_labels
    else:
        batch_labels = None # random CV
    model, model_args, model_kwargs, y_trained = build_model(reg_type, X, y, batch_labels=batch_labels, tune_model=par['tune_model'], temp_dir=par['temp_dir'])

    # Run CV again to get per-group scores
    if reg_type != 'NN':
        ordered_test_groups, cv = get_custom_cv(batch_labels)
        scorer = get_scorer(loss_function)
        group_scores = []

        fold_scores = {}
        for i, code in enumerate(ordered_test_groups):
            train_idx, test_idx = cv[i]
            model.fit(X[train_idx], y.iloc[train_idx])
            score = scorer(model, X[test_idx], y.iloc[test_idx])
            group_scores.append(score)
            fold_scores[dataset_code_map[code]] = round(score, 2)
        print(fold_scores)
    # - save the training performance
    adata_all.obs['predicted_age'] = y_trained.copy()
    adata_all.write(f"{par['temp_dir']}/{cell_type}_{data_type}_{feature_type}_{reg_type}_adata.h5ad")
    # - save the model and feature space
    save_function(model, gene_names, cell_type, data_type, feature_type, reg_type, version=par['version'], model_args=model_args, model_kwargs=model_kwargs)

    