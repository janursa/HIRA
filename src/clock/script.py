
from ciim.src.clock.helper import prepare_input
from ciim.src.common import cell_types



def wrapper_build_model_cell_type(cell_type):
    import os
    import joblib
    from sklearn.linear_model import Ridge
    from anndata import AnnData
    import numpy as np
    import pandas as pd
    import anndata as ad


    feature_type = 'tf_activity' #'gene_expression'

    save_dir = "/vol/projects/jnourisa/prior/clock/"
    # os.makedirs(save_dir, exist_ok=True)
    # - step1: merge the datasets
    from scipy.sparse import vstack
    from ciim.src.clock.helper import prepare_input
    datasets_clock = ['data1', 'data7_allTPs_jalil', 'data12', 'data13_Korean' , 'data13_Japanese']

    adata_store = []
    for dataset in datasets_clock:
        adata = prepare_input(dataset, cell_type, feature_type=feature_type)
        adata_store.append(adata)
    adata_all = ad.concat(adata_store, join='inner', axis=0)

    # - step2: train the models and save them
    X = adata_all.X
    age = adata_all.obs['age']

    model = Ridge(alpha=1.0)
    model.fit(X, age)

    # - save the model
    model_path = os.path.join(save_dir, f"{cell_type}_{feature_type}_ridge_model.pkl")
    joblib.dump(model, model_path)

    # save feature names
    expected_genes = adata_all.var_names.values
    np.savetxt(f'{save_dir}/feature_names_{cell_type}_{feature_type}.txt', expected_genes, fmt='%s')


def wrapper_build_model_all():
    for cell_type in cell_types:
        print('building model for cell type:', cell_type)
        wrapper_build_model_cell_type(cell_type)
    


if __name__ == "__main__":
    wrapper_build_model_all()