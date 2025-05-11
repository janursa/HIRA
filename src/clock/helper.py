def _df_to_adata(df):
    import anndata as ad
    import pandas as pd
    adata = ad.AnnData(
        X=df.values,
        obs=pd.DataFrame(index=df.index).reset_index(),
        var=pd.DataFrame(index=df.columns)
    )
    return adata
def prepare_input(dataset, cell_type, feature_type='tf_activity'):
    '''
    For tf_activity, it gets the consensus net and calculates the tf activity'''
    from ciim.src.tf_activity.helper import retrieve_adata_bulk, get_consensus_net, calculate_tf_activity
    from ciim.src.common import mapping_minor_2_major, datasets_all

    cell_type_major = mapping_minor_2_major.get(cell_type, cell_type)
    adata = retrieve_adata_bulk(dataset, cell_type=cell_type_major)
    if feature_type == 'tf_activity':
        net = get_consensus_net(datasets_all, cell_type_major, min_degree=4)
        adata = calculate_tf_activity(adata, net)
    elif feature_type == 'gene_expression':
        pass
    else:
        raise ValueError('Unknown feature type')
    print(dataset,' feature space: ', adata.X.shape)
    return adata

def predict_age(X, cell_type, feature_type='tf_activity'):
    import joblib
    import numpy as np
    from scipy import sparse
    import pandas as pd
    from scipy.stats import spearmanr
    from sklearn.metrics import r2_score

    if isinstance(X, pd.DataFrame):
        adata = _df_to_adata(X)
    else:
        adata = X
        

    save_dir = "/vol/projects/jnourisa/prior/clock/"
    expected_genes = np.loadtxt(f'{save_dir}/feature_names_{cell_type}_{feature_type}.txt', dtype=str)
    model = joblib.load(f"{save_dir}/{cell_type}_{feature_type}_ridge_model.pkl")


    var_names = np.array(adata.var_names)
    var_index = {gene: i for i, gene in enumerate(var_names)}

    # Collect indices or mark as -1 for missing
    idxs = np.array([var_index.get(gene, -1) for gene in expected_genes])

    # Create a matrix with correct shape
    rows = adata.shape[0]
    cols = len(expected_genes)
    X_aligned = sparse.lil_matrix((rows, cols))

    # Fill in available gene columns
    present = idxs != -1
    if present.sum() > 0:
        X_aligned[:, present] = adata.X[:, idxs[present]]

    # Convert to CSR for efficient prediction
    X_final = X_aligned.tocsr()
    predicted_age = model.predict(X_final)

    adata.obs['predicted_age'] = predicted_age.copy()

    # - show the score
    age = adata.obs['age']
    predicted_age = adata.obs['predicted_age']
    # print(predicted_age.shape, age.shape)
    rr_dict = {'Spearman': spearmanr(age, predicted_age)[0], 'R2': r2_score(age, predicted_age)}
    # print(rr_dict)
    return adata