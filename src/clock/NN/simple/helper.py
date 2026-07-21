


def wrapper_get_latent(adata, batch_idx=None, reg_type='NN', cell_type='CD8T', data_type='bulk', feature_type='gene_expression', version='v1.0'):
    from hira.src.clock.helper import retrieve_function, align_feature_space
    model, gene_names = retrieve_function(reg_type, cell_type, data_type, feature_type, version)
    X = align_feature_space(adata, gene_names)
    latent = model.get_latent(X, batch_idx=batch_idx)
    adata.obsm['VAE_latent'] = latent
    return adata
