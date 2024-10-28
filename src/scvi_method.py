import scanpy as sc
import scvi
par = {
    'categorical_covariate_keys' : ["donor_id"],
    'SCVI_LATENT_KEY' : "X_scVI",
    'SCVI_NORMALIZED_KEY' : "scvi_normalized"
}
def scvi_batch_correct(adata, par):
    scvi.model.SCVI.setup_anndata(
        adata,
        layer="counts",
        categorical_covariate_keys=par['categorical_covariate_keys']
    )
    model = scvi.model.SCVI(adata)
    model.train()
    latent = model.get_latent_representation()
    adata.obsm[par['SCVI_LATENT_KEY']] = latent

    adata.layers[par['SCVI_NORMALIZED_KEY']] = model.get_normalized_expression(library_size=10e4)

if __name__ == '__main__':
    adata = ad.read_h5ad('output/data/35_44.h5ad')
    
    scvi_batch_correct(adata, par)
    print(adata)