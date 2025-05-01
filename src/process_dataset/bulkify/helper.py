import scanpy as sc

from task_grn_inference.src.utils.util import sum_by

def normalize(adata):
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata
def bulkify_main(adata, cell_count_t=10, covariates=['cell_type', 'donor_id', 'age']):
    adata.obs['sum_by'] = ''
    for covariate in covariates:
        adata.obs['sum_by'] += '_' + adata.obs[covariate].astype(str)
    # adata.obs['sum_by'] = '_' + adata.obs['cell_type'].astype(str) + '_' + adata.obs['donor_id'].astype(str) + '_' + adata.obs['age'].astype(str) 
    adata.obs['sum_by'] = adata.obs['sum_by'].astype('category')
    adata_bulk = sum_by(adata, 'sum_by', unique_mapping=True)
    cell_count_df = adata.obs.groupby('sum_by').size().reset_index(name='cell_count')
    adata_bulk.obs = adata_bulk.obs.merge(cell_count_df, on='sum_by')
    adata_bulk = adata_bulk[adata_bulk.obs['cell_count']>cell_count_t]
    adata_bulk = normalize(adata_bulk)
    return adata_bulk