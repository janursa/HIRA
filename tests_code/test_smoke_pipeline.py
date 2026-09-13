"""
Smoke test for the hira package: import sanity + a synthetic end-to-end
run of basic_qc -> bulkify_func with schema checks. No real datasets required.

Usage
-----
    cd <hira repo root>
    python tests_code/test_smoke_pipeline.py

Exits 0 if all checks pass, 1 otherwise.
"""
import sys
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

import hira  # noqa: F401  (import sanity check for the whole package)
from hira.src.config import HIRA_DIR, DATA_DIR, PRIOR_DIR, OUTPUT_DIR
from hira.src.utils.util import basic_qc, bulkify_func

REQUIRED_SC_OBS_COLS = ['donor_id', 'age', 'sex', 'Major_CT', 'Sub_CT', 'dataset']
REQUIRED_BULK_OBS_COLS = ['donor_id', 'age', 'Major_CT', 'cell_count']


def make_synthetic_sc(n_cells=200, n_genes=50, seed=0):
    rng = np.random.default_rng(seed)
    X = sparse.csr_matrix(rng.poisson(2, size=(n_cells, n_genes)))
    donor_ids = [f'd{i % 10}' for i in range(n_cells)]
    # donor-level attributes (age, sex) must be constant per donor so bulkify_func's
    # one-to-one-mapping check keeps them after aggregation
    donor_age = {f'd{i}': int(rng.integers(20, 80)) for i in range(10)}
    donor_sex = {f'd{i}': rng.choice(['Male', 'Female']) for i in range(10)}
    obs = pd.DataFrame({
        'donor_id': donor_ids,
        'age': [donor_age[d] for d in donor_ids],
        'sex': [donor_sex[d] for d in donor_ids],
        'Major_CT': rng.choice(['CD4T', 'CD8T', 'B'], n_cells),
        'Sub_CT': rng.choice(['Tcm_Naive_CD4', 'Naive_B'], n_cells),
        'dataset': 'synthetic',
    })
    var = pd.DataFrame(index=[f'gene{i}' for i in range(n_genes)])
    return ad.AnnData(X=X, obs=obs, var=var)


def test_smoke_pipeline():
    print(f'[config] HIRA_DIR={HIRA_DIR}')
    print(f'[config] DATA_DIR={DATA_DIR}')
    print(f'[config] PRIOR_DIR={PRIOR_DIR}')
    print(f'[config] OUTPUT_DIR={OUTPUT_DIR}')

    sc_adata = make_synthetic_sc()
    sc_qc = basic_qc(sc_adata, min_genes_per_cell=0, max_genes_per_cell=10_000, min_cells_per_gene=0)
    assert sc_qc.n_obs == sc_adata.n_obs, 'basic_qc dropped cells unexpectedly on clean synthetic data'

    bulk = bulkify_func(sc_qc.copy(), cell_count_t=1, covariates=['Major_CT', 'donor_id', 'age'])
    assert bulk.n_obs > 0, 'bulkify_func produced an empty pseudobulk'
    assert (bulk.obs['cell_count'] > 0).all(), 'bulkify_func produced non-positive cell_count'

    import scanpy as scanpy_module
    scanpy_module.pp.normalize_total(bulk, target_sum=1e4)
    scanpy_module.pp.log1p(bulk)

    for col in REQUIRED_SC_OBS_COLS:
        assert col in sc_qc.obs.columns, f'missing required SC obs column: {col}'
        assert not sc_qc.obs[col].isna().any(), f'NaN in SC obs column: {col}'
    for col in REQUIRED_BULK_OBS_COLS:
        assert col in bulk.obs.columns, f'missing required bulk obs column: {col}'
        assert not bulk.obs[col].isna().any(), f'NaN in bulk obs column: {col}'
    # bulk is aggregated from sc, so the gene spaces must be identical
    assert set(bulk.var_names) == set(sc_qc.var_names), 'bulk gene space differs from sc'
    print('\nSMOKE TEST PASSED')


if __name__ == '__main__':
    test_smoke_pipeline()
    sys.exit(0)
