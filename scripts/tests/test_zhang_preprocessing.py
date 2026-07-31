"""
Validation test for zhang dataset preprocessing outputs.

Purpose
-------
Validates that new SC and pseudobulk outputs from the preprocessing pipeline
are structurally correct and consistent with the reference outputs. Run this
after any change to the preprocessing pipeline before reprocessing all datasets.

Pipeline overview
-----------------
The preprocessing pipeline (src/process_dataset/preprocess/script.py) does:
  1. Load raw SC h5ad (backed), apply basic QC (filter cells/genes), load to memory
  2. Annotate cell types via annotate_celltypist_fast():
     - Back up raw counts to /tmp/ (avoids X.copy() memory spike)
     - Normalize + log1p in-place
     - Compute 3000 HVGs (seurat flavor, no counts layer needed)
     - MiniBatchKMeans (n=500) on sparse HVG matrix as over-clustering
       (replaces PCA+neighbors+Leiden — runs in seconds even for 10M+ cells)
     - CellTypist majority voting using pre-computed clusters (no internal UMAP)
     - Restore raw counts from /tmp/; assign Major_CT and Sub_CT
  3. Write SC h5ad (raw integer counts, no layers)

Bulkify (src/process_dataset/bulkify/script.py) then:
  - Aggregates SC counts per (donor_id, age, Major_CT) group
  - Normalizes to CPM (1e6) + log1p
  - Filters samples with < 10 cells

Key design decisions
--------------------
- soundlife only: genes are pre-filtered to gene_names.txt before to_memory()
  to reduce peak memory during multi-file concat; all other datasets retain
  their full post-QC gene set.
- MiniBatchKMeans was chosen over Leiden because it works natively on sparse
  matrices with no PCA/graph construction, making it ~100x faster for large
  datasets while producing sufficient cluster granularity for majority voting.

Checks performed
----------------
SC:
  - Gene set matches reference (same 14,468 genes)
  - Cell count matches reference (538,266 cells)
  - Required obs columns present and NaN-free: donor_id, age, sex, Major_CT, Sub_CT, dataset
  - X is sparse, non-negative, integer-like (raw counts, not normalized)
  - Major_CT labels are from the expected vocabulary

Bulk:
  - Gene set matches the new SC output (bulk is aggregated from SC)
  - Required obs columns present and NaN-free: donor_id, age, sex, Major_CT, cell_count
  - cell_count > 0 for all samples
  - X is float and log-normalized (not integer counts)

Usage
-----
    cd <hira repo root>
    python scripts/tests/test_zhang_preprocessing.py \\
        --new_sc   temp/sc_new/zhang.h5ad \\
        --new_bulk temp/bulk_new/zhang.h5ad \\
        [--ref_sc  <path to reference sc h5ad>] \\
        [--ref_bulk <path to reference bulk h5ad>]

Reference paths default to $HIRA_BASE_DIR/datasets/{sc,bulk}/zhang.h5ad
(see .env / README) unless overridden via --ref_sc/--ref_bulk.

Exits 0 if all checks pass, 1 otherwise.
"""

import argparse
import os
import sys
import numpy as np
import anndata as ad
from hira.src.config import DATA_DIR

REF_SC   = os.path.join(DATA_DIR, 'sc/zhang.h5ad')
REF_BULK = os.path.join(DATA_DIR, 'bulk/zhang.h5ad')

REQUIRED_SC_OBS_COLS   = ['donor_id', 'age', 'sex', 'Major_CT', 'Sub_CT', 'dataset']
REQUIRED_BULK_OBS_COLS = ['donor_id', 'age', 'sex', 'Major_CT', 'cell_count']


def fail(msg):
    print(f'  FAIL: {msg}')
    return False


def ok(msg):
    print(f'  OK  : {msg}')
    return True


# ---------------------------------------------------------------------------
# SC checks
# ---------------------------------------------------------------------------

def check_sc(new: ad.AnnData, ref: ad.AnnData) -> bool:
    passed = True
    print('\n[SC] Gene space')

    if set(new.var_names) == set(ref.var_names):
        ok(f'Gene sets match ({new.n_vars} genes)')
    else:
        only_new = set(new.var_names) - set(ref.var_names)
        only_ref = set(ref.var_names) - set(new.var_names)
        passed &= fail(f'Gene mismatch: {len(only_new)} only in new, {len(only_ref)} only in ref')

    print('[SC] Shape')
    if new.n_obs == ref.n_obs:
        ok(f'Cell count matches: {new.n_obs}')
    else:
        passed &= fail(f'Cell count differs: new={new.n_obs}, ref={ref.n_obs}')

    print('[SC] obs columns')
    missing = [c for c in REQUIRED_SC_OBS_COLS if c not in new.obs.columns]
    if missing:
        passed &= fail(f'Missing obs columns: {missing}')
    else:
        ok(f'All required obs columns present: {REQUIRED_SC_OBS_COLS}')

    print('[SC] obs NaN check')
    nan_cols = [c for c in REQUIRED_SC_OBS_COLS if new.obs[c].isna().any()]
    if nan_cols:
        for c in nan_cols:
            n = new.obs[c].isna().sum()
            passed &= fail(f'{c} has {n} NaN values')
    else:
        ok('No NaN in required obs columns')

    print('[SC] X matrix — raw integer counts')
    X = new.X
    xmin = X.min()
    xmax = X.max()
    if xmin < 0:
        passed &= fail(f'X contains negative values (min={xmin})')
    else:
        ok(f'X non-negative (min={xmin})')

    # Check values are integer-like (counts, not log-normalized)
    import scipy.sparse as sp
    sample = X[:1000] if not sp.issparse(X) else X[:1000].toarray()
    if np.allclose(sample, np.round(sample)):
        ok(f'X values are integer-like (raw counts confirmed), max={xmax}')
    else:
        passed &= fail('X values are not integer-like (may be normalized instead of raw counts)')

    print('[SC] Major_CT values')
    valid_major = {'CD4T', 'CD8T', 'B', 'NK', 'MONO', 'DC', 'HSC',
                   'Megakaryocyte', 'ILC', 'T', 'Erythroid', 'Others'}
    unknown = set(new.obs['Major_CT'].unique()) - valid_major
    if unknown:
        # warn but don't fail — new models may produce new labels
        print(f'  WARN: Unexpected Major_CT labels: {unknown}')
    else:
        ok(f'All Major_CT labels valid')

    return passed


# ---------------------------------------------------------------------------
# Bulk checks
# ---------------------------------------------------------------------------

def check_bulk(new: ad.AnnData, ref: ad.AnnData, new_sc: ad.AnnData) -> bool:
    passed = True
    print('\n[Bulk] Gene space')

    # Bulk genes should exactly match the SC gene space (bulk is aggregated from SC)
    sc_genes  = set(new_sc.var_names)
    new_genes = set(new.var_names)
    if new_genes == sc_genes:
        ok(f'Bulk gene set matches SC gene space ({len(new_genes)} genes)')
    else:
        only_bulk = new_genes - sc_genes
        missing   = sc_genes - new_genes
        passed &= fail(f'Bulk/SC gene mismatch: {len(only_bulk)} only in bulk, {len(missing)} only in SC')

    print('[Bulk] obs columns')
    missing = [c for c in REQUIRED_BULK_OBS_COLS if c not in new.obs.columns]
    if missing:
        passed &= fail(f'Missing obs columns: {missing}')
    else:
        ok(f'All required bulk obs columns present')

    print('[Bulk] obs NaN check')
    nan_cols = [c for c in REQUIRED_BULK_OBS_COLS if new.obs[c].isna().any()]
    if nan_cols:
        for c in nan_cols:
            n = new.obs[c].isna().sum()
            passed &= fail(f'{c} has {n} NaN values')
    else:
        ok('No NaN in required bulk obs columns')

    print('[Bulk] cell_count > 0')
    if (new.obs['cell_count'] <= 0).any():
        passed &= fail('Some pseudobulk samples have cell_count <= 0')
    else:
        ok(f'All cell_count > 0 (min={new.obs["cell_count"].min()})')

    print('[Bulk] X matrix — log-normalized (CPM + log1p)')
    X = new.X
    import scipy.sparse as sp
    xmin = float(X.min())
    xmax = float(X.max())
    if xmin < 0:
        passed &= fail(f'X contains negative values (min={xmin})')
    else:
        ok(f'X non-negative (min={xmin})')

    sample = X[:10].toarray() if sp.issparse(X) else X[:10]
    if np.allclose(sample, np.round(sample)):
        passed &= fail('X looks like integer counts — expected log-normalized floats')
    else:
        ok(f'X is float / log-normalized (max={xmax:.3f})')

    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--new_sc',   required=True)
    parser.add_argument('--new_bulk', required=True)
    parser.add_argument('--ref_sc',   default=REF_SC)
    parser.add_argument('--ref_bulk', default=REF_BULK)
    args = parser.parse_args()

    print('Loading files...')
    new_sc   = ad.read_h5ad(args.new_sc)
    ref_sc   = ad.read_h5ad(args.ref_sc)
    new_bulk = ad.read_h5ad(args.new_bulk)
    ref_bulk = ad.read_h5ad(args.ref_bulk)
    print(f'  new SC  : {new_sc.shape}')
    print(f'  ref SC  : {ref_sc.shape}')
    print(f'  new bulk: {new_bulk.shape}')
    print(f'  ref bulk: {ref_bulk.shape}')

    sc_ok   = check_sc(new_sc, ref_sc)
    bulk_ok = check_bulk(new_bulk, ref_bulk, new_sc)

    print('\n' + '='*50)
    if sc_ok and bulk_ok:
        print('ALL CHECKS PASSED')
        sys.exit(0)
    else:
        print('SOME CHECKS FAILED')
        sys.exit(1)


if __name__ == '__main__':
    main()
