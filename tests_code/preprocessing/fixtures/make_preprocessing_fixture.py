"""Run once by hand to (re)generate data/preprocessing/{input,baseline_sc,baseline_bulk}.h5ad.

Uses the real pipeline functions (script.py's --run-test code path) on the zhang raw file
(smallest raw dataset, still opened backed so this doesn't require loading 3.5G into memory)
so the fixture is a faithful (if tiny) slice of the real preprocessing pipeline.
"""
import os
import sys

import numpy as np
import scanpy as sc

REPRO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIRA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(REPRO_DIR)))
sys.path.insert(0, HIRA_DIR)

import anndata as ad

from hira.src.process_data.preprocess.script import load_sc_data  # noqa: E402
from hira.src.process_data.preprocess.helper import basic_qc, annotate_celltypes  # noqa: E402
from hira.src.utils.util import bulkify_func, filter_rb_mt_genes  # noqa: E402
from hira.src.config import get_config, MAJOR_CT_LABEL, PRIOR_DIR  # noqa: E402
from hira.src.process_data.bulkify.script import normalize, qc_bulk  # noqa: E402

RAW_DATA_DIR = os.environ.get('HIRA_RAW_DIR', '/vol/projects/CIIM')
RAW_FILE = f'{RAW_DATA_DIR}/Healthy_Single_Cell_Data/count_matrix/data12_CMtx.h5ad'
DATASET = 'zhang'
OUT_DIR = os.path.join(REPRO_DIR, 'data', 'preprocessing')
N_DONORS = 4  # smallest-count donors -> real (not degenerate) but still tiny fixture


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f'Loading {RAW_FILE}, formatting, and subsetting to the {N_DONORS} smallest donors...', flush=True)
    adata = load_sc_data(RAW_FILE, DATASET, run_test=False)  # backed, formatted (bulk_group added)
    donor_sizes = adata.obs['donor_id'].value_counts().sort_values()
    keep_donors = donor_sizes.index[:N_DONORS]
    gene_names = np.loadtxt(f'{PRIOR_DIR}/gene_names.txt', dtype=str)
    adata = adata[adata.obs['donor_id'].isin(keep_donors), adata.var_names.isin(gene_names)].to_memory()
    print(f'Fixture input shape: {adata.shape} ({N_DONORS} donors)', flush=True)
    adata.write_h5ad(os.path.join(OUT_DIR, 'input.h5ad'), compression='gzip')

    print('Running basic_qc + annotate_celltypes (baseline_sc)...', flush=True)
    sc_baseline = basic_qc(adata.copy(), run_test=False)
    sc_baseline = annotate_celltypes(sc_baseline, DATASET)
    sc.pp.filter_genes(sc_baseline, min_counts=1)
    assert not sc_baseline.layers, 'sc baseline should not have layers before writing'
    sc_baseline.write_h5ad(os.path.join(OUT_DIR, 'baseline_sc.h5ad'), compression='gzip')
    print(f'baseline_sc shape: {sc_baseline.shape}', flush=True)

    print('Running bulkify (baseline_bulk, major cell types only)...', flush=True)
    bulk_group = get_config(DATASET).bulk_group
    bulk_baseline = bulkify_func(filter_rb_mt_genes(sc_baseline), covariates=bulk_group + [MAJOR_CT_LABEL])
    bulk_baseline = normalize(bulk_baseline)
    bulk_baseline = qc_bulk(bulk_baseline, run_test=False)
    bulk_baseline.write_h5ad(os.path.join(OUT_DIR, 'baseline_bulk.h5ad'), compression='gzip')
    print(f'baseline_bulk shape: {bulk_baseline.shape}', flush=True)


if __name__ == '__main__':
    main()
