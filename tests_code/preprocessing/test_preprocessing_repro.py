"""Preprocessing stage: basic_qc + annotate_celltypes (+ bulkify) on data/preprocessing/input.h5ad.

test_*_matches_baseline: rerun once, diff vs the committed baseline_*.h5ad.
test_*_is_deterministic: rerun twice on the same input, diff the two outputs.

A failure on Major_CT/Sub_CT columns is a KNOWN, expected finding, not a new bug: CellTypist's
majority-voting over-clustering step (annotate_celltypist_majority_voting in
src/process_data/preprocess/helper.py) has no fixed random_state, so cell-type labels are not
guaranteed stable across reruns. See reproducibility.md. X/var_names/donor columns are expected
to match exactly since nothing upstream of annotation is stochastic.
"""
import os
import sys

import anndata as ad
import pytest
import scanpy as sc

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))  # tests/preprocessing/
DATA_DIR = os.path.join(STAGE_DIR, 'data')
sys.path.insert(0, os.path.dirname(STAGE_DIR))
from _repro_utils import PARENT_DIR, compare_adata  # noqa: E402

sys.path.insert(0, PARENT_DIR)
from hira.src.process_data.preprocess.helper import basic_qc, annotate_celltypes  # noqa: E402
from hira.src.utils.util import bulkify_func, filter_rb_mt_genes  # noqa: E402
from hira.src.config import get_config, MAJOR_CT_LABEL  # noqa: E402
from hira.src.process_data.bulkify.script import normalize, qc_bulk  # noqa: E402

FIXTURE_DIR = os.path.join(DATA_DIR, 'preprocessing')
DATASET = 'wang'
DETERMINISTIC_OBS_COLS = ['donor_id', 'age', 'bulk_group']
CELLTYPE_OBS_COLS = ['Major_CT', 'Sub_CT']


def _run_preprocessing():
    adata = ad.read_h5ad(os.path.join(FIXTURE_DIR, 'input.h5ad'))
    adata = basic_qc(adata, run_test=False)
    adata = annotate_celltypes(adata, DATASET)
    sc.pp.filter_genes(adata, min_counts=1)
    return adata


def _run_bulkify(sc_adata):
    bulk_group = get_config(DATASET).bulk_group
    bulk = bulkify_func(filter_rb_mt_genes(sc_adata.copy()), covariates=bulk_group + [MAJOR_CT_LABEL])
    bulk = normalize(bulk)
    bulk = qc_bulk(bulk, run_test=False)
    return bulk


@pytest.fixture(scope='module')
def sc_baseline():
    return ad.read_h5ad(os.path.join(FIXTURE_DIR, 'baseline_sc.h5ad'))


@pytest.fixture(scope='module')
def bulk_baseline():
    return ad.read_h5ad(os.path.join(FIXTURE_DIR, 'baseline_bulk.h5ad'))


def test_sc_matches_baseline_deterministic_columns(sc_baseline):
    new = _run_preprocessing()
    problems = compare_adata(new, sc_baseline, obs_cols=DETERMINISTIC_OBS_COLS)
    assert not problems, 'preprocessing output drifted from baseline: ' + '; '.join(problems)


def test_sc_matches_baseline_celltype_labels(sc_baseline):
    new = _run_preprocessing()
    problems = compare_adata(new, sc_baseline, obs_cols=CELLTYPE_OBS_COLS)
    assert not problems, (
        'cell-type labels drifted from baseline (known gap: unseeded CellTypist majority-voting '
        'over-clustering, see reproducibility.md): ' + '; '.join(problems)
    )


def test_bulk_matches_baseline(sc_baseline, bulk_baseline):
    new_bulk = _run_bulkify(sc_baseline)
    problems = compare_adata(new_bulk, bulk_baseline, obs_cols=['donor_id', 'age', MAJOR_CT_LABEL, 'cell_count'])
    assert not problems, 'bulkify output drifted from baseline: ' + '; '.join(problems)


def test_sc_is_deterministic():
    run1 = _run_preprocessing()
    run2 = _run_preprocessing()
    problems = compare_adata(run1, run2, obs_cols=DETERMINISTIC_OBS_COLS + CELLTYPE_OBS_COLS)
    assert not problems, (
        'two runs on the identical input produced different output (known gap: unseeded '
        'CellTypist majority-voting over-clustering, see reproducibility.md): ' + '; '.join(problems)
    )
