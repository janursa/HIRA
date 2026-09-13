"""GRN inference stage: basic_qc + main_inference on data/preprocessing/baseline_sc.h5ad,
one cell type at a time (mirrors wrapper_grn in src/grn_inference/script.py).

Spearman correlation + BH-FDR has no randomness once the per-bulk_group sampling is seeded
(random_state=0, already the case in production), so both checks are expected to PASS —
a failure here would be a real regression, unlike the preprocessing stage's celltype-label test.
"""
import glob
import os
import sys

import anndata as ad
import pytest

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))  # tests/grn_inference/
DATA_DIR = os.path.join(STAGE_DIR, 'data')
sys.path.insert(0, os.path.dirname(STAGE_DIR))
from _repro_utils import compare_csv_to_baseline  # noqa: E402

sys.path.insert(0, os.path.join(STAGE_DIR, 'fixtures'))
from make_grn_fixture import infer_for_cell_type, INPUT_PATH  # noqa: E402
from hira.src.config import MAJOR_CT_LABEL  # noqa: E402

GRN_BASELINE_DIR = os.path.join(DATA_DIR, 'grn')
CELL_TYPES = sorted(
    os.path.basename(f).removeprefix('baseline_net_').removesuffix('.csv')
    for f in glob.glob(os.path.join(GRN_BASELINE_DIR, 'baseline_net_*.csv'))
)


@pytest.fixture(scope='module')
def adata_full():
    return ad.read_h5ad(INPUT_PATH)


@pytest.mark.parametrize('cell_type', CELL_TYPES)
def test_grn_matches_baseline(adata_full, cell_type):
    net = infer_for_cell_type(adata_full.copy(), cell_type, 'bulk_group')
    assert net is not None, f'{cell_type}: no network inferred from fixture (too few cells/genes after QC)'
    new_path = os.path.join(DATA_DIR, f'_tmp_net_{cell_type}.csv')
    net.to_csv(new_path, index=False)
    try:
        problems = compare_csv_to_baseline(
            new_path, os.path.join(GRN_BASELINE_DIR, f'baseline_net_{cell_type}.csv'),
            sort_cols=['source', 'target'],
        )
    finally:
        os.remove(new_path)
    assert not problems, f'{cell_type}: GRN drifted from baseline: ' + '; '.join(problems)


@pytest.mark.parametrize('cell_type', CELL_TYPES)
def test_grn_is_deterministic(adata_full, cell_type):
    net1 = infer_for_cell_type(adata_full.copy(), cell_type, 'bulk_group')
    net2 = infer_for_cell_type(adata_full.copy(), cell_type, 'bulk_group')
    assert net1 is not None and net2 is not None
    p1 = os.path.join(DATA_DIR, f'_tmp_net1_{cell_type}.csv')
    p2 = os.path.join(DATA_DIR, f'_tmp_net2_{cell_type}.csv')
    net1.to_csv(p1, index=False)
    net2.to_csv(p2, index=False)
    try:
        problems = compare_csv_to_baseline(p1, p2, sort_cols=['source', 'target'])
    finally:
        os.remove(p1)
        os.remove(p2)
    assert not problems, f'{cell_type}: two runs on identical input produced different GRNs: ' + '; '.join(problems)
