"""Aging clock stage: grnimmuneclock.train_aging_clock on data/clocks/input_<CT>.h5ad.

Unlike GRN inference and feature association (both seeded end-to-end), aging-clock training goes
through tune_ridge_params (GRNimmuneClock/grnimmuneclock/training.py), which tunes Ridge's alpha
via optuna.create_study(direction="maximize") with NO sampler seed — so BOTH checks below are
EXPECTED TO FAIL as of this writing. That's the known gap already flagged in reproducibility.md
(training.py:104, TUNE_CLOCK=True by default). This suite exists to make that gap visible and
regression-checkable, not to assert the clock is currently reproducible.
"""
import glob
import os
import sys

import anndata as ad
import pytest

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))  # tests/clocks/
DATA_DIR = os.path.join(STAGE_DIR, 'data')
sys.path.insert(0, os.path.dirname(STAGE_DIR))
from _repro_utils import PARENT_DIR, compare_csv_to_baseline  # noqa: E402

sys.path.insert(0, PARENT_DIR)
from grnimmuneclock import train_aging_clock  # noqa: E402
from hira.src.config import CLOCK_CV_SCORING, TUNE_CLOCK  # noqa: E402

CLOCK_DATA_DIR = os.path.join(DATA_DIR, 'clocks')
CELL_TYPES = sorted(
    os.path.basename(f).removeprefix('input_').removesuffix('.h5ad')
    for f in glob.glob(os.path.join(CLOCK_DATA_DIR, 'input_*.h5ad'))
)


def _train(cell_type, output_dir):
    adata = ad.read_h5ad(os.path.join(CLOCK_DATA_DIR, f'input_{cell_type}.h5ad'))
    _, _, trained = train_aging_clock(
        adata=adata, cell_type=cell_type, version='test',
        output_dir=output_dir, reg_type='ridge',
        tune_model=TUNE_CLOCK, scoring=CLOCK_CV_SCORING, verbose=False,
    )
    preds = trained.obs[['dataset', 'donor_id', 'age', 'predicted_age']].sort_values(['dataset', 'donor_id'])
    path = os.path.join(output_dir, f'predictions_{cell_type}.csv')
    preds.to_csv(path, index=False)
    return path


@pytest.mark.parametrize('cell_type', CELL_TYPES)
def test_predictions_match_baseline(cell_type, tmp_path):
    new_path = _train(cell_type, str(tmp_path))
    baseline_path = os.path.join(CLOCK_DATA_DIR, f'baseline_predictions_{cell_type}.csv')
    problems = compare_csv_to_baseline(new_path, baseline_path, sort_cols=['dataset', 'donor_id'])
    assert not problems, (
        f'{cell_type}: predictions drifted from baseline (known gap: unseeded Optuna tuning in '
        'grnimmuneclock/training.py, see reproducibility.md): ' + '; '.join(problems)
    )


@pytest.mark.parametrize('cell_type', CELL_TYPES)
def test_predictions_are_deterministic(cell_type, tmp_path):
    p1 = _train(cell_type, str(tmp_path / 'run1'))
    p2 = _train(cell_type, str(tmp_path / 'run2'))
    problems = compare_csv_to_baseline(p1, p2, sort_cols=['dataset', 'donor_id'])
    assert not problems, (
        f'{cell_type}: two runs on identical input produced different predictions (known gap: '
        'unseeded Optuna tuning in grnimmuneclock/training.py, see reproducibility.md): ' + '; '.join(problems)
    )
