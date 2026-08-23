"""Run once by hand to (re)generate data/clocks/{input_<CT>.h5ad, baseline_predictions_<CT>.csv}.

Reuses ../../preprocessing/data/preprocessing/baseline_bulk.h5ad (real bulkified zhang fixture,
already committed for the preprocessing stage) instead of pulling from the live production
HIRA_BASE_DIR bulk datasets (mid-reorganization at the time this was written, not at their
configured paths). Real expression + age values; the donors are split into a synthetic 'dataset'
label purely so leave-one-cohort-out CV (which train_aging_clock requires) has >=2 groups —
not a real second cohort.
# ponytail: thin fixture (2 donors/group), fine for a reproducibility check since we're diffing
# reruns against each other, not judging model quality.

Trains with tune_model=TUNE_CLOCK (True by default, src/config.py) via the real
grnimmuneclock.train_aging_clock, so this exercises the exact production code path — including
its unseeded Optuna hyperparameter search (grnimmuneclock/training.py's tune_ridge_params) —
that reproducibility.md flags as a known gap.
"""
import os
import shutil
import sys

import anndata as ad

REPRO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tests/clocks/
sys.path.insert(0, os.path.dirname(REPRO_DIR))  # tests/, to find _repro_utils.py
from _repro_utils import PARENT_DIR  # noqa: E402

sys.path.insert(0, PARENT_DIR)
from grnimmuneclock import train_aging_clock  # noqa: E402
from hira.src.config import CLOCK_CV_SCORING, TUNE_CLOCK  # noqa: E402

SRC_BULK = os.path.join(os.path.dirname(REPRO_DIR), 'preprocessing', 'data', 'preprocessing', 'baseline_bulk.h5ad')
OUT_DIR = os.path.join(REPRO_DIR, 'data', 'clocks')
CELL_TYPES = ['CD4T', 'CD8T']  # the two Major_CTs with enough bulk rows (4) in the fixture to split


def build_input(cell_type):
    adata = ad.read_h5ad(SRC_BULK)
    adata = adata[adata.obs['Major_CT'] == cell_type].copy()
    donors = sorted(adata.obs['donor_id'].unique())
    half = len(donors) // 2
    group_map = {d: ('fixture_a' if i < half else 'fixture_b') for i, d in enumerate(donors)}
    adata.obs['dataset'] = adata.obs['donor_id'].map(group_map)
    return adata


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    models_dir = os.path.join(OUT_DIR, '_models')

    for cell_type in CELL_TYPES:
        print(f'=== {cell_type} ===', flush=True)
        adata = build_input(cell_type)
        adata.write_h5ad(os.path.join(OUT_DIR, f'input_{cell_type}.h5ad'))

        _, _, trained = train_aging_clock(
            adata=adata.copy(), cell_type=cell_type, version='baseline',
            output_dir=models_dir, reg_type='ridge',
            tune_model=TUNE_CLOCK, scoring=CLOCK_CV_SCORING, verbose=False,
        )
        preds = trained.obs[['dataset', 'donor_id', 'age', 'predicted_age']].sort_values(['dataset', 'donor_id'])
        preds.to_csv(os.path.join(OUT_DIR, f'baseline_predictions_{cell_type}.csv'), index=False)
        print(f'{cell_type}: baseline_predictions_{cell_type}.csv written', flush=True)

    shutil.rmtree(models_dir, ignore_errors=True)  # throwaway model artifacts, not committed


if __name__ == '__main__':
    main()
