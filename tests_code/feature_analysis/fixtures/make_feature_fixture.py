"""Run once by hand to (re)generate data/feature_association/<cohort>/{sandbox/, baseline_stats.csv}.

For each cohort (mirrors scripts/feature_analysis.sh's 6 invocations), downsamples the real
precomputed TF-activity matrices (results_folder/features/tfa_major_b/{dataset}_{cell_type}.h5ad)
into a small self-contained "sandbox" directory, then runs the real run_analysis.py --skip-features
against that sandbox (via HIRA_DIR/HIRA_BASE_DIR env override) to produce the baseline stats CSV.

--skip-features means wrapper_tf_activity (which needs raw counts + GRNs) never runs; only the
stats-computation step (wrapper_association_with_age_condition / wrapper_meta_analysis) is
exercised — that's the part we're checking for reproducibility.

Usage: python make_feature_fixture.py [cohort ...]   (default: all cohorts)
"""
import argparse
import os
import shutil
import tempfile
import sys

import anndata as ad

REPRO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tests/feature_analysis/
HIRA_DIR = os.path.dirname(os.path.dirname(REPRO_DIR))  # repo root
sys.path.insert(0, os.path.dirname(HIRA_DIR))  # needed for `from hira.src... import`
sys.path.insert(0, os.path.dirname(REPRO_DIR))  # tests/, to find _repro_utils.py

from hira.src.config import MAJOR_CTS, FEATURE_DATA_DIR, PRIOR_DIR  # noqa: E402
from _repro_utils import run_module, prepare_sandbox_run  # noqa: E402

REAL_TFA_DIR = f'{FEATURE_DATA_DIR}/tfa_major_b'
REAL_PRIOR_DIR = PRIOR_DIR
DATA_DIR = os.path.join(REPRO_DIR, 'data', 'feature_association')

N_PER_GROUP = 20   # grouped (disease/perturbation): rows kept per condition
N_CONTINUOUS = 40  # continuous (aging): rows kept total
SEED = 0

COHORTS = {
    'aging': dict(analysis_mode='multi-cohort', datasets=['aida', 'perez_sle', 'onek1k', 'abf300'],
                  association_type='continous', cell_types=None),
    'soundlife': dict(analysis_mode='single-cohort', datasets=['soundlife'],
                       association_type='continous', cell_types=None),
    'perez_sle': dict(analysis_mode='single-cohort', datasets=['perez_sle'],
                       association_type='grouped', cell_types=None),
    'parsebioscience': dict(analysis_mode='single-cohort', datasets=['parsebioscience'],
                             association_type='grouped', cell_types=['CD4T', 'CD8T']),
    'op': dict(analysis_mode='single-cohort', datasets=['op'],
               association_type='grouped', cell_types=['CD4T']),
    'cxcl9': dict(analysis_mode='single-cohort', datasets=['CXCL9'],
                   association_type='grouped', cell_types=['CD4T', 'CD8T']),
}


def downsample_tfa(dataset, cell_type, association_type):
    src = f'{REAL_TFA_DIR}/{dataset}_{cell_type}.h5ad'
    adata = ad.read_h5ad(src)
    if association_type == 'grouped':
        parts = []
        for _, grp in adata.obs.groupby('condition'):
            n = min(N_PER_GROUP, len(grp))
            parts.append(grp.sample(n=n, random_state=SEED))
        import pandas as pd
        keep_idx = pd.concat(parts).index
    else:
        n = min(N_CONTINUOUS, adata.n_obs)
        keep_idx = adata.obs.sample(n=n, random_state=SEED).index
    return adata[keep_idx].copy()


def build_sandbox(cohort, cfg):
    sandbox = os.path.join(DATA_DIR, cohort, 'sandbox')
    if os.path.exists(sandbox):
        shutil.rmtree(sandbox)
    os.makedirs(os.path.join(sandbox, 'features', 'tfa_major_b'))
    os.makedirs(os.path.join(sandbox, 'prior'))
    for f in ['gene_names.txt', 'tf_all.csv']:
        shutil.copy(os.path.join(REAL_PRIOR_DIR, f), os.path.join(sandbox, 'prior', f))

    cell_types = cfg['cell_types'] or MAJOR_CTS
    for dataset in cfg['datasets']:
        for ct in cell_types:
            src = f'{REAL_TFA_DIR}/{dataset}_{ct}.h5ad'
            if not os.path.exists(src):
                continue
            small = downsample_tfa(dataset, ct, cfg['association_type'])
            small.write_h5ad(os.path.join(sandbox, 'features', 'tfa_major_b', f'{dataset}_{ct}.h5ad'))
    return sandbox


def run_fixture(cohort, cfg, sandbox):
    run_root = tempfile.mkdtemp(dir=os.path.join(HIRA_DIR, 'temp'))
    env = prepare_sandbox_run(sandbox, run_root)
    args = [
        '--analysis-mode', cfg['analysis_mode'],
        '--analysis-name', 'tfa_major_b',
        '--datasets', *cfg['datasets'],
        '--association-type', cfg['association_type'],
        '--skip-features',
    ]
    if cfg['cell_types']:
        args += ['--cell-types', *cfg['cell_types']]
    run_module('src/feature_association/run_analysis.py', args, env=env, cwd=HIRA_DIR)

    stats_name = 'stats_multi_cohort.csv' if cfg['analysis_mode'] == 'multi-cohort' else f"stats_{cfg['datasets'][0]}.csv"
    shutil.copy(os.path.join(run_root, 'results_folder', 'features', 'tfa_major_b', 'stats', stats_name),
                os.path.join(DATA_DIR, cohort, 'baseline_stats.csv'))
    shutil.rmtree(run_root)  # disposable output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cohorts', nargs='*', default=list(COHORTS.keys()))
    args = parser.parse_args()

    for cohort in args.cohorts:
        cfg = COHORTS[cohort]
        print(f'=== {cohort} ===', flush=True)
        os.makedirs(os.path.join(DATA_DIR, cohort), exist_ok=True)
        sandbox = build_sandbox(cohort, cfg)
        run_fixture(cohort, cfg, sandbox)
        print(f'{cohort}: baseline_stats.csv written', flush=True)


if __name__ == '__main__':
    main()
