"""Feature association stage: run_analysis.py --skip-features on downsampled precomputed
TF-activity matrices, one per cohort from scripts/feature_analysis.sh (aging + 5 condition
cohorts). --skip-features means only the stats-computation step (mixed-effects / grouped
tests in wrapper_association_with_age_condition) is exercised, not TF-activity computation.
"""
import os
import sys

import pytest

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))  # tests/feature_analysis/
DATA_DIR = os.path.join(STAGE_DIR, 'data')
sys.path.insert(0, os.path.dirname(STAGE_DIR))
from _repro_utils import prepare_sandbox_run, run_module, compare_csv_to_baseline  # noqa: E402

sys.path.insert(0, os.path.join(STAGE_DIR, 'fixtures'))
from make_feature_fixture import COHORTS  # noqa: E402

FEATURE_DATA_DIR = os.path.join(DATA_DIR, 'feature_association')


def _run_cohort(cohort, cfg, run_root):
    env = prepare_sandbox_run(os.path.join(FEATURE_DATA_DIR, cohort, 'sandbox'), run_root)
    args = [
        '--analysis-mode', cfg['analysis_mode'],
        '--analysis-name', 'tfa_major_b',
        '--datasets', *cfg['datasets'],
        '--association-type', cfg['association_type'],
        '--skip-features',
    ]
    if cfg['cell_types']:
        args += ['--cell-types', *cfg['cell_types']]
    run_module('src/feature_association/run_analysis.py', args, env=env)

    if cfg['analysis_mode'] == 'multi-cohort':
        stats_name = 'stats_multi_cohort.csv'
    else:
        stats_name = f"stats_{cfg['datasets'][0]}.csv"
    return os.path.join(run_root, 'results_folder', 'features', 'tfa_major_b', 'stats', stats_name)


@pytest.mark.parametrize('cohort', list(COHORTS.keys()))
def test_stats_match_baseline(cohort, tmp_path):
    stats_path = _run_cohort(cohort, COHORTS[cohort], str(tmp_path / 'run'))
    baseline_path = os.path.join(FEATURE_DATA_DIR, cohort, 'baseline_stats.csv')
    problems = compare_csv_to_baseline(stats_path, baseline_path, sort_cols=['gene', 'cell_type'])
    assert not problems, f'{cohort}: stats drifted from baseline: ' + '; '.join(problems)


@pytest.mark.parametrize('cohort', list(COHORTS.keys()))
def test_stats_are_deterministic(cohort, tmp_path):
    cfg = COHORTS[cohort]
    stats1 = _run_cohort(cohort, cfg, str(tmp_path / 'run1'))
    stats2 = _run_cohort(cohort, cfg, str(tmp_path / 'run2'))
    problems = compare_csv_to_baseline(stats1, stats2, sort_cols=['gene', 'cell_type'])
    assert not problems, f'{cohort}: two runs on identical input produced different stats: ' + '; '.join(problems)
