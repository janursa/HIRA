"""Shared helpers for the tests/ reproducibility suite (one folder per pipeline stage).

Each stage test does two checks against a fixture in its own data/:
  - matches_baseline(): run stage once on the fixture input, diff vs committed baseline_*.
  - is_deterministic(): run stage twice on the same fixture input, diff the two outputs.

Single shared module (not one copy per stage folder): pytest's default rootdir import mode caches
modules by bare basename in sys.modules, so four identically-named `_repro_utils.py` files
collapse onto whichever one is imported first when stage folders are collected together.
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))  # tests/
HIRA_DIR = os.path.dirname(TESTS_DIR)  # repo root
PARENT_DIR = os.path.dirname(HIRA_DIR)  # needed on sys.path for `from hira.src... import`


def diff_dataframes(a: pd.DataFrame, b: pd.DataFrame, float_cols=(), atol=1e-8, sort_cols=None):
    """Return a list of human-readable mismatch descriptions; empty list means identical."""
    problems = []
    if sort_cols:
        a = a.sort_values(sort_cols).reset_index(drop=True)
        b = b.sort_values(sort_cols).reset_index(drop=True)
    if list(a.columns) != list(b.columns):
        problems.append(f'columns differ: {list(a.columns)} vs {list(b.columns)}')
        return problems
    if len(a) != len(b):
        problems.append(f'row count differs: {len(a)} vs {len(b)}')
        return problems
    for col in a.columns:
        if col in float_cols or pd.api.types.is_float_dtype(a[col]):
            ok = np.allclose(a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float), atol=atol, equal_nan=True)
        else:
            ok = a[col].astype(str).equals(b[col].astype(str))
        if not ok:
            n_diff = (~np.isclose(a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float), atol=atol, equal_nan=True)).sum() \
                if pd.api.types.is_numeric_dtype(a[col]) else (a[col].astype(str) != b[col].astype(str)).sum()
            problems.append(f'column "{col}" differs in {n_diff}/{len(a)} rows')
    return problems


def compare_csv_to_baseline(new_path, baseline_path, sort_cols=None, atol=1e-8):
    new = pd.read_csv(new_path)
    baseline = pd.read_csv(baseline_path)
    return diff_dataframes(new, baseline, sort_cols=sort_cols, atol=atol)


def run_module(module, args, env=None, cwd=None, timeout=1800):
    """Run `python -m module ...args` (or a script path) as a subprocess, raising on failure."""
    cmd = [sys.executable, module, *args]
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    result = subprocess.run(
        cmd, cwd=cwd or HIRA_DIR, env=full_env,
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(cmd)}\n--- stdout ---\n{result.stdout[-4000:]}\n--- stderr ---\n{result.stderr[-4000:]}"
        )
    return result


def compare_adata(new, baseline, obs_cols=(), atol=1e-8):
    """Compare two AnnData objects: var_names, X, and the given obs columns.

    Returns a list of human-readable mismatch descriptions; empty list means identical.
    """
    import numpy as np
    from scipy.sparse import issparse

    problems = []
    if list(new.var_names) != list(baseline.var_names):
        problems.append(f'var_names differ: {new.n_vars} vs {baseline.n_vars} genes')
        return problems
    if new.n_obs != baseline.n_obs:
        problems.append(f'n_obs differs: {new.n_obs} vs {baseline.n_obs}')
        return problems

    x_new = new.X.toarray() if issparse(new.X) else np.asarray(new.X)
    x_base = baseline.X.toarray() if issparse(baseline.X) else np.asarray(baseline.X)
    if not np.allclose(x_new, x_base, atol=atol, equal_nan=True):
        n_diff = int(np.sum(~np.isclose(x_new, x_base, atol=atol, equal_nan=True)))
        problems.append(f'X matrix differs in {n_diff}/{x_new.size} entries')

    for col in obs_cols:
        a = new.obs[col].astype(str).reset_index(drop=True)
        b = baseline.obs[col].astype(str).reset_index(drop=True)
        if not a.equals(b):
            n_diff = int((a != b).sum())
            problems.append(f'obs["{col}"] differs in {n_diff}/{len(a)} rows')
    return problems


def sandbox_env(sandbox_root):
    """Env overrides that point HIRA_DIR/HIRA_BASE_DIR at a self-contained sandbox directory.

    sandbox_root must contain: datasets/, prior/, results_folder/ (feature matrices under results_folder/features/), (grns/ symlinked
    to the real repo's committed GRN networks, features/ writable & disposable).
    """
    return {'HIRA_DIR': sandbox_root, 'HIRA_BASE_DIR': sandbox_root}


def prepare_sandbox_run(template_dir, run_root):
    """Symlink the read-only fixture data from a committed sandbox template into a fresh,
    per-run root with its own empty results_folder/, so parallel/repeated runs don't collide.

    'src' is never part of the committed template (it's an absolute-path, machine-specific
    symlink — see make_feature_fixture.py) so it's linked here straight from the real repo
    instead, needed for lookups like the meta-analysis R script.
    """
    os.makedirs(run_root, exist_ok=True)
    for name in ('prior', 'datasets'):
        src = os.path.join(template_dir, name)
        if os.path.isdir(src) or os.path.islink(src):
            os.symlink(os.path.realpath(src), os.path.join(run_root, name))
    os.symlink(os.path.join(HIRA_DIR, 'src'), os.path.join(run_root, 'src'))
    os.makedirs(os.path.join(run_root, 'results_folder'), exist_ok=True)
    # feature matrices sit next to their (writable) stats/ in results_folder/features/<analysis>/,
    # so link the files, not the directory, to keep run output out of the template
    features = os.path.join(template_dir, 'features')
    for analysis in (os.listdir(features) if os.path.isdir(features) else []):
        dst = os.path.join(run_root, 'results_folder', 'features', analysis)
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(os.path.join(features, analysis)):
            os.symlink(os.path.realpath(os.path.join(features, analysis, f)), os.path.join(dst, f))
    return sandbox_env(run_root)
