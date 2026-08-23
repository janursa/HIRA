# tests/

Regression + determinism checks for the pipeline, split one folder per stage. Each stage does
two checks against a fixture committed via git-lfs: matches_baseline() (rerun vs committed
baseline) and is_deterministic() (two reruns diffed against each other). See `reproducibility.md`
at repo root for the design rationale. `_repro_utils.py` (shared helpers: diffing, csv/adata
comparison, sandboxed subprocess runner) lives once at the tests/ root, not per-folder — pytest
caches same-named modules by basename when there's no `__init__.py`, so per-folder copies
silently collided when all four stages were collected together in one run.

```
tests/
├── _repro_utils.py                      shared helpers used by every stage below
├── preprocessing/                       raw-data freshness + preprocessing stage
│   ├── fixtures/
│   │   ├── make_raw_manifest.py         snapshots size+mtime of every raw dataset file + prior/ -> data/raw_manifest.json
│   │   └── make_preprocessing_fixture.py  subsets zhang raw data to 4 donors, runs QC+annotation+bulkify -> data/preprocessing/*.h5ad
│   ├── test_raw_data_freshness.py       checks raw files (+prior) haven't changed and the manifest itself isn't stale
│   ├── test_preprocessing_repro.py      checks QC/annotation/bulkify output matches baseline and is deterministic across reruns
│   └── data/                            raw_manifest.json; preprocessing/{input,baseline_sc,baseline_bulk}.h5ad
├── grn_inference/                       GRN inference stage
│   ├── fixtures/make_grn_fixture.py     runs GRN inference (top-300 genes) on ../../preprocessing/data/preprocessing/baseline_sc.h5ad -> data/grn/baseline_net_<CT>.csv
│   ├── test_grn_repro.py                matches-baseline + determinism, parametrized over 7 cell types
│   └── data/grn/                        baseline_net_<CT>.csv per cell type
├── feature_analysis/                    feature-association stage
│   ├── fixtures/make_feature_fixture.py downsamples precomputed TF-activity per cohort into a sandbox, runs run_analysis.py --skip-features -> data/feature_association/<cohort>/
│   ├── test_feature_association_repro.py  matches-baseline + determinism, parametrized over 6 cohorts (aging + 5 condition cohorts)
│   └── data/feature_association/<cohort>/  sandbox/ (self-contained mini pipeline input) + baseline_stats.csv
└── clocks/                              aging-clock training stage
    ├── fixtures/make_clock_fixture.py   real bulkified zhang fixture (borrowed from preprocessing/), donors split into a synthetic 2-group 'dataset' label so leave-one-cohort-out CV has something to fold over -> data/clocks/{input_<CT>.h5ad, baseline_predictions_<CT>.csv}
    ├── test_clock_repro.py              matches-baseline + determinism, parametrized over 2 cell types (CD4T, CD8T)
    └── data/clocks/                     input_<CT>.h5ad + baseline_predictions_<CT>.csv
```

Run all: `python -m pytest tests_code/preprocessing tests_code/grn_inference tests_code/feature_analysis tests_code/clocks`
(or any subset). Run one stage: `python -m pytest tests_code/<stage>/`. Needs the `py10` conda env —
`hira` env is missing an editable-install path shim that `from hira.src... import` relies on.

Heavy fixture data (`data/` under each stage) lives in `tests_data/` (git-lfs) and is symlinked
back into `tests_code/<stage>/data`, so code stays lightweight and data stays out of normal git
history.

Last full run: 36 passed, 0 failed.

## clocks: fixed unseeded Optuna tuning

`grnimmuneclock.train_aging_clock` (used with `TUNE_CLOCK=True`, the production default) tunes
Ridge's alpha via Optuna in `GRNimmuneClock/grnimmuneclock/training.py`. It originally had no
sampler seed, so `test_clock_repro.py` failed 4/4 (both matches-baseline checks, both determinism
checks, both cell types) — predicted ages differed between runs on identical input. Fixed by
passing `optuna.samplers.TPESampler(seed=42)` into `create_study`; all 4 clock tests now pass.

## raw-data freshness: prior watch trimmed

`make_raw_manifest.py` used to glob every file under `PRIOR_DIR`, which made
`test_raw_files_unchanged` trip on unrelated backup files (e.g.
`tf_all.csv.bak_pre_lambert_*`) getting cleaned up in production — not real drift the pipeline
cares about. It now watches only the two files the pipeline actually reads: `gene_names.txt` and
`tf_all.csv`.
