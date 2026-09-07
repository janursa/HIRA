# tests_code/

Regression + determinism checks, one folder per pipeline stage. Every stage runs the same two
checks against a git-lfs fixture:

- `matches_baseline()` — rerun the stage on the fixture input, diff against the committed baseline
- `is_deterministic()` — run it twice, diff the two runs against each other

| Folder | Stage under test | Fixture |
|---|---|---|
| `preprocessing/` | QC, cell type annotation, bulkify — plus a freshness check that the raw input files and priors haven't drifted | zhang raw data subset to 4 donors |
| `grn_inference/` | GRN inference, parametrized over the 6 cell types with a non-empty fixture network | top-300 genes of `baseline_sc.h5ad` |
| `feature_analysis/` | feature association, parametrized over 6 cohorts (aging + 5 condition) | downsampled precomputed TF activity per cohort |
| `clocks/` | clock training, parametrized over CD4T and CD8T | bulkified zhang, donors split into a synthetic 2-group `dataset` label so leave-one-cohort-out CV has folds |

Each folder holds `fixtures/make_*.py` (regenerates the fixture), the `test_*.py`, and a `data/`
symlink into `tests_data/` (git-lfs) where the heavy files actually live.

Two root-level tests are not stage regressions:

- `test_facts.py` — every number quoted in `manuscript/manuscript.md` (cohort sizes, TF counts,
  directional claims) still matches what the pipeline produces.
- `test_smoke_pipeline.py` — import sanity plus a synthetic `basic_qc -> bulkify_func` run.
  Needs no real datasets.

`_repro_utils.py` (diffing, csv/adata comparison, sandboxed subprocess runner) lives once at the
root, not per-folder — without `__init__.py` files pytest caches same-named modules by basename,
so per-folder copies collided when all stages were collected in one run.

```bash
python -m pytest tests_code/preprocessing tests_code/grn_inference tests_code/feature_analysis tests_code/clocks
python -m pytest tests_code/<stage>/   # one stage
```

Needs the `py10` conda env — the `hira` env is missing an editable-install path shim that
`from hira.src... import` relies on.
