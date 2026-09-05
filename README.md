# HIRA

Analysis pipeline for immune aging: cell-type-resolved GRN inference, aging clocks,
feature association, pathway/motif/trajectory analysis.

Stages run in order — priors, raw data, preprocessing, GRN inference, feature association,
clocks, supplementary figures. `scripts/readme.md` is the one-table index of all of them.

## Setup

```bash
git clone --recurse-submodules git@github.com:janursa/ciim.git hira
cd hira
conda create -n hira python=3.10 -y
conda activate hira
pip install -r requirements.txt
pip install -e GRNimmuneClock/
```

The `hira` package is the repo directory itself (not a package inside it), so its
*parent* directory must be on `PYTHONPATH`:

```bash
export PYTHONPATH="$(dirname "$(pwd)"):$PYTHONPATH"
```

Verify with `python -c "import hira"`.

## Configuration

Set these in `.env`

- `HIRA_DIR` — repo root. Required, no default.
- `HIRA_BASE_DIR` — where heavy data (datasets/priors/per-cell feature matrices) lives,
  outside the repo. Required, no default. Lightweight results (GRNs, summary stats,
  clock models, plots) are git-tracked and always live in `<HIRA_DIR>/results_folder`.
- `HIRA_RAW_DIR` — root of the raw data lake that preprocessing reads cohort input
  files from (see Data Acquisition below). Defaults to `/vol/projects/CIIM`.


## Data Acquisition

Preprocessing (`scripts/process_data/run_preprocess.sh`) reads raw per-cohort files from
`$HIRA_RAW_DIR` (default `/vol/projects/CIIM`). To fetch a cohort's raw data, or
print manual access instructions where a download can't be automated:

```bash
bash scripts/process_data/acquire/download_data.sh <cohort>
```

| Cohort | Access | Source |
|---|---|---|
| OneK1K (`onek1k`) | direct download | [CELLxGENE collection](https://cellxgene.cziscience.com/collections/dde06e0f-ab3b-46be-96a2-a8082383c4a1) |
| Perez SLE (`perez_sle`) | direct download | [CELLxGENE collection](https://cellxgene.cziscience.com/collections/436154da-bcf1-4130-9c8b-120ff9a888f2) |
| AIDA (`aida`) | direct download | [CELLxGENE collection](https://cellxgene.cziscience.com/collections/ced320a1-29f3-47c1-a735-513c7084d508) (Freeze v1) |
| ParseBioscience (`parsebioscience`) | direct download | Parse Biosciences S3 bucket |
| ABF300 (`abf300`) | manual, gated | Synapse `syn49637038` (account + data use agreement required) |
| Zhang (`zhang`) | manual, gated | Synapse `syn61609846` (account + data use agreement required) |
| OPSCA (`op`) | manual, gated | Kaggle competition `open-problems-single-cell-perturbations` (account + API token required) |
| SoundLife (`soundlife`) | manual, private | not publicly hosted — obtained via direct data transfer from study authors |
| CXCL9 (`CXCL9`) | internal | CIIM-only, no public source |

To use raw files from a location other than `$HIRA_RAW_DIR`'s default layout,
either set `HIRA_RAW_DIR` in `.env`, or set `INPUT_FILE_OVERRIDE` when invoking
`run_preprocess.sh` to point at a single custom path. Downloaded files may need light
column-name harmonization (donor/age/condition fields) to match what
`src/process_data/preprocess/helper.py:format_data` expects — it already
recognizes several common CELLxGENE/Synapse schema variants.

## Input data
Once raw data downloaded, preprocessing scripts does quality control, cell type annotation, and pseudobulking and
write them under `<HIRA_BASE_DIR>/datasets/{sc,bulk,bulk_minor,metacell}/<dataset>.h5ad`.

## GRN inference

Infers per-cell-type gene regulatory networks from the datasets under `<HIRA_BASE_DIR>/datasets/`.
Edit the dataset list and paths in `scripts/grn_inference/wrapper_grn_inference.sh`, then submit:

```bash
bash scripts/grn_inference/wrapper_grn_inference.sh
```

This `sbatch`-submits one SLURM job per dataset via `src/grn_inference/run_grn_inference.sh`.

## Age-associated TF activity and gene expression analysis

Computes features (TF activity, gene expression, etc.) and tests their association with age/condition, aggregating across cohorts via meta-analysis where applicable.

```bash
bash scripts/feature_association/wrapper_feature_analysis.sh [analysis_name] [task ...]
```

This `sbatch`-submits one SLURM job per task (`aging soundlife perez_sle parsebioscience op
CXCL9`), so they run in parallel; `il10_ruxolitinib` is chained after `op` and
`parsebioscience` since it reads their results. Consensus GRNs are built once by the wrapper
before submitting. Pass task names to submit only a subset, and re-run the wrapper per
feature type (e.g. once with `tfa_major_mc`, once with `ge_major_mc`). The `ge_*` feature
types only support `aging` — the condition/perturbation plots are unimplemented for them, so
the wrapper submits `aging` alone for those.

`analysis_name` (default `tfa_major_mc`) selects the feature type and the data it runs on; the
full vocabulary is `CONFIG_FA` in `src/config.py`.

## Aging clocks

Trains the per-cell-type Ridge clocks (via the `GRNimmuneClock` submodule), runs
cross-validation, compares against published clocks, and applies them to the disease and
perturbation cohorts.

```bash
bash scripts/clock_analysis.sh
```

Relevant config: `CLOCK_V` (model version), `TUNE_CLOCK` (Optuna alpha search),
`CLOCK_CV_SCORING`, `CLOCK_TRAINING_COHORTS`. Trained models land in `results_folder/clock/`
and are copied into the package by `python src/clock/build_package_data.py`, which also
regenerates the consensus GRNs and example dataset shipped with `GRNimmuneClock`.

## Supplementary figures

Cohort composition stats, GRN overlap plots, and the discovery-vs-validation TF tables:

```bash
bash scripts/supp_figs.sh
```

## Results layout

`results_folder/` is git-tracked and holds everything lightweight:

- `grns/` — per-cohort and consensus GRNs
- `features/` — association statistics and supplementary tables
- `clock/` — trained clock models and predictions
- `plots/` — all figures, `plots/assembled/` for multi-panel manuscript figures


## License

MIT, see `LICENSE`.
