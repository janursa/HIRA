# HIRA

Analysis pipeline for immune aging: cell-type-resolved GRN inference, aging clocks,
feature association, pathway/motif/trajectory analysis.

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

Preprocessing (`scripts/process_data/run_main.sh`) reads raw per-cohort files from
`$HIRA_RAW_DIR` (default `/vol/projects/CIIM`). To fetch a cohort's raw data, or
print manual access instructions where a download can't be automated:

```bash
bash scripts/process_data/download_data.sh <cohort>
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
`run_main.sh` to point at a single custom path. Downloaded files may need light
column-name harmonization (donor/age/condition fields) to match what
`src/process_data/preprocess/helper.py:format_data` expects — it already
recognizes several common CELLxGENE/Synapse schema variants.

## Input data
Once raw data downloaded, preprocessing scripts does quality control, cell type annotation, and pseudobulking and
write them under `<HIRA_BASE_DIR>/datasets/{sc,bulk,bulk_minor,metacell}/<dataset>.h5ad`.

## GRN inference

Infers per-cell-type gene regulatory networks from the datasets under `<HIRA_BASE_DIR>/datasets/`.
Edit the dataset list and paths in `scripts/wrapper_grn_inference.sh`, then submit:

```bash
bash scripts/wrapper_grn_inference.sh
```

This `sbatch`-submits one SLURM job per dataset via `src/grn_inference/run_grn_inference.sh`.

## Age-associated TF activity and gene expression analysis

Computes features (TF activity, gene expression, etc.) from the inferred GRNs and tests
their association with age/condition, aggregating across cohorts via meta-analysis where
applicable.

```bash
sbatch scripts/feature_analysis.sh
```

`analysis_name` selects the feature type (e.g. `tfa_major_b` for TF activity,
`ge_major_b` for gene expression.


## License

MIT, see `LICENSE`.
