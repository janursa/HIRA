# HIRA

Analysis pipeline for immune aging: cell-type-resolved GRN inference, aging clocks,
feature association, pathway/motif/trajectory analysis.

## Quick look
Once the repo pulled, the main results can be found in `results_folder` (e.g. inferred GRN models and summary stats of aging, disease, and perturbations). Run `notebooks/summary.ipynb` to produce the main figures.

## Setup

```bash
git clone --recurse-submodules git@github.com:janursa/ciim.git hira
cd hira
cp .env.example .env       # then fill it in (see Configuration)
bash singularity/build.sh  # -> singularity/hira.sif (~550 MB, ~15 min)
```

That's it. Every script sources `scripts/_env.sh`, which loads `.env`, puts the repo and
`GRNimmuneClock/` on `PYTHONPATH`, and runs `python` inside `singularity/hira.sif`.
`HIRA_DIR`, `HIRA_BASE_DIR` and `HIRA_RAW_DIR` are bind-mounted
at their host paths, so nothing else needs configuring. Run all scripts from the repo root.

The image bundles everything in `requirements.txt` plus `bedtools`; `GRNimmuneClock/` is
*not* baked in — the repo copy is used via `PYTHONPATH`, so it's never stale. 

**Without Singularity:** set `HIRA_SIF=` (empty) in `.env`, then

```bash
conda create -n hira python=3.10 -y && conda activate hira
pip install -r requirements.txt && pip install -e GRNimmuneClock/
```
Currently, HIRA is directly depending on GRNimmuneClock recent updated. In the future, this should be replaced by pip installation.

Verify either way with `bash -c 'source scripts/_env.sh; python -c "import hira"'`.

## Configuration

Copy `.env.example` to `.env` and set:

- `HIRA_BASE_DIR` — where the processed datasets (`datasets/`) are stored. Defaults to the repo.
  Priors live in `<repo>/prior`, results (GRNs, feature matrices, summary stats, clock models,
  plots) in `<repo>/results_folder`. Git tracks only what `notebooks/summary.ipynb` needs:
  summary stats, consensus GRNs, clock models and predictions, the `tfa_major_b` TF-activity
  matrices, and three prior files.
- `HIRA_RAW_DIR` — downloaded public cohorts (see Data acquisition below). Required for preprocessing.
- `HIRA_SIF` — path to the Singularity image. Defaults to `singularity/hira.sif`; set it
  empty to use the host/conda interpreter instead. The `.sif` is not in git — build it locally.
- `HIRA_OP_RAW_FILE` — raw `op` cohort h5ad; only needed for that cohort.
- `HIRA_SBATCH_MAIL` — optional; set it to get `--mail-type=END,FAIL` on submitted SLURM jobs.

`HIRA_DIR` is derived from the repo location — don't set it.

Everything else — cell types, cohort lists, feature definitions, clock settings — is in `src/config.py`.

## Pipeline stages

Run in order; each stage consumes the previous one's output.

| # | Stage | Command | Writes |
|---|---|---|---|
| 1 | Prior files | `bash scripts/prior/acquire.sh <file>` | `prior/` |
| 2 | Raw data | `bash scripts/process_data/acquire/download_data.sh <cohort>` | `$HIRA_RAW_DIR/` |
| 3 | Preprocess | `bash scripts/process_data/wrapper_run_preprocess.sh` | `$HIRA_BASE_DIR/datasets/{sc,bulk,bulk_minor,metacell}/` |
| 4 | GRN inference | `bash scripts/grn_inference/wrapper_grn_inference.sh` | `results_folder/grns/` |
| 5 | Feature association | `bash scripts/feature_association/wrapper_feature_analysis.sh [analysis_name] [task ...]` | `results_folder/features/`, `results_folder/plots/` |
| 6 | Aging clocks | `bash scripts/clock_analysis.sh` | `results_folder/clock/`, `results_folder/plots/` |
| 7 | Supplementary + stress analyses | `bash scripts/exp_analysis.sh [analysis_name]` | `results_folder/plots/exp_analysis/`, `results_folder/exp_analysis/`, `results_folder/features/`, `results_folder/clock_stress/`

Stages 3–5 `sbatch`-submit one SLURM job per dataset/task.


### Explanatory / stress analyses

Not part of the main chain; each reads the outputs above. All run from `scripts/exp_analysis.sh`.
Each task's plots get their own subfolder under `results_folder/plots/exp_analysis/`; non-plot
tables (confounders) go under `results_folder/exp_analysis/`.

- **Supplementary tables/figures** — cohort stats (`plots/exp_analysis/cohort_stats/`), GRN overlap
  (`plots/exp_analysis/grn_overlap/`), discovery-vs-validation (`results_folder/features/`),
  activation-vs-expression (`plots/exp_analysis/activation_vs_expression/`).
- **Confounders** — age-correlated donor metadata, informs `CONFOUND_COVARIATES` in
  `src/config.py` →
  `exp_analysis/confounders/confounders_{overall,by_celltype}.csv` and
  `plots/exp_analysis/confounders/confounders.png`.
  Pass extra args after `analysis_name`, e.g.
  `bash scripts/exp_analysis.sh tfa_major_b --cohorts ...`.
- **Clock stress test** (bottom of the script, sbatch-submitted) — retrains the clock under
  one perturbed setting at a time (model family, metacell instead of pseudobulk, whole genome
  instead of GRN targets) and checks whether the baseline's CV, SLE and rejuvenation readouts
  still hold. Submits one job per variant, then
  `python src/exp_analysis/clock_stress.py --aggregate` combines and plots them →
  `results_folder/clock_stress/`. 

## Data acquisition

Preprocessing reads raw per-cohort files from `$HIRA_RAW_DIR` (default `/vol/projects/CIIM`).
To fetch a cohort's raw data, or print manual access instructions where a download can't be
automated:

```bash
bash scripts/process_data/acquire/download_data.sh <cohort>
```

| Cohort | Access | Source |
|---|---|---|
| OneK1K (`onek1k`) | direct download | [CELLxGENE collection](https://cellxgene.cziscience.com/collections/dde06e0f-ab3b-46be-96a2-a8082383c4a1) |
| Perez SLE (`perez_sle`) | direct download | [CELLxGENE collection](https://cellxgene.cziscience.com/collections/436154da-bcf1-4130-9c8b-120ff9a888f2) |
| AIDA (`aida`) | direct download | [CELLxGENE collection](https://cellxgene.cziscience.com/collections/ced320a1-29f3-47c1-a735-513c7084d508) (Freeze v2) |
| ParseBioscience (`parsebioscience`) | direct download | Parse Biosciences S3 bucket |
| ABF300 (`abf300`) | manual, gated | Synapse `syn49637038` (account + data use agreement required) |
| Wang (`wang`) | manual, gated | Synapse `syn61609846` (account + data use agreement required) |
| OPSCA (`op`) | manual, gated | Kaggle competition `open-problems-single-cell-perturbations` (account + API token required) |
| SoundLife (`soundlife`) | manual, private | not publicly hosted — obtained via direct data transfer from study authors |
| CXCL9 (`CXCL9`) | internal | CIIM-only, no public source |


## Repo layout

```
scripts/    bash entry points — one per stage above, sbatch wrappers included
src/        all python
  config.py            paths, cohort lists, cell types, CONFIG_FA feature definitions
  process_data/        QC, cell type annotation, pseudobulk/metacell, prior file builders
  grn_inference/       per-cell-type GRN inference
  feature_association/ feature computation + age/condition association, meta-analysis, plots
  clock/               clock training, CV, comparison against published clocks
  exp_analysis/        confounder and clock stress analyses
  pathway_analysis/    gene-set / aging-hallmark enrichment over associated features
  network_analysis/    shared plot primitives (dotplots, category colouring) used by the above
  utils/               shared IO and plotting helpers
tests_code/ regression + determinism tests, one folder per stage (see tests_code/readme.md)
tests_data/ heavy test fixtures (git-lfs), symlinked into tests_code/<stage>/data
GRNimmuneClock/  submodule — the installable clock package
```

Results (`results_folder/`, git-tracked, everything lightweight):

- `grns/` — per-cohort and consensus GRNs
- `features/` — association statistics and supplementary tables
- `clock/` — trained clock models and predictions
- `exp_analysis/` — confounder tables; `clock_stress/` — stress test outputs
- `plots/` — all figures, `plots/exp_analysis/` for the supplementary-analysis plots (one subfolder
  per task), `plots/assembled/` for multi-panel manuscript figures

## Key outputs → code

Where each headline result lives and which code produced it.

### 1. Processed single-cell and pseudobulk data

`$HIRA_BASE_DIR/datasets/`, one `<cohort>.h5ad` per cohort in each subfolder:

| Output | Produced by |
|---|---|
| `sc/` — QC'd, cell-type-annotated single cells | `src/process_data/preprocess/script.py` |
| `bulk/` — donor × major-cell-type pseudobulk (carries minor-cell-type counts in `.obs`) | `src/process_data/bulkify/script.py` |
| `bulk_minor/` — donor × minor-cell-type pseudobulk | `src/process_data/bulkify/script.py` |
| `metacell/` — metacells (clock stress test only) | `src/process_data/metacell/script.py` |

Entry point: `bash scripts/process_data/wrapper_run_preprocess.sh` (one SLURM job per cohort,
all three stages per job). Raw inputs come from `$HIRA_RAW_DIR` — see Data acquisition.

### 2. GRN models

`results_folder/grns/`:

| Output | Produced by |
|---|---|
| `<cohort>/{sc,bulk}/net_<celltype>.csv` — per-cohort GRN | `src/grn_inference/script.py` |
| `consensus_net_<celltype>.csv` — edges shared by ≥ `CONSENSUS_MIN_DEGREE` discovery cohorts | `src/feature_association/consensus_nets.py` |

Cell types: B, CD4T, CD8T, MONO, NK. Discovery cohorts in `DISCOVERY_COHORTS` (`src/config.py`).
Entry point: `bash scripts/grn_inference/wrapper_grn_inference.sh`. The consensus nets are rebuilt
at the top of `wrapper_feature_analysis.sh`, so stage 5 refreshes them automatically.

### 3. Summary statistics — aging, SLE, perturbation

All under `results_folder/features/<analysis_name>/stats/`, same schema (`gene`, `slope`,
`meta_p_adj`, `cell_type`, `dataset`, `comparison`, `trend`, …). `tfa_major_b` is TF activity on
pseudobulk — the main analysis; `ge_major_b` is the gene-expression counterpart.

| File | Contrast | Cohorts |
|---|---|---|
| `stats_multi_cohort.csv` | aging (meta-analysis) | `DISCOVERY_COHORTS` |
| `stats_soundlife.csv` | aging (validation) | soundlife |
| `stats_perez_sle.csv` | SLE vs healthy | perez_sle |
| `stats_parsebioscience.csv` | IL-10 | parsebioscience |
| `stats_op.csv` | Ruxolitinib | op |
| `stats_CXCL9.csv` | LPS, Ruxolitinib (vs RPMI and vs LPS) | CXCL9 |

All produced by `src/feature_association/run_analysis.py` (one task per file; association logic in
`helper.py` / `helper_condition.py`). Entry point:
`bash scripts/feature_association/wrapper_feature_analysis.sh [analysis_name] [task ...]` — the
`aging` task runs first because the condition tasks read `stats_multi_cohort.csv`.

Read them in python with `retrieve_stats` / `retrieve_sig_stats` from
`src/feature_association/helper.py` rather than parsing the CSVs directly — they apply the
significance and consistency filters used in the manuscript.

## License

MIT, see `LICENSE`.
