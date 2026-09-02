# Pipeline stages

Run in order; each stage consumes the previous one's output. Paths from `src/config.py`:
`HIRA_BASE_DIR` holds heavy data, `results_folder/` holds git-tracked results.

| # | Stage | Command | Writes |
|---|---|---|---|
| 1 | Prior files | `bash scripts/prior/acquire.sh <file>` | `$HIRA_BASE_DIR/prior/` |
| 2 | Raw data | `bash scripts/process_data/acquire/download_data.sh <cohort>` | `$HIRA_RAW_DIR/` |
| 3 | Preprocess | `bash scripts/process_data/wrapper_run_preprocess.sh` | `$HIRA_BASE_DIR/datasets/{sc,bulk,bulk_minor,metacell}/` |
| 4 | GRN inference | `bash scripts/grn_inference/wrapper_grn_inference.sh` | `results_folder/grns/` |
| 5 | Feature association | `sbatch scripts/feature_analysis.sh <analysis_name>` | `results_folder/features/`, `results_folder/plots/` |
| 6 | Aging clocks | `bash scripts/clock_analysis.sh` | `results_folder/clock/`, `results_folder/plots/` |
| 7 | Supplementary figures | `bash scripts/supp_figs.sh` | `results_folder/plots/`, `results_folder/features/` |

Stages 3, 4 and 5 submit SLURM jobs; the rest run locally.

`<file>` for stage 1 and `<cohort>` for stage 2: run the script with no argument to list them.
`<analysis_name>` for stage 5 (default `tfa_major_b`) comes from `CONFIG_FA` in `src/config.py`:
`tfa_major_b tfa_major_sc tfa_major_mc tfa_sub_b ct_tf_markers ge_major_b ge_sub_b tfa_peg
ct_freq ct_pol_dist ccc_sub_b ccc_major_b`.

Supplementary figures also run individually: `python src/process_data/dataset_stats.py`,
`python src/grn_inference/plot_overlap.py`,
`python src/feature_association/discovery_validation_tables.py`,
`python src/feature_association/activation_vs_expression.py`.

Assembled multi-panel manuscript figures: `python scripts/assemble_figs/<script>.py`.
Data shipped inside the GRNimmuneClock package: `python src/clock/build_package_data.py`.
