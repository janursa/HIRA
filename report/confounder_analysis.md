# Confounder analysis

Donor metadata correlated with age, independent of biology, confounds age-association
results unless controlled for (baseline only adjusts for `cell_count`).

`src/exp_analysis/confounders.py` tests each donor-level covariate against age (R² from
ANOVA/correlation, flag at R²>0.05 & p<0.05) per cohort, and re-checks flagged covariates
within each cell type's donor subset (confound is donor-level, not cell-type-specific).

## Findings

| Cohort | Flagged | R² | Selected for adjustment |
|---|---|---|---|
| aida | `batch_info` (49 batches) | 0.40 | no — too many groups, use coarsened `site` instead |
| aida | `race` | 0.28 | no — reflects population structure, not a technical batch effect |
| aida | `site` (batch_info coarsened to 5 groups) | 0.27 | **yes** |
| onek1k | `batch_info` (75 batches) | 0.23 | **yes** |
| abf300, perez_sle | none flagged | — | — |

Selected covariates (`CONFOUND_COVARIATES` in `src/config.py`) are adjusted for via
empirical-Bayes shrinkage residualization in `association_with_age()`
(`src/feature_association/helper.py`). After adjustment, robust-hit slope sign is uniform
across cohorts (~43.6% positive each), removing the prior cohort-dominant bias
(aida 96.8% vs onek1k 10.6% positive).

Plot: `results_folder/exp_analysis/confounders.png`
