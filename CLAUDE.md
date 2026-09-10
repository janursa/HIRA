# Guideline to Claude on HIRA project

- input sc/bulk/metacell are all in /vol/projects/jnourisa/hira/datasets
- we write all the outputs to results_folder
- run feature association: scripts/feature_association
- run clock analysis: scripts/clock_analysis
- run data preprocessing: scripts/preprocess
- run grn inference: scripts/grn_inference (reads sc data directly, not bulk -- can run without waiting for bulkify)
- run any explanatory/stress analysis: scripts/exp_analysis


see README.md
see README.md

**CRITICAL** for any experimental work, first write it to `temp/` folder on this folder (`hira`). once we check and approved, put the code into the right place inside `src` and `scripts`

`src`-> only python/source code
`scripts` -> bash files making calls to `src` files

**CRITICAL** i summarize the current issues we are working on in `plans/plan.md`. When you given a prompt, check that file to see any relevant information. 


To retrieve sig features associated with aging/condition, use `retrieve_sig_stats`

Plot clusters (groups of individual plots that belong together): `scripts/merge/clusters.yaml`. When asked to show one of these groups, read this file and send every listed plot together.
