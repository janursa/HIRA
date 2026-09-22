#!/bin/bash
# Submits one sbatch job per dataset by calling run_preprocess.sh with the dataset name.
# Usage: bash wrapper_run_preprocess.sh
# Datasets: discovery cohorts (onek1k, abf300, aida, perez_sle), rest: , parsebioscience, wang, soundlife, op, CXCL9

source scripts/_env.sh

datasets="onek1k abf300 aida perez_sle parsebioscience wang soundlife op CXCL9"

# Per-dataset overrides for run_preprocess.sh's default --mem=500GB --time=20:00:00
# (sbatch CLI flags win over the script's #SBATCH defaults):
#   soundlife: 13.8M-cell final merge OOMs at 500GB -> needs more memory
#   parsebioscience: 20 chunks x ~1h each runs past 20h; final merge OOMs at 500GB (peak 469GiB)
declare -A extra_sbatch_args=(
    [soundlife]="--mem=1000GB"
    [parsebioscience]="--time=30:00:00 --mem=1000GB"
)

# CellTypist labels are CPU-architecture dependent. OpenBLAS picks a different micro-kernel
# per CPU family -> float32 PCA differs in the last bits -> kNN ties flip -> CellTypist's
# leiden over-clustering partitions differently -> majority voting rewrites 2-5% of labels.
# Counts matrices are unaffected (byte-identical on any node); only Major_CT/Sub_CT move.
# Cluster families: bioinf014-021 Intel Xeon Gold 6146 (SkylakeX), bioinf027-032 EPYC 7443P
# (Zen3), bioinf033 EPYC 7702 (Zen2), bioinf035-039 EPYC 9654 (Zen4).
# Below: the family each cohort in datasets/backup_20260919 was produced on. Run with
# PIN_ARCH=1 to reproduce those labels exactly; leave unset for normal scheduling.
#   parsebioscience/soundlife skip CellTypist (published annotations) -> arch-irrelevant.
#   wang's producing node was not recoverable from the slurm logs.
declare -A backup_arch=(
    [onek1k]="bioinf[027-032]"     # Zen3, bioinf027 -- verified reproduces 1.0000
    [wang]="bioinf[027-032]"       # Zen3, node unknown -- verified reproduces 1.0000
    [CXCL9]="bioinf[027-032]"      # Zen3, bioinf027 -- see caveat below
    [abf300]="bioinf[035-039]"     # Zen4, bioinf039
    [aida]="bioinf[035-039]"       # Zen4, bioinf036
    [op]="bioinf[035-039]"         # Zen4, bioinf039
    [perez_sle]="bioinf[035-039]"  # Zen4, bioinf038
)
# CXCL9 does not reproduce even on bioinf027, its own backup node (Major_CT 0.973).
# Everything feeding the annotation is bit-identical (counts, doublet scores, cell and
# gene order) and repeat runs agree to the bit, so today's output is deterministic --
# but its backup was built on 2026-09-04 from an image later overwritten by the Sep-11
# rebuild, with the scrublet code still uncommitted. That environment is gone, so the
# Sep-4 labels are not recoverable; treat the current CXCL9 output as the baseline.

for ds in $datasets; do
    pin=""
    [[ -n "$PIN_ARCH" && -n "${backup_arch[$ds]}" ]] && pin="--nodelist=${backup_arch[$ds]}"
    sbatch $SBATCH_MAIL ${extra_sbatch_args[$ds]} $pin scripts/process_data/run_preprocess.sh $ds
    echo "Submitted: $ds $pin"
done


