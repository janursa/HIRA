#!/bin/bash
# Acquires raw input data for a cohort. For publicly hosted cohorts this downloads
# the file; for gated/private cohorts it prints the manual access steps.
#
# Usage: bash scripts/process_data/acquire/download_data.sh <cohort>
# Cohorts: onek1k perez_sle aida parsebioscience abf300 wang soundlife op CXCL9
#
# Downloads land under $HIRA_RAW_DIR, the same root
# scripts/process_data/run_preprocess.sh reads raw input from. See README > Data Acquisition.

set -e

source scripts/_env.sh

RAW_DIR="${HIRA_RAW_DIR:?set HIRA_RAW_DIR in .env}"
cohort="$1"

download_cellxgene() {
    # $1 = CELLxGENE collection id, $2 = output h5ad path, $3 = optional title filter
    python3 - "$1" "$2" "$3" <<'PY'
import sys, json, urllib.request
collection_id, out, filt = sys.argv[1], sys.argv[2], sys.argv[3]
url = f"https://api.cellxgene.cziscience.com/curation/v1/collections/{collection_id}"
data = json.loads(urllib.request.urlopen(url).read())
datasets = data["datasets"]
if filt:
    matches = [d for d in datasets if filt.lower() in d["title"].lower()]
    datasets = matches or datasets
ds = datasets[0]
asset = next(a for a in ds["assets"] if a["filetype"] == "H5AD")
print(f"Downloading: {ds['title']}\n  {asset['url']}\n  -> {out}")
urllib.request.urlretrieve(asset["url"], out)
PY
}

case "$cohort" in
  onek1k)
    out="${RAW_DIR}/onek1k/onek1k.h5ad"; mkdir -p "$(dirname "$out")"
    download_cellxgene dde06e0f-ab3b-46be-96a2-a8082383c4a1 "$out" ""
    ;;
  perez_sle)
    out="${RAW_DIR}/perez_sle/perez_sle.h5ad"; mkdir -p "$(dirname "$out")"
    download_cellxgene 436154da-bcf1-4130-9c8b-120ff9a888f2 "$out" ""
    ;;
  aida)
    # Freeze v2 — the version the CIIM copy used for preprocessing (1,265,624 cells).
    out="${RAW_DIR}/aida/aida.h5ad"; mkdir -p "$(dirname "$out")"
    download_cellxgene ced320a1-29f3-47c1-a735-513c7084d508 "$out" "Freeze v2"
    ;;
  parsebioscience)
    out="${RAW_DIR}/perturbation_data/Parse_10M_PBMC_cytokines.h5ad"; mkdir -p "$(dirname "$out")"
    wget -c -O "$out" https://parse-wget.s3.us-west-2.amazonaws.com/10m/Parse_10M_PBMC_cytokines.h5ad
    ;;
  abf300)
    cat <<EOF
ABF300 requires an approved Synapse account (accession syn49637038):
  1. Request access: https://www.synapse.org/Synapse:syn49637038
  2. pip install synapseclient && synapse login
  3. synapse get syn49637038 --downloadLocation "${RAW_DIR}/abf300/"
EOF
    ;;
  wang)
    cat <<EOF
Wang cohort requires an approved Synapse account (accession syn61609846):
  1. Request access: https://www.synapse.org/Synapse:syn61609846
  2. pip install synapseclient && synapse login
  3. synapse get syn61609846 --downloadLocation "${RAW_DIR}/wang/"
EOF
    ;;
  soundlife)
    cat <<EOF
SoundLife is publicly hosted by the Allen Institute for Immunology and requires
sign-in with a Google account (no API, browser download only):
  1. Visit https://apps.allenimmunology.org/aifi/insights/dynamics-imm-health-age/downloads/scrna/
  2. Download the 8 SoundLife_<AgeGroup>_<Sex>_<CMVStatus>.h5ad files
  3. Place them under "${RAW_DIR}/soundlife/".
EOF
    ;;
  op)
    cat <<EOF
OPSCA (NeurIPS 2023 Open Problems single-cell perturbation competition) is
distributed via Kaggle and requires a Kaggle account + API token (~/.kaggle/kaggle.json):
  1. pip install kaggle
  2. kaggle competitions download -c open-problems-single-cell-perturbations -p "${RAW_DIR}/perturbation_data/"
Data loaders/description: https://openproblems.bio/events/2023-08_neurips/
EOF
    ;;
  CXCL9)
    echo "CXCL9 has no public source; it is CIIM-internal. Obtain it from the CIIM data lake."
    ;;
  *)
    echo "Usage: bash scripts/process_data/acquire/download_data.sh <cohort>"
    echo "Cohorts: onek1k perez_sle aida parsebioscience abf300 wang soundlife op CXCL9"
    exit 1
    ;;
esac