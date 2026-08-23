"""Run once by hand to (re)generate data/raw_manifest.json.

Records size+mtime (not a content hash — raw files are 100s of GB) for every
raw dataset file and the specific PRIOR_DIR files the pipeline actually reads
(gene names + TF list), so test_raw_data_freshness.py can detect if any of
them changed since this snapshot was taken. Not a glob over PRIOR_DIR: that
previously also tripped on unrelated backup files (e.g. tf_all.csv.bak_*)
getting cleaned up, which isn't a real drift the pipeline cares about.
"""
import glob
import json
import os
import sys

REPRO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tests/preprocessing/
HIRA_DIR = os.path.dirname(os.path.dirname(REPRO_DIR))
sys.path.insert(0, os.path.dirname(HIRA_DIR))
from hira.src.config import PRIOR_DIR  # noqa: E402

MANIFEST_PATH = os.path.join(REPRO_DIR, 'data', 'raw_manifest.json')

RAW_DATA_DIR = os.environ.get('HIRA_RAW_DIR', '/vol/projects/CIIM')
OP_RAW_FILE = os.environ.get('HIRA_OP_RAW_FILE', '/vol/projects/jnourisa/genernbi/resources/datasets_raw/op_perturbation_sc_counts.h5ad')

# Mirrors scripts/process_data/run_preprocess.sh's input_file resolution per dataset.
RAW_FILES = {
    'zhang': f'{RAW_DATA_DIR}/Healthy_Single_Cell_Data/count_matrix/data12_CMtx.h5ad',
    'onek1k': f'{RAW_DATA_DIR}/Healthy_Single_Cell_Data/count_matrix/data1_CMtx.h5ad',
    'aida': f'{RAW_DATA_DIR}/Healthy_Single_Cell_Data/count_matrix/data13_CMtx.h5ad',
    'abf300': f'{RAW_DATA_DIR}/Healthy_Single_Cell_Data/count_matrix/data7_allTPs_jalil_CMtx.h5ad',
    'perez_sle': f'{RAW_DATA_DIR}/Healthy_Single_Cell_Data/count_matrix/SLE_CMtx.h5ad',
    'CXCL9': f'{RAW_DATA_DIR}/Healthy_Single_Cell_Data/count_matrix/CXCL9_TI.h5ad',
    'op': OP_RAW_FILE,
    'parsebioscience': f'{RAW_DATA_DIR}/perturbation_data/Parse_10M_PBMC_cytokines.h5ad',
}
RAW_DIRS = {
    'soundlife': f'{RAW_DATA_DIR}/soundlife/',
}
PRIOR_FILES = ['gene_names.txt', 'tf_all.csv']


def stat_entry(path):
    st = os.stat(path)
    return {'size': st.st_size, 'mtime': st.st_mtime}


def main():
    manifest = {}
    for dataset, path in RAW_FILES.items():
        if os.path.exists(path):
            manifest[f'raw:{dataset}:{path}'] = stat_entry(path)
        else:
            print(f'WARN: raw file for {dataset} not found at {path}, skipping', flush=True)
    for dataset, dir_path in RAW_DIRS.items():
        for f in sorted(glob.glob(os.path.join(dir_path, '*.h5ad'))):
            manifest[f'raw:{dataset}:{f}'] = stat_entry(f)
    for fname in PRIOR_FILES:
        f = os.path.join(PRIOR_DIR, fname)
        if os.path.isfile(f):
            manifest[f'prior:{f}'] = stat_entry(f)
        else:
            print(f'WARN: prior file {fname} not found at {f}, skipping', flush=True)

    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, 'w') as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f'Wrote {len(manifest)} entries to {MANIFEST_PATH}', flush=True)


if __name__ == '__main__':
    main()
