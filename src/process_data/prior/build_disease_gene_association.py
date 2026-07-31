"""
Build disease_gene_association.csv from OpenTargets' "association_overall_direct" export.

Manual step first (large download, not automated here):
    wget --recursive --no-parent --no-host-directories --cut-dirs 6 \\
      ftp://ftp.ebi.ac.uk/pub/databases/opentargets/platform/25.06/output/association_overall_direct \\
      -P <PRIOR_DIR>/

This script then maps disease EFO ids -> names (EBI OLS API) and target Ensembl ids ->
gene symbols (mygene.info API), which are slow (one request per id / per batch), so the
id->name maps are cached to disease_map.json / gene_map.json in PRIOR_DIR and reused on
rerun.

From notebooks/prepare_resources.ipynb ("# Gene-disease assosiations").

Usage:
    python src/process_data/prior/build_disease_gene_association.py
"""
import argparse
import json
import os
import sys

import pandas as pd
import requests

sys.path.insert(0, 'src')
from config import PRIOR_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--association_dir', default=f'{PRIOR_DIR}/association_overall_direct')
parser.add_argument('--out', default=f'{PRIOR_DIR}/disease_gene_association.csv')
args = parser.parse_args()

if not os.path.exists(args.association_dir):
    raise FileNotFoundError(
        f'{args.association_dir} not found. Download it first:\n'
        '  wget --recursive --no-parent --no-host-directories --cut-dirs 6 '
        'ftp://ftp.ebi.ac.uk/pub/databases/opentargets/platform/25.06/output/association_overall_direct '
        f'-P {PRIOR_DIR}/'
    )

df = pd.read_parquet(args.association_dir)


def get_disease_labels(efo_ids):
    labels = {}
    for efo_id in efo_ids:
        url = f"https://www.ebi.ac.uk/ols/api/ontologies/efo/terms?obo_id={efo_id.replace(':', '_')}"
        r = requests.get(url)
        if r.ok and r.json()["page"]["totalElements"] > 0:
            labels[efo_id] = r.json()["_embedded"]["terms"][0]["label"]
    return labels


def batch_get_gene_symbols(ensembl_ids, batch_size=1000):
    gene_map = {}
    url = "https://mygene.info/v3/query"
    for i in range(0, len(ensembl_ids), batch_size):
        batch_ids = ensembl_ids[i:i + batch_size]
        params = {"q": ",".join(batch_ids), "scopes": "ensembl.gene", "fields": "symbol", "species": "human"}
        try:
            r = requests.post(url, data=params)
            r.raise_for_status()
            for hit in r.json():
                if "symbol" in hit and "query" in hit:
                    gene_map[hit["query"]] = hit["symbol"]
        except Exception as e:
            print(f"Failed batch {i}-{i + batch_size}: {e}")
    return gene_map


disease_map_path = f'{PRIOR_DIR}/disease_map.json'
if os.path.exists(disease_map_path):
    with open(disease_map_path) as f:
        disease_map = json.load(f)
else:
    disease_map = get_disease_labels(df["diseaseId"].unique().tolist())
    with open(disease_map_path, 'w') as f:
        json.dump(disease_map, f, indent=4)

gene_map_path = f'{PRIOR_DIR}/gene_map.json'
if os.path.exists(gene_map_path):
    with open(gene_map_path) as f:
        gene_map = json.load(f)
else:
    gene_map = batch_get_gene_symbols(df['targetId'].unique().tolist())
    with open(gene_map_path, 'w') as f:
        json.dump(gene_map, f, indent=4)

df["disease_name"] = df["diseaseId"].map(disease_map)
df["gene_name"] = df["targetId"].map(gene_map)
df = df[~(df['disease_name'].isna() | df['gene_name'].isna())]
df = df[['disease_name', 'gene_name', 'score', 'evidenceCount']]

df.to_csv(args.out, index=False)
print(f'Saved: {args.out}')
