"""Pseudocell matrix for the published scImmuAging clocks (predicted in scimmuaging.sif).

Mirrors scImmuAging::PreProcess -- subset to the model's marker genes, log-normalize,
then 100 pseudocells of 15 random cells per donor. Columns are Ensembl ids because that
is what the model expects; gene_map.csv maps them to the symbols HIRA's sc data uses
(built once from the HGNC complete set + Ensembl REST /lookup/id for the leftovers).

Usage: python src/clock_benchmark/export_input.py --dataset aida --cell-type CD4T --out x.tsv.gz
Writes: <out> (donor_id, age, <ENSG...>) and <out>.meta.csv (donor_id, age, sex, condition, n_cells)
"""
import argparse
import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from hira.src.config import DATA_DIR, MAJOR_CT_LABEL, get_config

N_PSEUDO, PSEUDO_SIZE = 100, 15  # scImmuAging::pseudocell defaults
MIN_CELLS = 10                   # bulkify_func's cell_count_t, so the same samples as the clock analysis
AGE_LIMIT = 20                   # retrieve_adata's age_limit, ditto
CHUNK = 20_000
GENE_MAP = os.path.join(os.path.dirname(__file__), 'gene_map.csv')


def pseudocells(Xd, rng):
    """N_PSEUDO x n_genes, each row the mean of PSEUDO_SIZE cells drawn from Xd."""
    n = Xd.shape[0]
    if n <= PSEUDO_SIZE:  # scImmuAging's replace="dynamic"
        draws = rng.integers(0, n, (N_PSEUDO, PSEUDO_SIZE))
    else:
        draws = np.argsort(rng.random((N_PSEUDO, n)), axis=1)[:, :PSEUDO_SIZE]
    return Xd[draws].mean(axis=1)


def harmonize_obs(obs, dataset):
    """The obs conventions of retrieve_adata(data_type='bulk'), so donors and conditions
    line up one-to-one with the clock analysis this benchmark is compared against."""
    obs = obs.copy()
    name_map = get_config(dataset).name_mapping or {}
    obs['condition'] = (obs['condition'].astype(str).map(lambda x: name_map.get(x, x))
                        if 'condition' in obs else 'healthy')
    donor_map = {d: f'Donor {i + 1}' for i, d in enumerate(sorted(obs['donor_id'].unique()))}
    obs['donor_id'] = obs['donor_id'].map(donor_map)
    obs['age'] = obs['age'].astype(float).astype(int)
    obs['donor_age'] = obs['donor_id'].astype(str) + '- age: ' + obs['age'].astype(str)
    obs['sex'] = obs['sex'].map(lambda s: {'F': 'Female', 'M': 'Male'}.get(s, s))
    return obs


def main(dataset, cell_type, out, seed=0):
    genes = pd.read_csv(GENE_MAP).query('cell_type == @cell_type').dropna(subset=['symbol'])
    adata = ad.read_h5ad(f'{DATA_DIR}/sc/{dataset}.h5ad', backed='r')
    obs = harmonize_obs(adata.obs, dataset)
    keep = genes[genes.symbol.isin(adata.var_names)].drop_duplicates('symbol')
    cols = adata.var_names.get_indexer(keep.symbol)
    print(f'{dataset}/{cell_type}: {len(keep)}/{len(genes)} model genes found')

    rows = np.flatnonzero((obs[MAJOR_CT_LABEL].astype(str) == cell_type).values)
    assert len(rows), f'no {cell_type} cells in {dataset}'
    X = sp.vstack([adata.X[rows[i:i + CHUNK]][:, cols] for i in range(0, len(rows), CHUNK)]).tocsr()
    # Seurat NormalizeData: CP10K on the full transcriptome, then log1p
    X = sp.diags(1e4 / obs['total_counts'].values[rows]) @ X
    X = np.log1p(X)

    obs = obs.iloc[rows]
    rng = np.random.default_rng(seed)
    mats, meta = [], []
    # (donor_id, age) is PreProcess's grouping -- a donor sampled at two ages is two samples
    for (donor, age), pos in obs.groupby(['donor_id', 'age'], observed=True).indices.items():
        o, age = obs.iloc[pos[0]], float(age)
        if len(pos) < MIN_CELLS or age < AGE_LIMIT:
            continue
        mats.append(pd.DataFrame(pseudocells(np.asarray(X[pos].todense()), rng),
                                 columns=keep.ensembl.values).assign(donor_id=donor, age=age))
        meta.append({'donor_id': donor, 'age': age, 'donor_age': o['donor_age'],
                     'sex': o['sex'], 'condition': o['condition'], 'n_cells': len(pos)})

    df = pd.concat(mats, ignore_index=True)
    df = df[['donor_id', 'age'] + list(keep.ensembl.values)]  # AgingClockCalculator drops cols 1:2
    df.to_csv(out, sep='\t', index=False, float_format='%.5g')
    pd.DataFrame(meta).to_csv(f'{out}.meta.csv', index=False)
    print(f'{len(meta)} donors -> {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--cell-type', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    main(a.dataset, a.cell_type, a.out)
