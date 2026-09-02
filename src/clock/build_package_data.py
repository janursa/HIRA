"""
Regenerate the data files shipped inside the GRNimmuneClock package
(GRNimmuneClock/grnimmuneclock/data/):
  - consensus_grn_{CD4T,CD8T}.csv : consensus GRNs over DISCOVERY_COHORTS
  - example_data.h5ad             : one pseudobulk sample from one bulk cohort

Usage:
    python src/clock/build_package_data.py [--dataset aida] [--cell-type CD4T]
"""
import argparse
from pathlib import Path

import anndata as ad

from hira.src.config import HIRA_DIR, MAJOR_CT_LABEL
from hira.src.utils.util import retrieve_net_consensus, retrieve_adata

PKG_DATA_DIR = Path(HIRA_DIR) / 'GRNimmuneClock' / 'grnimmuneclock' / 'data'
EXAMPLE_OBS = ['age', 'donor_id', 'sex', 'dataset', 'cell_type']


def build_grns(cell_types=('CD4T', 'CD8T')):
    for cell_type in cell_types:
        net = retrieve_net_consensus(cell_type=cell_type)
        out = PKG_DATA_DIR / f'consensus_grn_{cell_type}.csv'
        net.to_csv(out, index=False)
        print(f'{out}: {len(net)} links, {net["source"].nunique()} TFs, {net["target"].nunique()} targets')


def build_example(dataset='aida', cell_type='CD4T'):
    """One donor's pseudobulk profile, donor_id anonymised (this is a public example file)."""
    adata = retrieve_adata(dataset=dataset, data_type='bulk', cell_type=cell_type)
    adata = adata[:1].copy()
    adata.obs['cell_type'] = adata.obs.get(MAJOR_CT_LABEL, cell_type)
    adata.obs['donor_id'] = [f'{dataset}_donor1']
    adata.obs = adata.obs[EXAMPLE_OBS]
    adata.var = adata.var[[]]
    adata.uns = {k: v for k, v in adata.uns.items() if k == 'log1p'}
    out = PKG_DATA_DIR / 'example_data.h5ad'
    adata.write_h5ad(out)
    print(f'{out}: {adata.shape[0]} sample x {adata.shape[1]} genes from {dataset}/{cell_type}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='aida', help='bulk cohort for example_data.h5ad')
    parser.add_argument('--cell-type', default='CD4T')
    args = parser.parse_args()

    PKG_DATA_DIR.mkdir(parents=True, exist_ok=True)
    build_grns()
    build_example(args.dataset, args.cell_type)
