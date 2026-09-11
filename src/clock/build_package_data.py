"""
Regenerate the data files shipped inside the GRNimmuneClock package:
  - grnimmuneclock/data/consensus_grn_{CD4T,CD8T}.csv : consensus GRNs over DISCOVERY_COHORTS
  - grnimmuneclock/data/example_data.h5ad             : one pseudobulk sample from one bulk cohort
  - grnimmuneclock/data/aging_stats_{CD4T,CD8T}.csv   : per-gene empirical aging direction (pooled_rho)
  - grnimmuneclock/models/{CD4T,CD8T}/                : the published clock models

Anything else already in those directories is deleted (prune_stale).

Usage:
    python src/clock/build_package_data.py [--dataset aida] [--cell-type CD4T]
"""
import argparse
import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import spearmanr

from hira.src.config import HIRA_DIR, MAJOR_CT_LABEL, CLOCK_TRAINING_COHORTS, CLOCK_TEST_COHORTS, \
    CLOCK_V, CLOCKS_DIR
from hira.src.utils.util import retrieve_net_consensus, retrieve_adata

PKG_DIR = Path(HIRA_DIR) / 'GRNimmuneClock' / 'grnimmuneclock'
PKG_DATA_DIR = PKG_DIR / 'data'
PKG_MODELS_DIR = PKG_DIR / 'models'
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


def build_aging_stats(cell_types=('CD4T', 'CD8T')):
    """pooled_rho per clock gene -- the empirical aging direction used to sign the clock
    coefficients before ULM."""
    from grnimmuneclock import retrieve_function
    from hira.src.feature_association.helper import retrieve_stats
    from hira.src.config import REF_GE_ANALYSIS

    for cell_type in cell_types:
        _, gene_names = retrieve_function(cell_type=cell_type, model_dir=CLOCKS_DIR, version=CLOCK_V)
        emp = retrieve_stats(analysis_name=REF_GE_ANALYSIS, cell_type=cell_type,
                             multi_cohort=True).drop_duplicates(subset='gene').set_index('gene')
        df = emp.reindex(list(gene_names))[['pooled_rho']].rename_axis('gene').dropna().reset_index()
        path = PKG_DATA_DIR / f'aging_stats_{cell_type}.csv'
        df.to_csv(path, index=False)
        print(f'{path}: {len(df)}/{len(gene_names)} clock genes with empirical aging stats')


def build_models(cell_types=('CD4T', 'CD8T')):
    """Bundle the models run_train.py already wrote to CLOCKS_DIR -- copied, not retrained, so
    the package ships exactly the clocks the figures were made with (tuning is not deterministic)."""
    from grnimmuneclock import retrieve_function

    for cell_type in cell_types:
        src = Path(CLOCKS_DIR) / cell_type
        dst = PKG_MODELS_DIR / cell_type
        dst.mkdir(parents=True, exist_ok=True)
        for name in (f'model_{CLOCK_V}.pkl', f'feature_names_{CLOCK_V}.txt'):
            if not (src / name).exists():
                raise FileNotFoundError(f'{src / name} missing -- run src/clock/run_train.py first')
            shutil.copy2(src / name, dst / name)
        print(f'{dst}: copied {CLOCK_V} model from {src}')

        model, gene_names = retrieve_function(cell_type=cell_type, model_dir=PKG_MODELS_DIR, version=CLOCK_V)
        y_true, y_pred = [], []
        for d in CLOCK_TEST_COHORTS:
            adata_test = retrieve_adata(dataset=d, data_type='bulk', cell_type=cell_type, condition='healthy')
            Xd = pd.DataFrame(adata_test.X.toarray() if issparse(adata_test.X) else np.asarray(adata_test.X),
                              columns=adata_test.var_names).reindex(columns=gene_names, fill_value=0)
            y_pred.append(model.predict(Xd.values))
            y_true.append(adata_test.obs['age'].astype(float).values)
        y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)

        metadata = {
            'cell_type': cell_type,
            'model_type': 'Ridge Regression',
            'training_data': ', '.join(CLOCK_TRAINING_COHORTS),
            'n_features': len(gene_names),
            'performance': {
                'spearman': round(float(spearmanr(y_true, y_pred).correlation), 3),
            },
            'version': CLOCK_V,
            'trained_date': date.today().isoformat(),
        }
        meta_path = PKG_MODELS_DIR / cell_type / 'metadata.json'
        meta_path.write_text(json.dumps(metadata, indent=2))
        print(f'{meta_path}: {metadata}')


def prune_stale(cell_types=('CD4T', 'CD8T')):
    """Delete anything the builders above didn't just write -- a file the package no longer
    needs (a dropped cell type, a retired analysis input) otherwise keeps shipping forever."""
    keep = {PKG_DATA_DIR / 'example_data.h5ad'}
    for cell_type in cell_types:
        keep |= {PKG_DATA_DIR / f'consensus_grn_{cell_type}.csv',
                 PKG_DATA_DIR / f'aging_stats_{cell_type}.csv'}
        keep |= {PKG_MODELS_DIR / cell_type / n for n in
                 (f'model_{CLOCK_V}.pkl', f'feature_names_{CLOCK_V}.txt', 'metadata.json')}

    for path in sorted(list(PKG_DATA_DIR.rglob('*')) + list(PKG_MODELS_DIR.rglob('*'))):
        if path.is_file() and path not in keep:
            path.unlink()
            print(f'pruned {path}')
    for d in sorted(PKG_MODELS_DIR.iterdir(), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
            print(f'pruned {d}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='aida', help='bulk cohort for example_data.h5ad')
    parser.add_argument('--cell-type', default='CD4T')
    args = parser.parse_args()

    PKG_DATA_DIR.mkdir(parents=True, exist_ok=True)
    build_grns()
    build_example(args.dataset, args.cell_type)
    build_models()
    build_aging_stats()
    prune_stale()
