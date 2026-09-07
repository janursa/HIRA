"""
Regenerate the data files shipped inside the GRNimmuneClock package:
  - grnimmuneclock/data/consensus_grn_{CD4T,CD8T}.csv : consensus GRNs over DISCOVERY_COHORTS
  - grnimmuneclock/data/example_data.h5ad             : one pseudobulk sample from one bulk cohort
  - grnimmuneclock/models/{CD4T,CD8T}/                : the published clock models

Usage:
    python src/clock/build_package_data.py [--dataset aida] [--cell-type CD4T]
"""
import argparse
import json
from datetime import date
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import spearmanr

from hira.src.config import HIRA_DIR, MAJOR_CT_LABEL, CLOCK_TRAINING_COHORTS, CLOCK_TEST_COHORTS, \
    CLOCK_V, CLOCK_CV_SCORING, TUNE_CLOCK
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


def build_models(cell_types=('CD4T', 'CD8T')):
    """Retrain and bundle the published models, same recipe as run_train.py, straight into
    the package's own models dir so retrieve_function() (no model_dir override) picks them up."""
    from grnimmuneclock import train_aging_clock, retrieve_function

    for cell_type in cell_types:
        adata_train = ad.concat([
            retrieve_adata(dataset=d, data_type='bulk', cell_type=cell_type, only_sig_genes=True)
            for d in CLOCK_TRAINING_COHORTS
        ])
        train_aging_clock(
            adata=adata_train, cell_type=cell_type, version=CLOCK_V, output_dir=PKG_MODELS_DIR,
            reg_type='ridge', tune_model=TUNE_CLOCK, scoring=CLOCK_CV_SCORING, verbose=True,
        )

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='aida', help='bulk cohort for example_data.h5ad')
    parser.add_argument('--cell-type', default='CD4T')
    args = parser.parse_args()

    PKG_DATA_DIR.mkdir(parents=True, exist_ok=True)
    build_grns()
    build_example(args.dataset, args.cell_type)
    build_models()
