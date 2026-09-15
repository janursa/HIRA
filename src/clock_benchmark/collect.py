"""Merge the per-cohort/cell-type scImmuAging predictions and score them.

Usage: python src/clock_benchmark/collect.py
Writes: results_folder/clock_benchmark/{predictions.csv,scores.csv}
"""
import glob
import os

import pandas as pd
from grnimmuneclock import evaluate_groupwise_median

from hira.src.config import HIRA_DIR, OUTPUT_DIR

PREDS = f'{HIRA_DIR}/temp/clock_benchmark'  # written by scripts/clock_benchmark/script.sh
DIR = f'{OUTPUT_DIR}/clock_benchmark'


def main():
    os.makedirs(DIR, exist_ok=True)
    preds = []
    for f in sorted(glob.glob(f'{PREDS}/pred_*.tsv')):
        dataset, cell_type = os.path.basename(f)[len('pred_'):-len('.tsv')].rsplit('_', 1)
        df = pd.read_csv(f, sep='\t')
        meta = pd.read_csv(f'{PREDS}/meta_{dataset}_{cell_type}.csv')
        df['age'], meta['age'] = df.age.astype(float), meta.age.astype(float)
        merged = df.merge(meta, on=['donor_id', 'age'])
        assert len(merged) == len(df), f'{f}: meta does not cover every sample'
        preds.append(merged.assign(dataset=dataset))
    preds = pd.concat(preds, ignore_index=True)
    preds.to_csv(f'{DIR}/predictions.csv', index=False)

    healthy = preds[preds.condition == 'healthy']
    # same scoring as the GRN clock: median per donor_age, then spearman against true age
    scores = healthy.groupby(['dataset', 'cell_type']).apply(
        lambda d: pd.Series({'n_donors': d.donor_age.nunique(), **evaluate_groupwise_median(d)}),
        include_groups=False).reset_index().round(3)
    scores.to_csv(f'{DIR}/scores.csv', index=False)
    print(scores.to_string(index=False))


if __name__ == '__main__':
    main()
