"""Supplementary Tables 1 & 2: age-associated TFs from the discovery meta-analysis
compared against an independent validation cohort.

Usage: python src/feature_association/discovery_validation_tables.py \
           [--analysis-name tfa_major_b] [--validation-cohort soundlife]
Writes into FEATURES_DIR/<analysis_name>/:
  supp_table1_discovery_validation.csv   every discovery TF x lineage, with validation slope/p
  supp_table2_nonsignificant_validation.csv  the subset that lost significance in validation
  missing_tfs_validation.csv             not significant AND near-zero slope in validation
  inconsistent_tfs_validation.csv        significant in validation but with an opposite slope
"""
import argparse
import os

import numpy as np

from hira.src.config import FEATURES_DIR
from hira.src.feature_association.helper import retrieve_sig_stats, retrieve_stats

SIG_THRESHOLD = 0.05
FLAT_SLOPE = 0.1  # |slope| below this in validation counts as "no trend", not "opposite trend"


def discovery_stats(analysis_name):
    stats = retrieve_sig_stats(analysis_name=analysis_name)[['gene', 'slope', 'meta_p_adj', 'cell_type']]
    stats = stats.groupby(['cell_type', 'gene']).agg({'slope': 'mean', 'meta_p_adj': 'min'}).reset_index()
    stats.columns = ['Cell type', 'TF', 'Slope (discovery)', 'Meta-p adj (discovery)']
    stats['Cell type'] = stats['Cell type'].astype(str)
    return stats


def validation_stats(analysis_name, dataset):
    stats = retrieve_stats(analysis_name=analysis_name, dataset=dataset)
    stats = stats.drop_duplicates(subset=['cell_type', 'gene'])[['gene', 'slope', 'p_value_adj', 'cell_type']]
    stats.columns = ['TF', 'Slope (validation)', 'P adj (validation)', 'Cell type']
    stats['Cell type'] = stats['Cell type'].astype(str)
    return stats


def build_tables(analysis_name='tfa_major_b', dataset='soundlife'):
    ref, val = discovery_stats(analysis_name), validation_stats(analysis_name, dataset)

    table1 = ref.merge(val, on=['Cell type', 'TF'], how='left')
    table1['Direction concordant'] = np.sign(table1['Slope (discovery)']) == np.sign(table1['Slope (validation)'])
    table1['Significant in validation'] = table1['P adj (validation)'] <= SIG_THRESHOLD

    table2 = table1[~table1['Significant in validation']]
    missing = table1[(table1['P adj (validation)'] > SIG_THRESHOLD)
                     & (table1['Slope (validation)'] < FLAT_SLOPE)]
    inconsistent = table1[table1['Significant in validation'] & ~table1['Direction concordant']]

    print(f"Direction concordance: {table1['Direction concordant'].mean()*100:.1f}%")
    print(f"Not significant in validation: {(~table1['Significant in validation']).mean()*100:.1f}%")
    return {
        'supp_table1_discovery_validation': table1,
        'supp_table2_nonsignificant_validation': table2,
        'missing_tfs_validation': missing,
        'inconsistent_tfs_validation': inconsistent,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-name', default='tfa_major_b')
    parser.add_argument('--validation-cohort', default='soundlife')
    args = parser.parse_args()

    out_dir = os.path.join(FEATURES_DIR, args.analysis_name)
    os.makedirs(out_dir, exist_ok=True)
    for name, df in build_tables(args.analysis_name, args.validation_cohort).items():
        file_name = os.path.join(out_dir, f'{name}.csv')
        print(f'Saving to {file_name} ({len(df)} rows)')
        df.to_csv(file_name, index=False)
