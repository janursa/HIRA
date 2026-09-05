"""Age-associated confounders per cohort: which donor-metadata columns (batch/site,
sex, race, condition, QC metrics) correlate with age, independent of any real biology.
A covariate this confounded with age leaks into any age-association result computed
without adjusting for it -- association_with_age() in feature_association/helper.py
only adjusts for cell_count.

Not cell-type specific: candidate covariates are donor-level (repeated across a
donor's Major_CT rows), so the age-covariate relationship is fixed at the donor level.
For every flagged covariate this script also recomputes R^2 within each Major_CT's
donor subset (by_celltype table), to check whether cell-type-driven donor dropout
shifts the confound.

Usage: python src/exp_analysis/confounders.py [--cohorts aida perez_sle onek1k abf300]
Writes: OUTPUT_DIR/exp_analysis/confounders_overall.csv, confounders_by_celltype.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from hira.src.config import DISCOVERY_COHORTS, OUTPUT_DIR, MAJOR_CT_LABEL, CONFOUND_COVARIATES
from hira.src.utils.util import retrieve_adata, coarsen

EXCLUDE = {'age', 'age_group', 'donor_age', 'cell_count', MAJOR_CT_LABEL, 'dataset',
           'donor_id', 'donor_id_old', 'orig.ident', 'sum_by', 'bulk_group'}
R2_FLAG = 0.05
P_FLAG = 0.05


def candidate_covariates(donors):
    return [c for c in donors.columns
            if c not in EXCLUDE and not c.endswith('_count')
            and 2 <= donors[c].nunique(dropna=True) < len(donors)]


def test_covariate(age, values):
    """(R2, p_value, kind, n_groups) for one covariate vs age."""
    if pd.api.types.is_numeric_dtype(values) and values.nunique() > 10:
        mask = ~(age.isna() | values.isna())
        r, p = stats.pearsonr(age[mask], values[mask])
        return r ** 2, p, 'numeric', np.nan
    df = pd.DataFrame({'age': age, 'g': values}).dropna()
    groups = [g['age'].values for _, g in df.groupby('g', observed=True) if len(g) >= 2]
    if len(groups) < 2:
        return np.nan, np.nan, 'categorical', len(groups)
    _, p = stats.f_oneway(*groups)
    grand_mean = df['age'].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((df['age'] - grand_mean) ** 2).sum()
    r2 = ss_between / ss_total if ss_total > 0 else np.nan
    return r2, p, 'categorical', len(groups)


def analyze_cohort(dataset):
    obs = retrieve_adata(dataset, data_type='bulk', condition='healthy', only_obs=True)
    donors = obs.drop_duplicates('donor_id')
    age = donors['age'].astype(float)
    rows = []
    for col in candidate_covariates(donors):
        variants = [(col, donors[col])]
        is_stringlike = donors[col].dtype == object or isinstance(donors[col].dtype, pd.CategoricalDtype)
        if is_stringlike and donors[col].nunique() > 15:
            coarse = coarsen(donors[col])
            if coarse.nunique() < donors[col].nunique():
                variants.append((f'{col}__site', coarse))
        for name, values in variants:
            r2, p, kind, n_groups = test_covariate(age, values)
            rows.append({'cohort': dataset, 'covariate': name, 'type': kind,
                         'n_donors': len(donors), 'n_groups': n_groups, 'R2': r2, 'p_value': p,
                         'flagged': (r2 or 0) > R2_FLAG and (p if p == p else 1) < P_FLAG})
    return pd.DataFrame(rows), obs


def by_celltype_check(dataset, obs, flagged_covariates):
    rows = []
    for cell_type, sub in obs.groupby(MAJOR_CT_LABEL, observed=True):
        donors = sub.drop_duplicates('donor_id')
        if len(donors) < 5:
            continue
        age = donors['age'].astype(float)
        for cov in flagged_covariates:
            base_col = cov.replace('__site', '')
            if base_col not in donors.columns:
                continue
            values = coarsen(donors[base_col]) if cov.endswith('__site') else donors[base_col]
            r2, p, _, _ = test_covariate(age, values)
            rows.append({'cohort': dataset, 'cell_type': cell_type, 'covariate': cov,
                         'n_donors': len(donors), 'R2': r2, 'p_value': p})
    return pd.DataFrame(rows)


def is_selected(cohort, covariate):
    chosen = CONFOUND_COVARIATES.get(cohort, [])
    name = 'site' if covariate.endswith('__site') else covariate
    return name in chosen


def plot_covariates(overall, out_path):
    df = overall.dropna(subset=['R2']).copy()
    df['label'] = df['cohort'] + ':' + df['covariate']
    df['status'] = np.select(
        [df.apply(lambda r: is_selected(r['cohort'], r['covariate']), axis=1), df['flagged']],
        ['selected', 'flagged'], default='not flagged')
    colors = {'selected': '#1b7837', 'flagged': '#f4a582', 'not flagged': '#bbbbbb'}

    n = len(df)
    fig, ax = plt.subplots(figsize=(3, 3 + 0.5 * max(0, (n - 10) // 5)))
    ax.bar(df['label'], df['R2'], color=df['status'].map(colors))
    ax.axhline(R2_FLAG, ls='--', lw=0.8, color='black')
    ax.set_ylabel('R$^2$ (age vs. covariate)')
    ax.set_xticks(range(n))
    ax.set_xticklabels(df['label'], rotation=45, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.margins(y=0.1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, colors.keys(), frameon=False, loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname('Arial')
        tick.set_fontsize(10)
    ax.yaxis.label.set_fontname('Arial')
    ax.yaxis.label.set_fontsize(10)
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohorts', nargs='+', default=DISCOVERY_COHORTS)
    args = parser.parse_args()

    overall_parts, by_ct_parts = [], []
    for dataset in args.cohorts:
        df, obs = analyze_cohort(dataset)
        overall_parts.append(df)
        flagged = df.loc[df['flagged'], 'covariate'].tolist()
        if flagged:
            by_ct_parts.append(by_celltype_check(dataset, obs, flagged))

    overall = pd.concat(overall_parts, ignore_index=True).sort_values(['cohort', 'R2'], ascending=[True, False])
    by_ct = pd.concat(by_ct_parts, ignore_index=True) if by_ct_parts else pd.DataFrame()

    out_dir = f'{OUTPUT_DIR}/exp_analysis'
    os.makedirs(out_dir, exist_ok=True)
    overall.to_csv(f'{out_dir}/confounders_overall.csv', index=False)
    by_ct.to_csv(f'{out_dir}/confounders_by_celltype.csv', index=False)
    plot_covariates(overall, f'{out_dir}/confounders.png')

    pd.set_option('display.width', 160)
    print(overall.to_string(index=False))
    print(f'\n--- Flagged (R2>{R2_FLAG}, p<{P_FLAG}) ---')
    print(overall[overall['flagged']].to_string(index=False))
    if len(by_ct):
        print('\n--- Per-cell-type R2 for flagged covariates ---')
        print(by_ct.to_string(index=False))
    print(f'\nWrote {out_dir}/confounders_overall.csv')
    print(f'Wrote {out_dir}/confounders_by_celltype.csv')
    print(f'Wrote {out_dir}/confounders.png')


if __name__ == '__main__':
    main()
