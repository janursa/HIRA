"""Stress test for the aging clock: retrain under one perturbed setting at a time and
recompute a fixed readout vector. The `baseline` variant IS the ground truth -- it is the
current pipeline config (CLOCK_TRAINING_COHORTS, bulk, age-associated genes, tuned ridge --
the same only_sig_genes feature set src/clock/run_train.py trains the published clock on).

Readouts:
  cv     -- spearman/r2 on the held-out CLOCK_TEST_COHORTS, every major cell type
  sle    -- perez_sle age acceleration, SLE vs healthy, overall + Young/Old bins (T cells)
  rejuv  -- signed effect on predicted age for IL-10 (parsebioscience), Ruxolitinib (op),
            LPS in CXCL9 against RPMI, and Ruxolitinib in CXCL9 against both the RPMI and
            the LPS baseline (T cells)

Usage:
  python src/exp_analysis/clock_stress.py --variant baseline      # one variant, one sbatch job
  python src/exp_analysis/clock_stress.py --aggregate             # combine + compare + plot
"""
import argparse
import os

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from grnimmuneclock import train_aging_clock
from scipy import stats
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from hira.src.config import (CLOCK_CV_SCORING, CLOCK_TEST_COHORTS, CLOCK_TRAINING_COHORTS,
                             CLOCK_V, MAJOR_CTS, OUTPUT_DIR, TUNE_CLOCK, get_config)
from hira.src.clock.helper import wrapper_clock_predictions
from hira.src.utils.util import retrieve_adata, test_mixed_effects

STRESS_DIR = f'{OUTPUT_DIR}/clock_stress'

# One axis of stress at a time. Anything not listed keeps the baseline value.
VARIANTS = {
    'baseline':         {},
    'lasso':            {'reg_type': 'lasso'},
    'elasticnet':       {'reg_type': 'elasticnet'},
    'gradientboosting': {'reg_type': 'gradientboosting'},
    'nn':               {'reg_type': 'nn'},
    'metacell':         {'data_type': 'metacell'},
    'wholegenome':      {'only_sig_genes': False},
}
BASELINE = dict(reg_type='ridge', data_type='bulk', only_sig_genes=True)

# Condition contrasts: (dataset, control, treatment). Restricted to the T cell types
# the published claims are made on. LPS vs RPMI is the acceleration claim, the rest
# are the rejuvenation ones -- same signed readout either way.
REJUV_CONTRASTS = [
    ('parsebioscience', 'PBS', 'IL-10'),
    ('op', 'DMSO', 'Ruxolitinib'),
    ('CXCL9', 'RPMI', 'LPS'),
    ('CXCL9', 'RPMI', 'RPMI + ruxolitinib'),
    ('CXCL9', 'LPS', 'LPS + ruxolitinib'),
]
CTS = MAJOR_CTS         # clocks are trained and cross-validated on every major cell type
T_CTS = ['CD4T', 'CD8T']  # the SLE and rejuvenation claims are T cells only
# ponytail: the parsebioscience metacell file is 47GB; the IL-10 contrast is dropped from
# the metacell variant rather than sized around. Re-add if the other axes leave it open.
SKIP_CONTRASTS = {('metacell', 'parsebioscience')}


def median_per_donor(obs):
    """Metacell/sc give many rows per donor; collapse to one median prediction per sample.

    `bulk_group` is the dataset's own sample key (donor, or donor x condition), so this
    reproduces exactly the grouping the bulk variant gets from pseudobulking.
    """
    keys = ['dataset', 'cell_type', 'condition', 'bulk_group']
    agg = obs.groupby(keys, observed=True).agg(
        predicted_age=('predicted_age', 'median'),
        age=('age', 'first'),
        donor_id=('donor_id', 'first'),
        n_units=('predicted_age', 'size'),
    ).reset_index()
    agg['age_acceleration'] = agg['predicted_age'] - agg['age']
    agg['age_group'] = np.where(agg['age'] < 50, 'Young', 'Old')
    return agg


def predict(variant, datasets, cell_types, condition=None):
    cfg = {**BASELINE, **VARIANTS[variant]}
    obs = wrapper_clock_predictions(
        cell_types, datasets,
        data_type=cfg['data_type'],
        condition=condition,
        version=CLOCK_V,
        model_dir=f'{STRESS_DIR}/{variant}/models',
        only_sig_genes=cfg['only_sig_genes'],
    )
    return median_per_donor(obs) if cfg['data_type'] != 'bulk' else obs


def train(variant):
    cfg = {**BASELINE, **VARIANTS[variant]}
    out_dir = f'{STRESS_DIR}/{variant}/models'
    for cell_type in CTS:
        print(f"\n{'='*60}\n[{variant}] training {cell_type}\n{'='*60}", flush=True)
        adata = ad.concat([
            retrieve_adata(dataset=d, data_type=cfg['data_type'], cell_type=cell_type,
                           only_sig_genes=cfg['only_sig_genes'])
            for d in CLOCK_TRAINING_COHORTS
        ])
        train_aging_clock(
            adata=adata, cell_type=cell_type, version=CLOCK_V, output_dir=out_dir,
            reg_type=cfg['reg_type'], tune_model=TUNE_CLOCK, scoring=CLOCK_CV_SCORING,
            verbose=True,
        )


# ---------------------------------------------------------------- readouts

def readout_cv(variant):
    obs = predict(variant, CLOCK_TEST_COHORTS, CTS, condition='healthy')
    rows = []
    for (ct, ds), df in obs.groupby(['cell_type', 'dataset'], observed=True):
        rows.append(dict(readout='cv', cell_type=ct, contrast=ds, n=len(df),
                         value=spearmanr(df['age'], df['predicted_age']).correlation,
                         value2=r2_score(df['age'], df['predicted_age']), pvalue=np.nan))
    return rows


def readout_sle(variant):
    obs = predict(variant, ['perez_sle'], T_CTS)
    rows = []
    for ct, df in obs.groupby('cell_type', observed=True):
        for bin_name, sub in [('all', df), ('Young', df[df.age_group == 'Young']),
                              ('Old', df[df.age_group == 'Old'])]:
            a = sub.loc[sub.condition == 'SLE', 'age_acceleration'].dropna()
            b = sub.loc[sub.condition == 'healthy', 'age_acceleration'].dropna()
            if len(a) < 3 or len(b) < 3:
                continue
            rows.append(dict(readout='sle', cell_type=ct, contrast=f'SLE_vs_healthy_{bin_name}',
                             n=len(a) + len(b), value=a.mean() - b.mean(), value2=np.nan,
                             pvalue=stats.ttest_ind(a, b, equal_var=False).pvalue))
    return rows


def readout_rejuv(variant):
    rows = []
    for dataset, ctr, treat in REJUV_CONTRASTS:
        if (variant, dataset) in SKIP_CONTRASTS:
            print(f'[{variant}] skipping {dataset}: {treat} vs {ctr}')
            continue
        conf = get_config(dataset)
        obs = predict(variant, [dataset], T_CTS, condition=[ctr, treat])
        for ct, df in obs.groupby('cell_type', observed=True):
            pval, slope = test_mixed_effects(
                df, ctr, treat, target_variable='predicted_age', config=conf,
                group_key=conf.clock_group_key or conf.mixed_effects_group)
            rows.append(dict(readout='rejuv', cell_type=ct, contrast=f'{dataset}:{treat}_vs_{ctr}',
                             n=len(df), value=slope, value2=np.nan, pvalue=pval))
    return rows


def run(variant):
    train(variant)
    rows = readout_cv(variant) + readout_sle(variant) + readout_rejuv(variant)
    df = pd.DataFrame(rows)
    df.insert(0, 'variant', variant)
    path = f'{STRESS_DIR}/readouts_{variant}.csv'
    df.to_csv(path, index=False)
    print(f'\nreadouts: {os.path.abspath(path)}')
    print(df.to_string())


# ---------------------------------------------------------------- aggregation

def aggregate():
    paths = [f'{STRESS_DIR}/readouts_{v}.csv' for v in VARIANTS]
    df = pd.concat([pd.read_csv(p) for p in paths if os.path.exists(p)])
    # cv is reported for every major cell type, the claims only for T cells
    df = df[df.cell_type.isin(CTS) & ((df.readout == 'cv') | df.cell_type.isin(T_CTS))]
    base = df[df.variant == 'baseline'].set_index(['readout', 'cell_type', 'contrast'])
    key = ['readout', 'cell_type', 'contrast']
    df = df.join(base[['value', 'pvalue']].add_prefix('base_'), on=key)
    # Does the stressed run still say the same thing as the ground truth?
    df['same_sign'] = np.sign(df.value) == np.sign(df.base_value)
    df['both_sig'] = (df.pvalue < 0.05) == (df.base_pvalue < 0.05)
    df['delta'] = df.value - df.base_value
    out = f'{STRESS_DIR}/stress_summary.csv'
    df.to_csv(out, index=False)

    print('\n=== reproduced vs baseline (non-cv readouts) ===')
    m = df[(df.variant != 'baseline') & (df.readout != 'cv')]
    print(m.groupby('variant')[['same_sign', 'both_sig']].mean().round(2).to_string())
    print(f'\nsummary: {os.path.abspath(out)}')

    plot(df)
    figure(df)


def plot(df):
    """Six angles on the same comparison: per-readout dot panels (what each variant says),
    a scatter against the baseline (does it reproduce), an agreement bar (how often), and a
    delta heatmap (where it breaks)."""
    order = [v for v in VARIANTS if v in set(df.variant)]
    stressed = [v for v in order if v != 'baseline']
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'legend.fontsize': 9})
    figs = []

    def save(fig, name):
        path = f'{STRESS_DIR}/stress_{name}.png'
        fig.savefig(path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        figs.append(os.path.abspath(path))

    def tidy(ax, i, ylab, xticks=True):
        ax.set(xlabel='', ylabel=ylab if i == 0 else '')
        ax.margins(x=0.15, y=0.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if xticks:
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order, rotation=45, ha='right')
        if ax.legend_ is not None:
            ax.legend_.remove()

    READOUTS = [('cv', 'Spearman (held-out)'), ('sle', '$\\Delta$ age acceleration'),
                ('rejuv', 'Effect on predicted age')]

    # 1-3. per-readout: every variant's value, split by cell type
    for readout, ylab in READOUTS:
        sub = df[df.readout == readout]
        if sub.empty:
            continue
        cts = sorted(sub.cell_type.unique())
        fig, axes = plt.subplots(1, len(cts), figsize=(3 * len(cts) - 1 + 1.5, 3),
                                 sharey=True, squeeze=False)
        for i, (ax, ct) in enumerate(zip(axes[0], cts)):
            sns.stripplot(data=sub[sub.cell_type == ct], x='variant', y='value', hue='contrast',
                          order=order, s=6, alpha=0.8, linewidth=0.5, edgecolor='gray', ax=ax)
            ax.axhline(0, color='gray', lw=0.5, ls='--')
            ax.set_title(ct)
            tidy(ax, i, ylab)
        axes[0][-1].legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False, title='')
        save(fig, readout)

    # 4. held-out r2 -- the CV panel above only shows rank agreement, r2 shows calibration
    cv = df[df.readout == 'cv']
    if not cv.empty:
        cts = sorted(cv.cell_type.unique())
        fig, axes = plt.subplots(1, len(cts), figsize=(3 * len(cts) - 1 + 1.5, 3),
                                 sharey=True, squeeze=False)
        for i, (ax, ct) in enumerate(zip(axes[0], cts)):
            sns.stripplot(data=cv[cv.cell_type == ct], x='variant', y='value2', hue='contrast',
                          order=order, s=6, alpha=0.8, linewidth=0.5, edgecolor='gray', ax=ax)
            ax.axhline(0, color='gray', lw=0.5, ls='--')
            ax.set_title(ct)
            tidy(ax, i, '$R^2$ (held-out)')
        axes[0][-1].legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False, title='')
        save(fig, 'cv_r2')

    # 5. variant vs baseline, identity line = perfect reproduction
    fig, axes = plt.subplots(1, len(READOUTS), figsize=(3 * len(READOUTS) - 1 + 1.5, 3),
                             squeeze=False)
    for i, (ax, (readout, ylab)) in enumerate(zip(axes[0], READOUTS)):
        sub = df[(df.readout == readout) & (df.variant != 'baseline')].dropna(subset=['base_value'])
        sns.scatterplot(data=sub, x='base_value', y='value', hue='variant', hue_order=stressed,
                        s=30, alpha=0.8, linewidth=0.5, edgecolor='gray', ax=ax)
        lim = [min(sub.base_value.min(), sub.value.min()), max(sub.base_value.max(), sub.value.max())]
        ax.plot(lim, lim, color='gray', lw=0.5, ls='--')
        ax.set_title(readout)
        ax.set_xlabel('baseline')
        tidy(ax, i, 'variant', xticks=False)
        ax.set_xlabel('baseline')
    axes[0][-1].legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False, title='')
    save(fig, 'scatter_vs_baseline')

    # 6. how often each variant agrees with the ground truth, split by readout
    agr = (df[(df.variant != 'baseline') & (df.readout != 'cv')]
           .melt(id_vars=['variant', 'readout'], value_vars=['same_sign', 'both_sig'],
                 var_name='criterion', value_name='agree'))
    fig, axes = plt.subplots(1, 2, figsize=(3 * 2 - 1 + 1.5, 3), sharey=True, squeeze=False)
    for i, (ax, crit) in enumerate(zip(axes[0], ['same_sign', 'both_sig'])):
        sns.barplot(data=agr[agr.criterion == crit], x='variant', y='agree', hue='readout',
                    order=stressed, errorbar=None, ax=ax)
        ax.set_ylim(0, 1)
        ax.set_title(crit)
        ax.set(xlabel='', ylabel='fraction agreeing with baseline' if i == 0 else '')
        ax.set_xticks(range(len(stressed)))
        ax.set_xticklabels(stressed, rotation=45, ha='right')
        ax.margins(x=0.15, y=0.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ax.legend_ is not None:
            ax.legend_.remove()
    axes[0][-1].legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False, title='')
    save(fig, 'agreement')

    # 7. where it breaks: delta per individual readout. * marks a sign flip.
    hm = df[df.readout != 'cv'].copy()
    hm['row'] = hm.readout + ' | ' + hm.cell_type + ' | ' + hm.contrast
    mat = hm.pivot_table(index='row', columns='variant', values='delta').reindex(columns=stressed)
    flip = hm.pivot_table(index='row', columns='variant', values='same_sign').reindex(columns=stressed)
    ann = np.where(flip.fillna(True).values, '', '*')
    fig, ax = plt.subplots(figsize=(3 + 0.5 * max(0, len(stressed) - 5) / 2,
                                    3 + 0.5 * max(0, len(mat) - 5) / 2))
    v = np.nanmax(np.abs(mat.values))
    sns.heatmap(mat, cmap='RdBu_r', center=0, vmin=-v, vmax=v, annot=ann, fmt='',
                annot_kws={'fontsize': 9}, cbar_kws={'label': 'value $-$ baseline'}, ax=ax)
    ax.set(xlabel='', ylabel='')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    save(fig, 'delta_heatmap')

    print('figures:\n  ' + '\n  '.join(figs))


# ponytail: panels a-c of the supplementary figure; the seven diagnostic PNGs above stay
# for inspection, this is the one meant to go in the paper.
FIG_LABEL = {'baseline': 'Baseline', 'lasso': 'Lasso', 'elasticnet': 'ElasticNet',
             'wholegenome': 'Whole genome', 'metacell': 'Metacell',
             'gradientboosting': 'Gradient boosting', 'nn': 'Neural net'}
FIG_CONTRAST = {'CXCL9:LPS_vs_RPMI': 'LPS (ctr: RPMI)',
                'op:Ruxolitinib_vs_DMSO': 'Ruxolitinib (OPSCA)',
                'parsebioscience:IL-10_vs_PBS': 'IL-10 (Parse)',
                'CXCL9:LPS + ruxolitinib_vs_LPS': 'Ruxolitinib (ctr: LPS)',
                'CXCL9:RPMI + ruxolitinib_vs_RPMI': 'Ruxolitinib (ctr: RPMI)'}
FIG_SLE = {'SLE_vs_healthy_all': 'overall', 'SLE_vs_healthy_Young': 'young (<50y)',
           'SLE_vs_healthy_Old': 'old ($\\geq$50y)'}
FIG_CT = 'CD4T'


def figure(df):
    """a: held-out accuracy per variant. b: does the SLE age acceleration survive.
    c: does the CD4T rejuvenation call survive."""
    order = [v for v in VARIANTS if v in set(df.variant)]
    stressed = [v for v in order if v != 'baseline']
    colors = dict(zip(stressed, sns.color_palette('colorblind', len(stressed))))
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9})

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.1), gridspec_kw={'width_ratios': [1.1, 1.2, 1.2]})

    ax = axes[0]
    cv = (df[df.readout == 'cv'].pivot_table(index='cell_type', columns='variant', values='value')
          .reindex(columns=order))
    sns.heatmap(cv, cmap='Blues', vmin=0, vmax=.85, annot=True, fmt='.2f', annot_kws={'fontsize': 7},
                cbar_kws={'label': 'Spearman (held-out)', 'pad': .02}, ax=ax, linewidths=.5)
    ax.set(xlabel='', ylabel='')
    ax.set_xticklabels([FIG_LABEL.get(v, v) for v in order], rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    def dots(ax, sub, rows, labels, xlabel, legend=False):
        """One row per readout: baseline diamond, one coloured dot per stressed variant."""
        for y, r in enumerate(rows):
            d = sub[sub.row == r]
            if y % 2 == 0:
                ax.axhspan(y - .5, y + .5, color='#f2f2f2', zorder=0)
            for v in stressed:
                val = d.loc[d.variant == v, 'value']
                if len(val):
                    ax.scatter(val.iloc[0], y, color=colors[v], s=32, zorder=3,
                               label=FIG_LABEL.get(v, v) if y == 0 else None)
            base = d.loc[d.variant == 'baseline', 'value']
            if len(base):
                ax.scatter(base.iloc[0], y, marker='D', color='black', s=42, zorder=4,
                           label='Baseline' if y == 0 else None)
        ax.axvline(0, color='black', lw=.8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(labels)
        ax.set_ylim(len(rows) - .5, -.5)
        ax.set_xlabel(xlabel)
        ax.spines[['top', 'right']].set_visible(False)
        if legend:
            ax.legend(loc='upper left', bbox_to_anchor=(0, -.35), ncol=3, frameon=False,
                      fontsize=7.5)

    sle = df[df.readout == 'sle'].copy()
    sle['row'] = sle.cell_type.astype(str) + '|' + sle.contrast
    rows = [r for ct in sorted(sle.cell_type.unique()) for c in FIG_SLE
            if (r := f'{ct}|{c}') in set(sle.row)]
    dots(axes[1], sle, rows, [f'{r.split("|")[0]}, {FIG_SLE[r.split("|")[1]]}' for r in rows],
         '$\\Delta$ age acceleration (yrs)\nSLE $-$ healthy', legend=True)

    rj = df[(df.readout == 'rejuv') & (df.cell_type == FIG_CT)].copy()
    rj['row'] = rj.contrast
    rows = [c for c in FIG_CONTRAST if c in set(rj.row)]
    dots(axes[2], rj, rows, [FIG_CONTRAST[c] for c in rows],
         f'$\\Delta$ predicted age (yrs), {FIG_CT}')

    for ax, letter in zip(axes, 'abc'):
        ax.set_title(letter, loc='left', weight='bold', fontsize=11, pad=6)
    fig.tight_layout()
    path = f'{STRESS_DIR}/stress_figure.png'
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'figure: {os.path.abspath(path)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=list(VARIANTS))
    parser.add_argument('--aggregate', action='store_true')
    args = parser.parse_args()
    os.makedirs(STRESS_DIR, exist_ok=True)
    if args.aggregate:
        aggregate()
    else:
        assert args.variant, 'pass --variant <name> or --aggregate'
        run(args.variant)
