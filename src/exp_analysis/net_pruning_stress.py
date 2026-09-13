"""Stress test for the aging-TF findings under different GRN pruning.

First: how the three prunings reshape the consensus GRNs (topology -- edge/TF/target counts,
density; the skeleton_effect analysis). Then: whether the paper's aging-TF
claims survive being re-derived on each pruned network. `skeleton` IS the ground truth -- it is
the pipeline config (NET_SKELETON). Nothing here touches the pipeline: consensus nets are rebuilt
in memory (cache=False) and TF activities are never written to FEATURE_DATA_DIR.

Readouts (value = the headline number, value2 = a second view of the same comparison):
  n_sig      -- significant aging TFs per cell type
  ct_overlap -- per cell-type pair: Jaccard of the aging TF sets (value) and, among the shared
                ones, the fraction with the same direction (value2)
  <dataset>  -- aging vs condition, over TFs significant in both: fraction same sign (value)
                and Spearman of the effect sizes (value2), per comparison and age group.
                For the rejuvenation datasets (op, parsebioscience, CXCL9) the claim is the
                mirror image: rejuvenation means a LOW same-sign fraction.

Usage:
  python src/exp_analysis/net_pruning_stress.py --variant skeleton   # one variant, one job
  python src/exp_analysis/net_pruning_stress.py --aggregate          # topology + combine + plot
"""
import argparse
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from hira.src.config import (COARSENED_COVARIATES, CONFOUND_COVARIATES, CORR_THRESHOLD,
                             DISCOVERY_COHORTS, MAJOR_CT_LABEL, MAJOR_CTS, META_MIN_COHORT,
                             OUTPUT_DIR, get_config)
from hira.src.exp_analysis.skeleton_effect import collect as collect_topology
from hira.src.feature_association.helper import (associate_with_condition, association_with_age,
                                                 calculate_tf_activity, run_meta_analysis)
from hira.src.utils.util import coarsen, retrieve_adata, retrieve_net_consensus

OUT_DIR = f'{OUTPUT_DIR}/net_pruning_stress'
VARIANTS = {'skeleton': 'skeleton', 'promotor': 'promotor', 'unfiltered': None}
BASELINE = 'skeleton'
CTS = MAJOR_CTS
T_CTS = ['CD4T', 'CD8T']  # the rejuvenation claims are T cells only
COND_DATASETS = {'perez_sle': CTS, 'op': T_CTS, 'parsebioscience': T_CTS, 'CXCL9': T_CTS}
SLE_AGE_GROUPS = ['Both age groups', 'Younger than 50']


# ---------------------------------------------------------------- TF activity + associations

def tf_activity(adata, cell_type, net, dataset):
    sub = adata[adata.obs[MAJOR_CT_LABEL] == cell_type].copy()
    if len(sub) == 0:
        return None
    tfa = calculate_tf_activity(sub, net)
    tfa.obs['dataset'] = dataset
    return tfa


def aging_stats(nets):
    """Per-cohort partial Spearman with age, then the same Fisher meta-analysis the pipeline runs."""
    store = []
    for dataset in DISCOVERY_COHORTS:
        adata = retrieve_adata(dataset=dataset, data_type='bulk', condition='healthy')
        categorical = CONFOUND_COVARIATES.get(dataset, [])
        for cell_type, net in nets.items():
            tfa = tf_activity(adata, cell_type, net, dataset)
            if tfa is None:
                continue
            for c in categorical:
                if c not in tfa.obs.columns and c in COARSENED_COVARIATES:
                    tfa.obs[c] = coarsen(tfa.obs[COARSENED_COVARIATES[c]])
            stats = association_with_age(tfa, association_type='partial_spearman',
                                         categorical_covariates=categorical)
            stats['dataset'], stats['cell_type'] = dataset, cell_type
            store.append(stats)
            print(f'[aging] {dataset} {cell_type}: {len(stats)} TFs', flush=True)
    meta = run_meta_analysis(pd.concat(store), min_degree=META_MIN_COHORT)
    meta = meta.drop_duplicates(subset=['cell_type', 'gene'])
    # same rule as retrieve_stats(): FDR, sign agreement across cohorts, effect size
    meta['is_significant'] = ((meta['meta_p_adj'] < 0.05) & meta['sign_consistent'].astype(bool)
                              & (meta['pooled_rho'].abs() > CORR_THRESHOLD))
    return meta


def condition_stats(dataset, nets, cell_types):
    """Case-vs-control per TF (perez_sle also gets the age-stratified rows, see determine_stats_condition)."""
    config = get_config(dataset)
    adata = retrieve_adata(dataset=dataset, data_type='bulk', condition=config.treatment_groups)
    store = []
    for cell_type in cell_types:
        tfa = tf_activity(adata, cell_type, nets[cell_type], dataset)
        if tfa is None:
            continue
        stats = associate_with_condition(tfa, config)
        stats['cell_type'] = cell_type
        store.append(stats)
        print(f'[{dataset}] {cell_type}: {len(stats)} rows', flush=True)
    stats = pd.concat(store)
    stats['is_significant'] = (stats['p_value_adj'] < 0.05) & (stats['slope'].abs() > CORR_THRESHOLD)
    return stats


# ---------------------------------------------------------------- readouts

def readout_ct_overlap(aging):
    rows, sig = [], {}
    for cell_type, df in aging.groupby('cell_type', observed=True):
        sig[cell_type] = set(df.loc[df['is_significant'], 'gene'])
        rows.append(dict(readout='n_sig', cell_type=cell_type, contrast='aging',
                         n=len(df), value=len(sig[cell_type]), value2=np.nan))
    rho = aging.set_index(['cell_type', 'gene'])['pooled_rho']
    for a, b in itertools.combinations([ct for ct in CTS if ct in sig], 2):
        shared, union = sig[a] & sig[b], sig[a] | sig[b]
        same = np.mean([np.sign(rho[(a, g)]) == np.sign(rho[(b, g)]) for g in shared]) if shared else np.nan
        rows.append(dict(readout='ct_overlap', cell_type=f'{a}|{b}', contrast='jaccard',
                         n=len(shared), value=len(shared) / len(union) if union else np.nan,
                         value2=same))
    return rows


def readout_concordance(aging, cond, readout, cell_types, age_groups=None):
    """Over TFs significant in both: fraction of same-sign effects, and their rank correlation."""
    rows = []
    for cell_type in cell_types:
        a = aging[(aging['cell_type'] == cell_type) & aging['is_significant']]
        a = a.set_index('gene')['pooled_rho']
        sub = cond[(cond['cell_type'] == cell_type) & cond['is_significant']]
        for (comparison, age_group), c in sub.groupby(['comparison', 'age_group'], observed=True):
            if age_groups is not None and age_group not in age_groups:
                continue
            c = c.groupby('gene')['slope'].mean()
            shared = a.index.intersection(c.index)
            if len(shared) < 3:
                print(f'[{readout}] {cell_type} {comparison} {age_group}: '
                      f'only {len(shared)} shared TFs, skipping')
                continue
            rows.append(dict(readout=readout, cell_type=cell_type,
                             contrast=f'{comparison} | {age_group}', n=len(shared),
                             value=float(np.mean(np.sign(a[shared]) == np.sign(c[shared]))),
                             value2=spearmanr(a[shared], c[shared]).correlation))
    return rows


def run(variant):
    nets = {ct: retrieve_net_consensus(cell_type=ct, skeleton=VARIANTS[variant], cache=False)
            for ct in CTS}
    for ct, net in nets.items():
        print(f'[{variant}] {ct}: {len(net)} edges, {net["source"].nunique()} TFs', flush=True)

    # ponytail: the aging side is the expensive half and does not change when a condition
    # dataset is added -- reuse it if this variant already ran. Delete the csv to force a redo.
    aging_path = f'{OUT_DIR}/stats_aging_{variant}.csv'
    if os.path.exists(aging_path):
        print(f'reusing {aging_path}')
        aging = pd.read_csv(aging_path)
    else:
        aging = aging_stats(nets)
        aging.to_csv(aging_path, index=False)

    rows = readout_ct_overlap(aging)
    for dataset, cell_types in COND_DATASETS.items():
        cond = condition_stats(dataset, nets, cell_types)
        cond.to_csv(f'{OUT_DIR}/stats_{dataset}_{variant}.csv', index=False)
        rows += readout_concordance(aging, cond, dataset, cell_types,
                                    age_groups=SLE_AGE_GROUPS if dataset == 'perez_sle' else None)

    df = pd.DataFrame(rows)
    df.insert(0, 'variant', variant)
    path = f'{OUT_DIR}/readouts_{variant}.csv'
    df.to_csv(path, index=False)
    print(f'\nreadouts: {os.path.abspath(path)}')
    print(df.to_string())


# ---------------------------------------------------------------- aggregation

def aggregate():
    paths = [f'{OUT_DIR}/readouts_{v}.csv' for v in VARIANTS]
    df = pd.concat([pd.read_csv(p) for p in paths if os.path.exists(p)])
    key = ['readout', 'cell_type', 'contrast']
    base = df[df['variant'] == BASELINE].set_index(key)
    df = df.join(base[['value', 'value2', 'n']].add_prefix('base_'), on=key)
    df['delta'] = df['value'] - df['base_value']
    out = f'{OUT_DIR}/pruning_summary.csv'
    df.to_csv(out, index=False)

    print('\n=== deviation from the skeleton baseline ===')
    print(df[df['variant'] != BASELINE].groupby(['variant', 'readout'])['delta']
          .agg(['mean', 'min', 'max']).round(3).to_string())
    print(f'\nsummary: {os.path.abspath(out)}')

    aging = {v: pd.read_csv(f'{OUT_DIR}/stats_aging_{v}.csv') for v in VARIANTS
             if os.path.exists(f'{OUT_DIR}/stats_aging_{v}.csv')}

    topo = topology()
    figure_concordance(aging)
    figure_facts(df)
    figure_overview(topo, aging, df)


def topology():
    """Rebuild every (cell type, pruning) consensus net and describe its shape.
    Reuses skeleton_effect.collect; variant names are remapped to this module's."""
    stats, _ = collect_topology()
    stats['variant'] = stats['variant'].map({'Unfiltered': 'unfiltered',
                                             'Promoter filtered': 'promotor',
                                             'Skeleton filtered': 'skeleton'})
    stats.to_csv(f'{OUT_DIR}/topology_stats.csv', index=False)
    print(stats.round(5).to_string(index=False))
    return stats


FIG_LABEL = {'skeleton': 'Skeleton', 'promotor': 'Promoter only', 'unfiltered': 'Unfiltered'}
VARIANT_COLOR = {'skeleton': '#D55E00', 'promotor': '#0072B2', 'unfiltered': '#999999'}
# the three settings agree closely -- offset them within the row or they hide each other
VARIANT_DY = {'skeleton': 0.0, 'promotor': -0.18, 'unfiltered': 0.18}
AGE_LABEL = {'Both age groups': 'all donors', 'All age groups': 'all donors',
             'Younger than 50': 'young (<50y)'}
CONTRAST_LABEL = {'op:Ruxolitinib': 'Ruxolitinib (OPSCA)',
                  'parsebioscience:IL-10': 'IL-10 (Parse)',
                  'CXCL9:Ruxolitinib (ctr: RPMI)': 'Ruxolitinib (ctr: RPMI)',
                  'CXCL9:Ruxolitinib (ctr: LPS)': 'Ruxolitinib (ctr: LPS)',
                  'CXCL9:LPS (ctr: RPMI)': 'LPS (ctr: RPMI)'}
SIG_LINE = -np.log10(0.05)
MIN_SHARED_TFS = 10  # contrasts with fewer shared TFs than this are not plotted
# the cell types each claim is made on -- figure only, the stats are computed for all of them
FIG_CTS = {'perez_sle': T_CTS, 'parsebioscience': T_CTS, 'op': ['CD4T'], 'CXCL9': ['CD4T']}


# ---------------------------------------------------------------- figure 1: sig TF concordance

def concordance_table(ref, var):
    """Signed significance in both settings for the TFs significant in either and present in both
    nets (a TF missing from one net has no score to compare and is dropped)."""
    signed = lambda d: -np.log10(d['meta_p_adj'] + 1e-300) * np.sign(d['pooled_rho'])
    ref, var = ref.set_index('gene'), var.set_index('gene')
    union = sorted((set(ref.index[ref['is_significant']]) | set(var.index[var['is_significant']])))
    both = [g for g in union if g in ref.index and g in var.index]  # a TF can be absent from a net
    ref, var = ref.loc[both], var.loc[both]
    return pd.DataFrame({
        'x': signed(var), 'y': signed(ref),
        'consistent': np.sign(var['pooled_rho']) == np.sign(ref['pooled_rho']),
        'sig_both': ref['is_significant'] & var['is_significant'],
        'missing': len(union) - len(both),
    })


def draw_concordance(axes, aging):
    """Aging TFs significant under either pruning, scored under both. One panel per cell type,
    one row per pruning. Same geometry as plot_directional_consistency_scatter."""
    stressed = [v for v in VARIANTS if v != BASELINE and v in aging]
    cts = [ct for ct in CTS if ct in set(aging[BASELINE]['cell_type'])]
    for row, variant in enumerate(stressed):
        for col, cell_type in enumerate(cts):
            ax = axes[row][col]
            pick = lambda d: d[d['cell_type'] == cell_type]
            tab = concordance_table(pick(aging[BASELINE]), pick(aging[variant]))
            counts = {}
            for keep, color, edge, name in [(False, 'indianred', 'darkred', 'Opposing'),
                                            (True, 'darkseagreen', 'darkgreen', 'Consistent')]:
                sub = tab[tab['consistent'] == keep]
                ax.scatter(sub['x'], sub['y'], s=3, alpha=0.6, linewidths=0.2, c=color,
                           edgecolors=edge, label=name if (row, col) == (0, 0) else None)
                counts[keep] = len(sub)
            ax.text(0.03, 0.97, f'{counts[True]} consistent\n{counts[False]} opposing', va='top',
                    transform=ax.transAxes, fontsize=5.5)
            for line in (-SIG_LINE, SIG_LINE):
                ax.axhline(line, color='gray', lw=0.3, ls='--')
                ax.axvline(line, color='gray', lw=0.3, ls='--')
            ax.axhline(0, color='black', lw=0.5, alpha=0.5)
            ax.axvline(0, color='black', lw=0.5, alpha=0.5)
            ax.set_xlabel(f'{FIG_LABEL[variant]} (significance)' if col == 0 else '', fontsize=7)
            ax.set_ylabel(f'{FIG_LABEL[BASELINE]} (significance)' if col == 0 else '', fontsize=7)
            ax.set_title(cell_type, pad=3, fontsize=8)
            ax.margins(0.12)
            ax.locator_params(nbins=4)
            ax.tick_params(labelsize=6, length=2, pad=1)
            ax.spines[['top', 'right']].set_visible(False)
    return axes[0][0].get_legend_handles_labels()


def bracket(container, axes, label, letter=None, pad=0.04, fontsize=9):
    """One labelled bracket spanning a group of panels. Call after the layout is final --
    positions are read off the axes. `container` is a figure or a subfigure."""
    boxes = [a.get_position() for a in axes]
    x0, x1 = min(b.x0 for b in boxes), max(b.x1 for b in boxes)
    y = max(b.y1 for b in boxes) + pad
    container.add_artist(plt.Line2D([x0, x0, x1, x1], [y - 0.008, y, y, y - 0.008],
                                    color='black', lw=0.8))
    container.text((x0 + x1) / 2, y + 0.006, label, ha='center', va='bottom', fontsize=fontsize)
    if letter:
        container.text(0.01, y, letter, ha='left', va='bottom', weight='bold', fontsize=12)


def concordance_rows(aging):
    stressed = [v for v in VARIANTS if v != BASELINE and v in aging]
    cts = [ct for ct in CTS if ct in set(aging[BASELINE]['cell_type'])]
    return stressed, cts


def figure_concordance(aging):
    stressed, cts = concordance_rows(aging)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 7})
    fig, axes = plt.subplots(len(stressed), len(cts), figsize=(6, 4), squeeze=False)
    handles = draw_concordance(axes, aging)
    fig.legend(*handles, loc='lower center', ncol=2, frameon=False, fontsize=7,
               markerscale=3.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    return _save(fig, 'pruning_concordance.png')


# ---------------------------------------------------------------- figure 2: the paper's facts

def prep_facts(df):
    """Keep the contrasts each claim is actually made on, label them and assign them a panel."""
    df = df[[r in FIG_CTS and c in FIG_CTS[r] for r, c in zip(df['readout'], df['cell_type'])]].copy()
    # ponytail: a fraction over a handful of TFs is noise -- drop the contrast, don't plot it
    thin = df['base_n'] < MIN_SHARED_TFS
    if thin.any():
        print(f'dropping contrasts with < {MIN_SHARED_TFS} shared TFs:\n'
              f'{df.loc[thin, ["readout", "cell_type", "contrast", "base_n"]].drop_duplicates().to_string(index=False)}')
        df = df[~thin]
    df[['comparison', 'age_group']] = df['contrast'].str.split(' | ', regex=False, expand=True)
    df['label'] = np.where(
        df['readout'] == 'perez_sle',
        df['cell_type'] + ', SLE, ' + df['age_group'].map(AGE_LABEL).fillna(df['age_group']),
        df['cell_type'] + ', ' + (df['readout'] + ':' + df['comparison'])
        .map(CONTRAST_LABEL).fillna(df['comparison']))

    # SLE and LPS are claimed to accelerate (effects in the aging direction), ruxolitinib and
    # IL-10 to rejuvenate (effects reversed). Each panel plots the fraction supporting its claim,
    # so in both a high value means the claim holds.
    df['panel'] = np.where(df['readout'].eq('perez_sle') | df['comparison'].str.startswith('LPS'),
                           'a', 'b')
    return df


PANELS = [('a', False, 'Aging TFs with the same sign\nunder the condition (acceleration)'),
          ('b', True, 'Aging TFs reversed by\nthe treatment (rejuvenation)')]


def draw_facts(axes, df, fontsize=7):
    """Same dot layout as the clock stress figure: one row per contrast, baseline diamond,
    one coloured dot per pruning variant. a: what accelerates. b: what rejuvenates."""
    stressed = [v for v in VARIANTS if v != BASELINE and v in set(df['variant'])]
    for ax, (panel, flip, xlab) in zip(axes, PANELS):
        sub = df[df['panel'] == panel].copy()
        if flip:
            sub['value'] = 1 - sub['value']
        rows = list(dict.fromkeys(sub.sort_values(['cell_type', 'readout', 'contrast'])['label']))
        for y, r in enumerate(rows):
            d = sub[sub['label'] == r]
            if y % 2 == 0:
                ax.axhspan(y - .5, y + .5, color='#f2f2f2', zorder=0)
            for v in stressed:
                val = d.loc[d['variant'] == v, 'value']
                if len(val):
                    ax.scatter(val.iloc[0], y + VARIANT_DY[v], color=VARIANT_COLOR[v], s=26,
                               zorder=3, edgecolor='white', linewidth=0.4,
                               label=FIG_LABEL[v] if y == 0 else None)
            base = d.loc[d['variant'] == BASELINE, 'value']
            if len(base):
                ax.scatter(base.iloc[0], y + VARIANT_DY[BASELINE], marker='D',
                           color=VARIANT_COLOR[BASELINE], s=30,
                           zorder=4, edgecolor='black', linewidth=0.4,
                           label=f'{FIG_LABEL[BASELINE]} (pipeline)' if y == 0 else None)
        ax.axvline(0.5, color='gray', lw=0.8, ls='--')
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=fontsize)
        ax.set_ylim(len(rows) - .5, -.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel(xlab, fontsize=fontsize)
        ax.tick_params(labelsize=fontsize, length=2, pad=1)
        ax.spines[['top', 'right']].set_visible(False)
    return axes[-1].get_legend_handles_labels()


def figure_facts(df):
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 7})
    fig, axes = plt.subplots(1, 2, figsize=(5, 4))
    handles = draw_facts(axes, prep_facts(df))
    axes[-1].legend(*handles, loc='upper left', bbox_to_anchor=(1, 1), frameon=False, fontsize=6)
    fig.tight_layout()
    return _save(fig, 'pruning_facts.png')


# ---------------------------------------------------------------- figure 3: topology + stress

METRICS = [('n_edges', 'Edges'), ('n_tfs', 'TFs'), ('n_targets', 'Targets'), ('density', 'Density')]


def draw_topology(axes, stats, fontsize=8):
    order = [v for v in FIG_LABEL]
    for ax, (col, label) in zip(axes, METRICS):
        wide = stats.pivot(index='cell_type', columns='variant', values=col).loc[CTS, order]
        x = np.arange(len(CTS))
        for i, variant in enumerate(order):
            ax.bar(x + (i - 1) * 0.28, wide[variant], width=0.28, color=VARIANT_COLOR[variant])
        ax.set_xticks(x)
        ax.set_xticklabels(CTS, rotation=45, ha='right')
        ax.set_ylabel(label, fontsize=fontsize)
        ax.tick_params(labelsize=fontsize, length=2, pad=1)
        ax.margins(x=0.05, y=0.15)
        ax.spines[['top', 'right']].set_visible(False)


def figure_overview(topo, aging, df):
    """One figure, top to bottom: what pruning does to the networks, what it does to the aging
    TFs themselves, and what it does to the paper's condition claims."""
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 8})
    fig = plt.figure(figsize=(11, 7.8))
    sf_top, sf_conc, sf_facts = fig.subfigures(3, 1, height_ratios=[1.9, 3.4, 2.0])

    taxes = sf_top.subplots(1, 4)
    draw_topology(taxes, topo)
    var_h = [plt.Line2D([], [], marker='s', ls='', color=VARIANT_COLOR[v], label=FIG_LABEL[v])
             for v in FIG_LABEL]
    taxes[-1].legend(handles=var_h, frameon=False, fontsize=7, loc='upper left',
                     bbox_to_anchor=(1, 1.05))
    # the legend sits outside the last axis -- keep the axes+legend block centred in the figure
    sf_top.subplots_adjust(wspace=0.55, left=0.13, right=0.73, bottom=0.34, top=0.86)
    bracket(sf_top, taxes, 'Effect of pruning on GRN topology', 'A')

    stressed, cts = concordance_rows(aging)
    caxes = sf_conc.subplots(len(stressed), len(cts), squeeze=False)
    handles = draw_concordance(caxes, aging)
    sf_conc.legend(*handles, loc='lower center', ncol=2, frameon=False, fontsize=7,
                   markerscale=3.5, bbox_to_anchor=(0.5, 0.0))
    sf_conc.subplots_adjust(wspace=0.45, hspace=0.7, left=0.14, right=0.80, bottom=0.16, top=0.86)
    bracket(sf_conc, caxes.ravel(), 'Effect of pruning on the aging TFs', 'B', pad=0.055)

    faxes = sf_facts.subplots(1, 2)
    handles = draw_facts(faxes, prep_facts(df))
    faxes[-1].legend(*handles, loc='upper left', bbox_to_anchor=(1, 1), frameon=False, fontsize=7)
    sf_facts.subplots_adjust(wspace=1.5, left=0.22, right=0.60, bottom=0.34, top=0.84)
    bracket(sf_facts, faxes, "Effect of pruning on the paper's findings", 'C')
    return _save(fig, 'pruning_overview.png')


def _save(fig, name):
    path = f'{OUT_DIR}/{name}'
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'figure: {os.path.abspath(path)}')
    return path


def selftest():
    close = lambda x, y: abs(x - y) < 1e-9
    aging = pd.DataFrame({
        'cell_type': ['CD4T'] * 4 + ['CD8T'] * 4,
        'gene': ['A', 'B', 'C', 'D'] * 2,
        'pooled_rho': [0.5, -0.5, 0.4, 0.02, 0.5, 0.5, 0.4, 0.3],
        'meta_p_adj': [1e-4] * 8,
        'is_significant': [True, True, True, False] + [True] * 4,
    })
    ov = {(r['readout'], r['cell_type']): r for r in readout_ct_overlap(aging)}
    assert ov[('n_sig', 'CD4T')]['value'] == 3 and ov[('n_sig', 'CD8T')]['value'] == 4
    pair = ov[('ct_overlap', 'CD4T|CD8T')]
    assert close(pair['value'], 3 / 4) and close(pair['value2'], 2 / 3)  # B flips sign

    cond = pd.DataFrame({
        'cell_type': ['CD4T'] * 4, 'gene': ['A', 'B', 'C', 'D'], 'slope': [0.5] * 4,
        'comparison': ['SLE'] * 4, 'age_group': ['all'] * 4, 'is_significant': [True] * 4,
    })
    (row,) = readout_concordance(aging, cond, 'sle', ['CD4T'])
    assert row['n'] == 3 and close(row['value'], 2 / 3)  # D not aging-significant, B flipped

    ref = pd.DataFrame({'gene': ['A', 'B', 'C', 'E'], 'pooled_rho': [0.5, -0.5, 0.4, 0.4],
                        'meta_p_adj': [1e-4, 1e-4, 0.5, 1e-4],
                        'is_significant': [True, True, False, True]})
    var = pd.DataFrame({'gene': ['A', 'B', 'C'], 'pooled_rho': [0.3, 0.3, 0.4],
                        'meta_p_adj': [1e-2, 1e-2, 1e-4], 'is_significant': [True, False, True]})
    tab = concordance_table(ref, var)
    assert list(tab.index) == ['A', 'B', 'C']  # C sig in the variant only, E absent from the net
    assert list(tab['consistent']) == [True, False, True]
    assert list(tab['sig_both']) == [True, False, False] and tab['missing'].iloc[0] == 1
    assert close(tab.loc['A', 'x'], 2)
    print('selftest ok')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=list(VARIANTS))
    parser.add_argument('--aggregate', action='store_true')
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        os.makedirs(OUT_DIR, exist_ok=True)
        if args.aggregate:
            aggregate()
        else:
            assert args.variant, 'pass --variant <name> or --aggregate'
            run(args.variant)
