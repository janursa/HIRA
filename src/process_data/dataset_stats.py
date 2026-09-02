"""Cohort composition summary table and figures for the supplement.

Usage: python src/process_data/dataset_stats.py
Writes into PLOTS_DIR: datasets_summary_table.png, datasets_summary_{Cells,Donors}.png,
datasets_age_distribution.png, sex_distribution.png, race_composition.png,
disease_composition_<dataset>.png
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from hira.src.config import (
    AGING_COHORTS,
    PLOTS_DIR,
    colors_blind,
    palette_datasets_pretty,
    palette_genders,
    surrogate_names,
)
from hira.src.utils.util import retrieve_adata

SEX_LABELS = {'F': 'Female', 'M': 'Male'}


def collect_stats(cohorts=AGING_COHORTS):
    """Donor/cell/gene counts per cohort (healthy donors only)."""
    rows = []
    for dataset in cohorts:
        adata = retrieve_adata(dataset=dataset, condition='healthy')
        rows.append({
            'Cohort': surrogate_names[dataset],
            'Donors': adata.obs['donor_id'].nunique(),
            'Cells': adata.obs['cell_count'].sum(),
            'Genes': adata.n_vars,
        })
    return pd.DataFrame(rows)


def collect_donors(cohorts):
    """One row per unique donor-age, with a harmonised `sex` column."""
    store = []
    for dataset in cohorts:
        obs = retrieve_adata(dataset).obs.copy()
        obs['donor_age'] = obs['donor_id'].astype(str) + '_' + obs['age'].astype(str)
        obs['sex'] = obs['sex'].map(lambda name: SEX_LABELS.get(name, name))
        obs = obs.drop_duplicates(subset=['donor_age'])
        obs['dataset'] = dataset
        store.append(obs)
    return pd.concat(store)


def plot_summary_table(df, drop=('Zhang',)):
    df = df[~df['Cohort'].isin(drop)]
    fig, ax = plt.subplots(figsize=(3.5, .5))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#D3D3D3')
    for cell in table.get_celld().values():
        cell.set_linewidth(0.1)
    return _save('datasets_summary_table.png')


def plot_counts(df, cols=('Cells', 'Donors')):
    for i, metric in enumerate(cols):
        fig, ax = plt.subplots(figsize=(2, 2))
        sns.barplot(data=df, x='Cohort', y=metric, alpha=.7, ax=ax, hue='Cohort',
                    width=.7, palette=palette_datasets_pretty, legend=False)
        for p in ax.patches:
            value = int(p.get_height())
            label = f'{value/1e6:.1f}M' if value > 100000 else f'{value}'
            ax.text(p.get_x() + p.get_width() / 2, p.get_height(), label,
                    ha='center', va='bottom', fontsize=9, rotation=45)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.margins(x=0.1, y=0.3)
        ax.set_ylabel('Count' if i == 0 else '')
        ax.set_xlabel('')
        plt.tight_layout()
        _save(f'datasets_summary_{metric}.png', dpi=200)


def _binned_sex_counts(df):
    counts = df.groupby(['age', 'sex'])['donor_age'].nunique().reset_index(name='count')
    pivot = counts.pivot(index='age', columns='sex', values='count').fillna(0)
    pivot['binned'] = pd.cut(pivot.index, bins=10)
    return counts, pivot.groupby('binned').sum()


def _style_age_axis(ax, binned):
    ax.set_xlabel('Age bins')
    ax.set_xticklabels([int(interval.mid) for interval in binned.index.categories], rotation=45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_age_distribution(df, age_min=20):
    """Pooled age x sex distribution across all cohorts."""
    _, binned = _binned_sex_counts(df[df['age'] > age_min])
    fig, ax = plt.subplots(figsize=(2.7, 1.7))
    binned.plot(kind='bar', stacked=True, ax=ax, color=palette_genders, alpha=0.7, width=.8)
    _style_age_axis(ax, binned)
    ax.set_ylabel('Sample')
    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min - .5, x_max + .5)
    ax.legend(loc=[1.01, .5], frameon=False, title='Gender')
    return _save('datasets_age_distribution.png')


def plot_sex_distribution(df, cohorts):
    """Per-cohort age x sex distribution."""
    fig, axes = plt.subplots(1, len(cohorts), figsize=(3 * len(cohorts), 2), sharey=False)
    for i, dataset in enumerate(cohorts):
        _, binned = _binned_sex_counts(df[df['dataset'] == dataset])
        binned.plot(kind='bar', stacked=True, ax=axes[i], color=palette_genders, alpha=0.7, width=.8)
        _style_age_axis(axes[i], binned)
        axes[i].set_ylabel('Sample (donor-age)' if i == 0 else '')
        axes[i].set_title(surrogate_names[dataset], pad=15)
        x_min, x_max = axes[i].get_xlim()
        axes[i].set_xlim(x_min - 1, x_max + 1)
        axes[i].get_legend().remove()
    axes[-1].legend(loc=(1.02, .5), frameon=False, title='Gender')
    return _save('sex_distribution.png', dpi=200)


def plot_ethnicity(cohorts=AGING_COHORTS, min_pct=2.0):
    """Nested rings, one per cohort, of donor ethnicity."""
    per_dataset, all_races = [], set()
    for dataset in cohorts:
        obs = retrieve_adata(dataset).obs
        race = obs['race'] if 'race' in obs.columns else pd.Series('European', index=obs.index)
        df = obs.assign(race=race).groupby('race')['donor_id'].nunique().reset_index(name='count')
        df = df[df['count'] / df['count'].sum() * 100 >= min_pct]
        if len(df):
            per_dataset.append((dataset, df))
            all_races.update(df['race'])

    all_races = sorted(all_races)
    race_colors = {'European': colors_blind[1]}
    palette = sns.color_palette('tab20', n_colors=len(all_races))
    for race in all_races:
        race_colors.setdefault(race, palette[len(race_colors)])

    outer_radius, ring_width = 1.0, 0.15
    fig, ax = plt.subplots(figsize=(4, 3))
    for i, (dataset, df) in enumerate(per_dataset):
        df = df.sort_values('race')
        radius = outer_radius - i * ring_width
        ax.pie(df['count'], radius=radius, colors=[race_colors[r] for r in df['race']],
               startangle=90, counterclock=False,
               wedgeprops=dict(width=ring_width, edgecolor='white', linewidth=1))
        ax.text(-1, radius - ring_width / 2 - 0.05, surrogate_names[dataset],
                ha='right', va='center', fontsize=9, weight='bold')
    ax.set(aspect='equal')
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=race_colors[r], markersize=10)
               for r in all_races]
    ax.legend(handles, all_races, loc='center left', bbox_to_anchor=(.9, 0.5),
              frameon=False, fontsize=9, labelspacing=0.5)
    return _save('race_composition.png', dpi=200)


def plot_disease(dataset='perez_sle'):
    obs = retrieve_adata(dataset).obs
    info = obs.groupby('condition')['donor_id'].nunique().reset_index(name='count')
    info['condition'] = info['condition'].replace({'healthy': 'Control'})
    colors = {'Control': colors_blind[0], 'SLE': colors_blind[2]}
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.pie(info['count'], labels=info['condition'], autopct='%1.1f%%',
           colors=[colors[c] for c in info['condition']])
    return _save(f'disease_composition_{dataset}.png', dpi=200)


def _save(name, dpi=300):
    file_name = os.path.join(PLOTS_DIR, name)
    print(f'Saving to {file_name}')
    plt.savefig(file_name, bbox_inches='tight', dpi=dpi, transparent=True)
    plt.close()
    return file_name


if __name__ == '__main__':
    os.makedirs(PLOTS_DIR, exist_ok=True)

    stats = collect_stats()
    print(stats.to_string(index=False))
    print(f"Total: {stats['Cells'].sum()} cells, {stats['Donors'].sum()} donors")
    plot_summary_table(stats)
    plot_counts(stats)

    # AGING_COHORTS already includes zhang; the notebook appended it a second time.
    cohorts = AGING_COHORTS
    donors = collect_donors(cohorts)
    plot_age_distribution(donors)
    plot_sex_distribution(donors, cohorts)

    plot_ethnicity()
    plot_disease()
