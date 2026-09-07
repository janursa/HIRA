"""Do the age-associated TFs from tfa_major_b survive once naive/effector composition
(naive_ratio, see feature_association/helper.py:add_naive_ratio) is added as a covariate?
Bars are colored/grouped exactly like aging_features_count_tfa_major_b (trend: Increase/Decrease
in aging); the hatched top of each bar is the fraction of that bar lost after naive_ratio
correction.

Needs tfa_major_b_ctNaiveToEffector's stats already computed:
`python src/feature_association/run_analysis.py --analysis-name tfa_major_b_ctNaiveToEffector --analysis-mode multi-cohort --association-type continous`

Usage: python src/exp_analysis/naive_effector.py
Writes: OUTPUT_DIR/exp_analysis/naive_effector/survival.csv
        PLOTS_DIR/exp_analysis/naive_effector/survival.png
"""
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from hira.src.config import NAIVE_EFFECTOR_DIR, NAIVE_EFFECTOR_PLOTS_DIR, palette_trend_2
from hira.src.feature_association.helper import retrieve_sig_stats, NAIVE_EFFECTOR_COLS

CELL_TYPES = list(NAIVE_EFFECTOR_COLS)  # CD8T, CD4T, MONO -- the only cell types the covariate applies to
TRENDS = list(palette_trend_2)  # ['Increase in aging', 'Decrease in aging']


def sig_pairs(analysis_name):
    stats = retrieve_sig_stats(analysis_name=analysis_name)
    return set(zip(stats['cell_type'], stats['gene']))


def build_survival_table():
    baseline = retrieve_sig_stats(analysis_name='tfa_major_b')
    baseline = baseline[baseline['cell_type'].isin(CELL_TYPES)]
    baseline = baseline.drop_duplicates(subset=['cell_type', 'gene'])[['cell_type', 'gene', 'trend']]

    corrected = sig_pairs('tfa_major_b_ctNaiveToEffector')
    baseline['lost'] = ~baseline.apply(lambda r: (r['cell_type'], r['gene']) in corrected, axis=1)

    summary = baseline.groupby(['cell_type', 'trend']).agg(
        n_baseline_sig=('gene', 'size'), n_lost=('lost', 'sum')
    ).reset_index()
    summary['n_survived'] = summary['n_baseline_sig'] - summary['n_lost']
    summary['survival_pct'] = 100 * summary['n_survived'] / summary['n_baseline_sig']
    return baseline, summary


def plot_survival(summary, out_path):
    """Same grouping/coloring as aging_features_count_tfa_major_b (trend x cell_type); the
    hatched top of each bar is the fraction lost to naive_ratio correction."""
    cell_types = [ct for ct in CELL_TYPES if ct in summary['cell_type'].unique()]
    x_label_count = len(cell_types)
    fig, ax = plt.subplots(1, 1, figsize=(.3 * x_label_count + 1, 2))

    n_hue = len(TRENDS)
    group_w = 0.8
    bar_w = group_w / n_hue * 0.9
    for i, ct in enumerate(cell_types):
        for j, trend in enumerate(TRENDS):
            row = summary[(summary['cell_type'] == ct) & (summary['trend'] == trend)]
            n_survived = row['n_survived'].sum()
            n_lost = row['n_lost'].sum()
            x = i - group_w / 2 + group_w / n_hue * (j + 0.5)
            color = palette_trend_2[trend]
            ax.bar(x, n_survived, width=bar_w, color=color, alpha=.8)
            ax.bar(x, n_lost, width=bar_w, bottom=n_survived, color=color, alpha=.8,
                   hatch='///', edgecolor='black', linewidth=0)
            total = n_survived + n_lost
            if total > 0:
                pct_lost = 100 * n_lost / total
                ax.text(x, total, f'{pct_lost:.0f}%', ha='center', va='bottom', fontsize=9, fontname='Arial')

    ax.set_xticks(range(x_label_count))
    ax.set_xticklabels(cell_types, rotation=45, ha='right')
    ax.set_xlim(-0.5, x_label_count - 0.5)
    ax.set_ylabel('Significant TFs')
    ax.margins(x=0.1 if x_label_count < 5 else 0.05, y=0.15)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    handles = [mpatches.Patch(facecolor=palette_trend_2[t], alpha=.8, label=t) for t in TRENDS]
    handles.append(mpatches.Patch(facecolor='white', edgecolor='black', hatch='///',
                                   label='Attributed to Naive decline'))
    ax.legend(handles=handles, loc=(1, 0.5), frameon=False, title='Trend', fontsize=9, title_fontsize=9)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname('Arial')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    genes, summary = build_survival_table()
    csv_path = f'{NAIVE_EFFECTOR_DIR}survival.csv'
    png_path = f'{NAIVE_EFFECTOR_PLOTS_DIR}survival.png'
    summary.to_csv(csv_path, index=False)
    plot_survival(summary, png_path)
    print(summary.to_string(index=False))
    print(f'\nWrote {csv_path}')
    print(f'Wrote {png_path}')
