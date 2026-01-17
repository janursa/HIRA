
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from hiara.src.clock.plots import plot_scatter_age_vs_predictedAge
from sklearn.metrics import r2_score
from hiara import CLOCK_TEST_COHORTS, wrapper_clock_predictions, PLOTS_DIR, CELL_TYPES, surrogate_names, palette_datasets_pretty, colors_blind

obs = wrapper_clock_predictions(CELL_TYPES, CLOCK_TEST_COHORTS, condition='healthy')

fold_scores = {}
all_preds = []
for (cell_type_name, dataset_name), df in obs.groupby(['cell_type', 'dataset']):
    y_true = df['age'].values
    y_pred = df['predicted_age'].values
    spearman_corr = spearmanr(y_true, y_pred).correlation
    r2 = r2_score(y_true, y_pred)
    fold_scores[(cell_type_name, dataset_name)] = {
        'spearman': round(spearman_corr, 3),
        'r2': round(r2, 3)
    }
    pred_df = df.copy()
    all_preds.append(pred_df)

scores_df = pd.DataFrame.from_dict(fold_scores, orient='index')
scores_df.index = pd.MultiIndex.from_tuples(scores_df.index, names=['cell_type', 'dataset'])
predictions_df = pd.concat(all_preds)
scores_df = scores_df.reset_index()  # make cell_type and dataset columns
scores_df['dataset'] = scores_df['dataset'].map(lambda name: surrogate_names.get(name, name))

def plot_scores(cv_scores, metric, figsize=[2.5, 2]):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.stripplot(
        data=cv_scores, x='cell_type', y=metric,
        hue='dataset', dodge=False, jitter=False, s=7,
        alpha=0.7, linewidth=0.5, edgecolor='gray', palette=palette_datasets_pretty,
        ax=ax
    )
    sns.barplot(
        data=cv_scores, x='cell_type', y=metric,
        estimator='mean', errorbar=None,
        color=colors_blind[1], alpha=0.5, ax=ax
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.margins(x=0.1, y=0.1)
    ax.spines[['top', 'right']].set_visible(False)
    # Legend
    ax.legend(title='', bbox_to_anchor=(.9, 1), loc='upper left', frameon=False)
    ax.set(
        ylabel=metric.capitalize(),
        xlabel=''
    )
    return fig

# Plot R²
scores_df['cell_type'] = pd.Categorical(scores_df['cell_type'], categories=CELL_TYPES, ordered=True)
fig = plot_scores(scores_df, 'r2', figsize=(2.2, 2))
file_name = f'{PLOTS_DIR}/clock_validation_r2.png'
print('r2 scores fig: ',file_name)
fig.savefig(file_name,
            bbox_inches='tight', dpi=300, transparent=True)

# Plot Spearman
fig = plot_scores(scores_df, 'spearman', figsize=(2.2, 2))
file_name = f'{PLOTS_DIR}/clock_validation_spearman.png'
print('spearman scores fig: ',file_name)
fig.savefig(file_name,
            bbox_inches='tight', dpi=300, transparent=True)

predictions_df['dataset'] = predictions_df['dataset'].apply(lambda name: surrogate_names.get(name, name))

for cell_type in CELL_TYPES:
    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5), sharey=True)
    df = predictions_df[predictions_df['cell_type'] == cell_type]
    plot_scatter_age_vs_predictedAge(df, dataset='', ax=ax, hue='dataset', palette=palette_datasets_pretty, s=30, alpha=0.7)
    ax.legend(loc=(1.1, .2), frameon=False, title='Dataset')
    file_name = f'{PLOTS_DIR}/clock_scatter_{cell_type}_all_datasets.png'
    print(f'Scatter plot for {cell_type}: ', file_name)
    fig.savefig(file_name, bbox_inches='tight', dpi=300, transparent=True)