
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import numpy as np
from pandas.api.types import CategoricalDtype
from statsmodels.stats.multitest import multipletests
from ciim.src.common import surrogate_names

palette_disease = {'Healthy': '#56B4E9', 'SLE': '#F0E442', 'Mild': '#2ca02c', 'Severe': '#e377c2'}



def wrapper_plot_age_acceleration_disease_bins(obs, disease_dataset):
    if disease_dataset == 'Covid_50MHH':
        disease_name = 'COVID-19'
        ctr = 'mild'
        cond = 'severe'
    elif disease_dataset == 'SLE_European':
        disease_name = 'SLE'
        ctr = 'normal'
        cond = 'systemic lupus erythematosus'
    else:
        raise ValueError("Unknown dataset")

    obs_disease = obs[obs['dataset'] == disease_dataset].copy()
    obs_disease['age'] = obs_disease['age'].astype(float)
    obs_disease['donor_id'] = obs_disease['donor_id'].astype(str)

    cell_types = obs_disease['cell_type'].unique()
    n_cell_types = len(cell_types)

    fig, axes = plt.subplots(1, n_cell_types, figsize=(1.5 * n_cell_types+2, 2.5), sharey=True)

    for i, cell_type in enumerate(cell_types):
        ax = axes[i] if len(cell_types) > 1 else axes

        obs_ct = obs_disease[obs_disease['cell_type'] == cell_type].copy()
        obs_ct = obs_ct.groupby(['donor_id', 'age', 'condition'])['predicted_age'].median().reset_index()

        age_bins = [20, 50, 80]
        obs_ct['age_bin'] = pd.cut(obs_ct['age'], bins=age_bins, right=False)
        obs_ct['age_bin'] = obs_ct['age_bin'].astype(str)
        age_bin_order = [str(b) for b in pd.cut([21, 51], bins=age_bins, right=False).categories]

        plot_data = []
        pvals = []
        positions = []

        for age_bin in age_bin_order:
            for group in [ctr, cond]:
                vals = obs_ct[(obs_ct['condition'] == group) & (obs_ct['age_bin'] == age_bin)]['predicted_age']
                for v in vals:
                    plot_data.append({'age_bin': age_bin, 'condition': group, 'predicted_age': v})

            # Store p-values for later FDR correction
            cond_vals = obs_ct[(obs_ct['condition'] == cond) & (obs_ct['age_bin'] == age_bin)]['predicted_age']
            ctr_vals = obs_ct[(obs_ct['condition'] == ctr) & (obs_ct['age_bin'] == age_bin)]['predicted_age']
            if len(cond_vals) >= 3 and len(ctr_vals) >= 3:
                tstat, pval = ttest_ind(cond_vals, ctr_vals, equal_var=False)
                pvals.append(pval)
                positions.append(age_bin)
            else:
                pvals.append(np.nan)
                positions.append(age_bin)

        plot_df = pd.DataFrame(plot_data)
        plot_df_c = plot_df.copy()
        plot_df_c['condition'] = plot_df_c['condition'].apply(lambda name: surrogate_names.get(name, name))
        sns.barplot(data=plot_df_c, x='age_bin', y='predicted_age', hue='condition', width=0.5,
                    ax=ax, palette=palette_disease, ci='sd', errorbar='sd', capsize=0.1, linewidth=.1, alpha=.8, errcolor='black', errwidth=1.5)
        # sns.stripplot(data=plot_df_c, x='age_bin', y='predicted_age', hue='condition',
        #             ax=ax, palette=palette_disease, alpha=.7)

        # Correct for multiple testing
        corrected = multipletests([p for p in pvals if not np.isnan(p)], method='bonferroni')
        corrected_pvals = dict(zip([pos for p, pos in zip(pvals, positions) if not np.isnan(p)], corrected[1]))
        # Annotate significance with full brackets and stars
        for j, age_bin in enumerate(age_bin_order):
            if age_bin in corrected_pvals:
                p = corrected_pvals[age_bin]
                if p < 0.05:
                    if p < 0.001:
                        star = '***'
                    elif p < 0.01:
                        star = '**'
                    else:
                        star = '*'

                    # Compute bracket height above error bars
                    group_data = plot_df[plot_df['age_bin'] == age_bin]
                    group_means = group_data.groupby('condition')['predicted_age'].mean()
                    group_stds = group_data.groupby('condition')['predicted_age'].std()
                    y_max = (group_means + group_stds).max() + 7  # leave more space

                    h = 2  # bracket height
                    bar_x1 = j - 0.2
                    bar_x2 = j + 0.2
                    bar_y = y_max
                    linewidth = 1

                    # Bracket: left vertical, horizontal, right vertical
                    ax.plot([bar_x1, bar_x1, bar_x2, bar_x2],
                            [bar_y - h, bar_y, bar_y, bar_y - h],
                            lw=linewidth, c='black')

                    ax.text(j, bar_y + 0.3, star, ha='center', va='bottom', fontsize=13, color='black')
        ax.set_title(cell_type, fontsize=10, weight='bold', pad=15)
        ax.set_xlabel("Age group")
        if i == 0:
            ax.set_ylabel("Predicted age")
        else:
            ax.set_ylabel("")
        ax.get_legend().remove()
        ax.spines[['top', 'right']].set_visible(False)
        ax.margins(x=0.2, y=0.1)

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Condition', frameon=False)
    plt.suptitle(f"Age shift in {disease_name}", fontsize=13, y=1.05, weight='bold')
    plt.tight_layout()
    plt.show()