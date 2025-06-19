
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import numpy as np
from pandas.api.types import CategoricalDtype
from statsmodels.stats.multitest import multipletests
from ciim.src.common import surrogate_names

palette_disease = {'Healthy': '#56B4E9', 'SLE': '#F0E442', 'Mild': '#2ca02c', 'Severe': '#e377c2'}
def wrapper_age_acceleration_disease(obs, disease_dataset):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_ind
    from statsmodels.stats.multitest import multipletests
    import numpy as np

    if disease_dataset == 'Covid_50MHH':
        disease_name = 'COVID-19'
        ctr = 'mild'
        cond = 'severe'
    elif disease_dataset == 'SLE_European':
        disease_name = 'SLE'
        ctr = 'normal'
        cond = 'systemic lupus erythematosus'

    obs_disease = obs[obs['dataset'] == disease_dataset].copy()
    obs_disease['age'] = obs_disease['age'].astype(float)
    obs_disease['donor_id'] = obs_disease['donor_id'].astype(str)
    obs_disease['cell_type'] = obs_disease['cell_type'].astype(str)
    obs_disease['condition'] = obs_disease['condition'].astype(str)

    # Only keep relevant conditions
    obs_disease = obs_disease[obs_disease['condition'].isin([ctr, cond])]
    obs_disease['cell_type'] = pd.Categorical(obs_disease['cell_type'], categories=cell_types, ordered=True)

    fig, ax = plt.subplots(figsize=(4, 2.5))

    # Prepare a grouping variable for (cell_type, condition)
    obs_disease['group'] = obs_disease['cell_type'].astype(str) + ' | ' + obs_disease['condition']

    obs_disease_c = obs_disease.copy()
    obs_disease_c['condition'] = obs_disease_c['condition'].apply(lambda x: surrogate_names.get(x, x))
    if disease_dataset == 'Covid_50MHH':
        obs_disease_c['condition'] = obs_disease_c['condition'].astype(CategoricalDtype(categories=['Mild', 'Severe'], ordered=True))
    elif disease_dataset == 'SLE_European':
        obs_disease_c['condition'] = obs_disease_c['condition'].astype(CategoricalDtype(categories=['Healthy', 'SLE'], ordered=True))
    # Plot stripplot
    sns.stripplot(
        data=obs_disease_c,
        x='cell_type', y='predicted_age', hue='condition',
        dodge=True, jitter=True, alpha=0.7, ax=ax, palette='Set2', size=5
    )

    # Statistical test and brackets
    pvals = []
    brackets = []
    for i, cell_type in enumerate(cell_types):
        data_ctr = obs_disease[(obs_disease['cell_type'] == cell_type) & (obs_disease['condition'] == ctr)]['predicted_age']
        data_cond = obs_disease[(obs_disease['cell_type'] == cell_type) & (obs_disease['condition'] == cond)]['predicted_age']

        if len(data_ctr) > 1 and len(data_cond) > 1:
            _, pval = ttest_ind(data_ctr, data_cond, equal_var=False)
        else:
            pval = 1.0

        pvals.append(pval)

        # Save bracket info: x positions, height, and p-value
        if len(data_ctr) > 0 and len(data_cond) > 0:
            x1, x2 = i - 0.2, i + 0.2
            y_max = max(data_ctr.max(), data_cond.max())
            brackets.append((x1, x2, y_max + 3, pval))

    # FDR correction
    _, pvals_fdr, _, _ = multipletests(pvals, method='fdr_bh')

    # Add brackets and stars
    for i, (x1, x2, y, pval_fdr) in enumerate(zip([b[0] for b in brackets], [b[1] for b in brackets], [b[2] for b in brackets], pvals_fdr)):
        star = '***' if pval_fdr < 0.001 else '**' if pval_fdr < 0.01 else '*' if pval_fdr < 0.05 else ''
        bracket_height = 2.5  # You can adjust this value as needed
        y_bracket = y + 4
        if star:
            ax.plot([x1, x1, x2, x2],
                    [y_bracket, y_bracket + bracket_height, y_bracket + bracket_height, y_bracket],
                    lw=1.5, color='black')
            ax.text((x1 + x2) / 2,
                    y_bracket + bracket_height + 1,
                    star,
                    ha='center', va='bottom', fontsize=10)

    ax.set_ylabel("Predicted age")
    ax.set_xlabel("")
    ax.set_xticklabels(cell_types, rotation=45)
    ax.spines[['top', 'right']].set_visible(False)
    ax.margins(x=0.1, y=0.1)
    ax.legend(title='Condition', bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
    ax.set_title(f"{disease_name}", fontsize=12, weight='bold', pad=15)
    plt.tight_layout()

def wrapper_age_acceleration_disease(obs, disease_dataset):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_1samp
    from statsmodels.stats.multitest import multipletests
    import numpy as np

    if disease_dataset == 'Covid_50MHH':
        disease_name = 'COVID-19'
        ctr = 'mild'
        cond = 'severe'
    elif disease_dataset == 'SLE_European':
        disease_name = 'SLE'
        ctr = 'normal'
        cond = 'systemic lupus erythematosus'

    obs_disease = obs[obs['dataset'] == disease_dataset].copy()

    obs_disease['age'] = obs_disease['age'].astype(float)
    obs_disease['donor_id'] = obs_disease['donor_id'].astype(str)
    obs_disease['cell_type'] = obs_disease['cell_type'].astype(str)
    obs_disease['condition'] = obs_disease['condition'].astype(str)

    df_pivot = obs_disease.pivot_table(index=['age', 'cell_type'], columns='condition', values='predicted_age')
    df_pivot = df_pivot.reset_index()
    df_pivot['diff'] = df_pivot[cond] - df_pivot[ctr]
    df_pivot = df_pivot[~df_pivot['diff'].isna()]
    df_pivot['cell_type'] = pd.Categorical(df_pivot['cell_type'], categories=cell_types, ordered=True)

    fig, ax = plt.subplots(figsize=(4, 2.7))

    # Stripplot
    sns.stripplot(
        ax=ax, data=df_pivot, y='diff', x='cell_type',
        alpha=0.7, hue='age', palette='viridis', s=6
    )

    # Bar plot for medians (frame-only bars)
    medians = df_pivot.groupby('cell_type')['diff'].median().reindex(cell_types)
    bar_x = np.arange(len(cell_types))
    ax.bar(
        bar_x, medians, width=0.5, fill=False, edgecolor='black', linewidth=1.5,
        zorder=0.1, label='Median', hatch='////'
    )

    # Statistical tests (unpaired)
    pvals = []
    mean_diffs = []
    for cell_type in cell_types:
        group = obs_disease[obs_disease['cell_type'] == cell_type]
        group_ctr = group[group['condition'] == ctr]['predicted_age']
        group_cond = group[group['condition'] == cond]['predicted_age']

        if len(group_ctr) > 1 and len(group_cond) > 1:
            from scipy.stats import ttest_ind
            _, pval = ttest_ind(group_cond, group_ctr, equal_var=False)  # Welch’s t-test
            mean_diff = group_cond.mean() - group_ctr.mean()
        else:
            pval = 1.0
            mean_diff = np.nan

        pvals.append(pval)
        mean_diffs.append(mean_diff)

    _, pvals_fdr, _, _ = multipletests(pvals, method='fdr_bh')
    for i, (mean_diff, pval_fdr) in enumerate(zip(mean_diffs, pvals_fdr)):
        
        star = '***' if pval_fdr < 0.001 else '**' if pval_fdr < 0.01 else '*' if pval_fdr < 0.05 else ''
        text = f"{star}"
        y_max = df_pivot[df_pivot['cell_type'] == cell_types[i]]['diff'].max()
        ax.text(i, y_max + 5, text, ha='center', va='bottom', fontsize=10, weight='bold', color='black')

    ax.set_ylabel("Age acceleration (years)")
    ax.set_xlabel("")
    ax.set_xticks(bar_x)
    ax.set_xticklabels(cell_types, rotation=45)
    ax.spines[['top', 'right']].set_visible(False)
    ax.margins(x=0.2, y=0.2)
    ax.legend(title='Actual age', bbox_to_anchor=(1.01, .9), loc='upper left', frameon=False)
    ax.axhline(0, linestyle='--', color='gray', linewidth=1)
    ax.set_title(f"{disease_name}", fontsize=12, weight='bold', pad=15)
    plt.tight_layout()
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