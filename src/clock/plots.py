
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import numpy as np
from pandas.api.types import CategoricalDtype
from statsmodels.stats.multitest import multipletests
from hiara import surrogate_names, MAJOR_CTS, palette_genders

palette_disease = {'Healthy': '#56B4E9', 'SLE': '#F0E442', 'Mild': '#2ca02c', 'Severe': '#e377c2'}

def plot_scatter_age_vs_predictedAge(df, dataset='', ax=None, hue='sex', palette={},  s=50, alpha=0.5):
    # from hiara.src.clock.helper import evaluate_groupwise_median
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    sns.scatterplot(data=df, x='age', y='predicted_age', s=s, alpha=alpha, ax=ax, palette=palette, hue=hue)
    
    # Add ideal fit line (y=x)
    min_age = df['age'].min()
    max_age = df['age'].max()
    ax.plot([min_age, max_age], [min_age, max_age], 'k--', linewidth=1.5, alpha=0.5, label='Ideal fit')
    
    # Move legend to the right side
    if ax.get_legend():
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

    ax.set_xlabel("Actual Age")
    ax.set_ylabel("Predicted Age")
    ax.margins(x=0.1, y=0.1)
    ax.spines[['top', 'right']].set_visible(False)
    dataset = surrogate_names.get(dataset, dataset)
    if False:
        rr = evaluate_groupwise_median(df)
        spearman = rr['Spearman']
        R2 = rr['R2']
        ax.set_title(f"{dataset} \n (S: {spearman:.2f}, R2:{R2:.2f})", pad=15)
    else:
        ax.set_title(f"{dataset}", pad=15)

# def wrapper_age_acceleration_disease(obs, disease_dataset, figsize=(4, 2.5)):
#     import pandas as pd
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     from scipy.stats import ttest_ind
#     from statsmodels.stats.multitest import multipletests
#     import numpy as np

#     if disease_dataset == 'Covid_50MHH':
#         disease_name = 'COVID-19'
#         ctr = 'mild'
#         cond = 'severe'
#     elif disease_dataset == 'SLE_European':
#         disease_name = 'SLE'
#         ctr = 'normal'
#         cond = 'systemic lupus erythematosus'

#     obs_disease = obs[obs['dataset'] == disease_dataset].copy()
#     obs_disease['age'] = obs_disease['age'].astype(float)
#     # obs_disease['donor_id'] = obs_disease['donor_id'].astype(str)
#     obs_disease['cell_type'] = obs_disease['cell_type'].astype(str)
#     obs_disease['condition'] = obs_disease['condition'].astype(str)

#     # Only keep relevant conditions
#     obs_disease = obs_disease[obs_disease['condition'].isin([ctr, cond])]
#     obs_disease['cell_type'] = pd.Categorical(obs_disease['cell_type'], categories=cell_types, ordered=True)

#     fig, ax = plt.subplots(figsize=figsize)

#     # Prepare a grouping variable for (cell_type, condition)
#     obs_disease['group'] = obs_disease['cell_type'].astype(str) + ' | ' + obs_disease['condition']

#     obs_disease_c = obs_disease.copy()
#     obs_disease_c['condition'] = obs_disease_c['condition'].apply(lambda x: surrogate_names.get(x, x))
#     if disease_dataset == 'Covid_50MHH':
#         obs_disease_c['condition'] = obs_disease_c['condition'].astype(CategoricalDtype(categories=['Mild', 'Severe'], ordered=True))
#     elif disease_dataset == 'SLE_European':
#         obs_disease_c['condition'] = obs_disease_c['condition'].astype(CategoricalDtype(categories=['Healthy', 'SLE'], ordered=True))
#     # Plot stripplot
#     sns.stripplot(
#         data=obs_disease_c,
#         x='cell_type', y='predicted_age', hue='condition',
#         dodge=True, jitter=True, alpha=0.7, ax=ax, palette='Set2', size=5
#     )

#     # Statistical test and brackets
#     pvals = []
#     brackets = []
#     for i, cell_type in enumerate(cell_types):
#         data_ctr = obs_disease[(obs_disease['cell_type'] == cell_type) & (obs_disease['condition'] == ctr)]['predicted_age']
#         data_cond = obs_disease[(obs_disease['cell_type'] == cell_type) & (obs_disease['condition'] == cond)]['predicted_age']

#         if len(data_ctr) > 1 and len(data_cond) > 1:
#             _, pval = ttest_ind(data_ctr, data_cond, equal_var=False)
#         else:
#             pval = 1.0

#         pvals.append(pval)

#         # Save bracket info: x positions, height, and p-value
#         if len(data_ctr) > 0 and len(data_cond) > 0:
#             x1, x2 = i - 0.2, i + 0.2
#             y_max = max(data_ctr.max(), data_cond.max())
#             brackets.append((x1, x2, y_max + 3, pval))

#     # FDR correction
#     _, pvals_fdr, _, _ = multipletests(pvals, method='fdr_bh')

#     # Add brackets and stars
#     for i, (x1, x2, y, pval_fdr) in enumerate(zip([b[0] for b in brackets], [b[1] for b in brackets], [b[2] for b in brackets], pvals_fdr)):
#         star = '***' if pval_fdr < 0.001 else '**' if pval_fdr < 0.01 else '*' if pval_fdr < 0.05 else ''
#         bracket_height = 2.5  # You can adjust this value as needed
#         y_bracket = y + 4
#         if star:
#             ax.plot([x1, x1, x2, x2],
#                     [y_bracket, y_bracket + bracket_height, y_bracket + bracket_height, y_bracket],
#                     lw=1.5, color='black')
#             ax.text((x1 + x2) / 2,
#                     y_bracket + bracket_height + 1,
#                     star,
#                     ha='center', va='bottom', fontsize=10)

#     ax.set_ylabel("Predicted age")
#     ax.set_xlabel("")
#     ax.set_xticklabels(cell_types, rotation=45)
#     ax.spines[['top', 'right']].set_visible(False)
#     ax.margins(x=0.1, y=0.1)
#     ax.legend(title='Condition', bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
#     ax.set_title(f"{disease_name}", fontsize=12, weight='bold', pad=15)
#     plt.tight_layout()

def wrapper_age_acceleration_disease(obs, disease_dataset, ctr, cond, figsize=(4, 2.7)):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_1samp
    from statsmodels.stats.multitest import multipletests
    import numpy as np

    cell_types = MAJOR_CTS
    
    obs_disease = obs[obs['dataset'] == disease_dataset].copy()

    obs_disease['age'] = obs_disease['age'].astype(float)
    # obs_disease['donor_id'] = obs_disease['donor_id'].astype(str)
    obs_disease['cell_type'] = obs_disease['cell_type'].astype(str)
    obs_disease['condition'] = obs_disease['condition'].astype(str)

    df_pivot = obs_disease.pivot_table(index=['age', 'cell_type'], columns='condition', values='predicted_age')
    df_pivot = df_pivot.reset_index()
    
    # Convert control and treatment values to strings for comparison
    ctr = str(ctr)
    cond = str(cond)
    
    # Check if expected condition values exist in pivot columns
    # If not, try case-insensitive match
    available_cols = [c for c in df_pivot.columns if c not in ['age', 'cell_type']]
    if ctr not in df_pivot.columns:
        # Try case-insensitive match
        ctr_match = [c for c in available_cols if str(c).lower() == ctr.lower()]
        if ctr_match:
            ctr = ctr_match[0]
    if cond not in df_pivot.columns:
        # Try case-insensitive match
        cond_match = [c for c in available_cols if str(c).lower() == cond.lower()]
        if cond_match:
            cond = cond_match[0]
    
    df_pivot['age_shift'] = df_pivot[cond] - df_pivot[ctr]
    df_pivot = df_pivot[~df_pivot['age_shift'].isna()]
    df_pivot['cell_type'] = pd.Categorical(df_pivot['cell_type'], categories=cell_types, ordered=True)

    fig, ax = plt.subplots(figsize=figsize)

    # Stripplot
    sns.stripplot(
        ax=ax, data=df_pivot, y='age_shift', x='cell_type',
        alpha=0.7, hue='age', palette='viridis', s=6
    )

    # Bar plot for medians (frame-only bars)
    medians = df_pivot.groupby('cell_type')['age_shift'].median().reindex(cell_types)
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
        y_max = df_pivot[df_pivot['cell_type'] == cell_types[i]]['age_shift'].max()
        ax.text(i, y_max + 5, text, ha='center', va='bottom', fontsize=10, weight='bold', color='black')

    ax.set_ylabel("Age acceleration (years)")
    ax.set_xlabel("")
    ax.set_xticks(bar_x)
    ax.set_xticklabels(cell_types, rotation=45)
    ax.spines[['top', 'right']].set_visible(False)
    ax.margins(x=0.2, y=0.2)
    ax.legend(title='Actual age', bbox_to_anchor=(1.01, .9), loc='upper left', frameon=False)
    ax.axhline(0, linestyle='--', color='gray', linewidth=1)
    # ax.set_title(f"{disease_name}", fontsize=12, weight='bold', pad=15)
    plt.tight_layout()


def plot_age_acceleration_by_group(obs, dataset, ctr, cond):
    """
    Plot age acceleration for a specific age group (used when data is already filtered by age_group).
    Shows a simple bar plot comparing conditions within a single age group.
    
    Parameters
    ----------
    obs : pd.DataFrame
        Observations filtered to specific age group
    dataset : str
        Dataset name
    config : ConditionConfig
        Configuration with condition mapping and display info
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_ind
    import numpy as np

    
    obs_disease = obs[obs['dataset'] == dataset].copy()
    obs_disease['age'] = obs_disease['age'].astype(float)
    obs_disease['cell_type'] = obs_disease['cell_type'].astype(str)
    obs_disease['condition'] = obs_disease['condition'].astype(str)
    
    # Calculate age acceleration for each donor
    df_pivot = obs_disease.pivot_table(index=['donor_id', 'age', 'cell_type'], columns='condition', values='predicted_age')
    df_pivot = df_pivot.reset_index()
    
    # Check if both conditions exist
    if ctr not in df_pivot.columns or cond not in df_pivot.columns:
        print(f"Warning: Missing condition columns. Available: {df_pivot.columns.tolist()}")
        return
    
    df_pivot['age_shift'] = df_pivot[cond] - df_pivot[ctr]
    df_pivot = df_pivot[~df_pivot['age_shift'].isna()]
    
    cell_types_present = df_pivot['cell_type'].unique()
    n_cell_types = len(cell_types_present)
    
    fig, ax = plt.subplots(figsize=(1.5 + 0.8 * n_cell_types, 2.5))
    
    # Bar plot showing age acceleration by cell type
    sns.barplot(data=df_pivot, x='cell_type', y='age_shift', ax=ax, 
                color='#56B4E9', alpha=0.7, errorbar='sd', capsize=0.1)
    
    # Add individual points
    sns.stripplot(data=df_pivot, x='cell_type', y='age_shift', ax=ax,
                  color='black', alpha=0.3, size=3)
    
    # Statistical tests
    from statsmodels.stats.multitest import multipletests
    pvals = []
    for cell_type in cell_types_present:
        group_data = df_pivot[df_pivot['cell_type'] == cell_type]['age_shift']
        if len(group_data) > 2:
            from scipy.stats import ttest_1samp
            _, pval = ttest_1samp(group_data, 0)  # Test against 0 (no acceleration)
        else:
            pval = 1.0
        pvals.append(pval)
    
    # Correct for multiple testing
    if len(pvals) > 0:
        _, pvals_fdr, _, _ = multipletests(pvals, method='fdr_bh')
        for i, (cell_type, pval_fdr) in enumerate(zip(cell_types_present, pvals_fdr)):
            star = '***' if pval_fdr < 0.001 else '**' if pval_fdr < 0.01 else '*' if pval_fdr < 0.05 else ''
            if star:
                y_max = df_pivot[df_pivot['cell_type'] == cell_type]['age_shift'].max()
                ax.text(i, y_max + 1, star, ha='center', va='bottom', fontsize=12, weight='bold')
    
    ax.axhline(0, linestyle='--', color='gray', linewidth=1, alpha=0.5)
    ax.set_ylabel("Age acceleration (years)")
    ax.set_xlabel("")
    ax.spines[['top', 'right']].set_visible(False)
    ax.margins(x=0.2, y=0.2)
    
    # Get age group label from config
    # age_group_label = [config.data_filter.get('age_group', 'Unknown')]
    # if isinstance(age_group_label, list):
    #     age_group_label = age_group_label[0]
    
    # title = f"{config.display_name}" if config.display_name else f"{dataset} - {age_group_label}"
    # ax.set_title(title, fontsize=11, weight='bold', pad=10)
    plt.tight_layout()


def wrapper_plot_age_acceleration_disease_bins(obs, disease_dataset, ctr, cond):
    AGE_CUTOFF_SLE = 50
    

    obs_disease = obs[obs['dataset'] == disease_dataset].copy()
    obs_disease['age'] = obs_disease['age'].astype(float)
    obs_disease['donor_id'] = obs_disease['donor_id'].astype(str)
    
    # Calculate signed residuals (age acceleration)
    obs_disease['age_residual'] = abs(obs_disease['predicted_age'] - obs_disease['age'])
    
    # Convert control and treatment values to strings for comparison
    ctr = str(ctr)
    cond = str(cond)
    
    # Check if expected condition values exist in data, use case-insensitive match if needed
    available_conditions = obs_disease['condition'].unique()
    if ctr not in available_conditions:
        ctr_match = [c for c in available_conditions if str(c).lower() == ctr.lower()]
        if ctr_match:
            ctr = ctr_match[0]
    if cond not in available_conditions:
        cond_match = [c for c in available_conditions if str(c).lower() == cond.lower()]
        if cond_match:
            cond = cond_match[0]

    cell_types = obs_disease['cell_type'].unique()
    n_cell_types = len(cell_types)

    fig, axes = plt.subplots(1, n_cell_types, figsize=(1.5 * n_cell_types+2, 2.5), sharey=True)

    for i, cell_type in enumerate(cell_types):
        ax = axes[i] if len(cell_types) > 1 else axes

        obs_ct = obs_disease[obs_disease['cell_type'] == cell_type].copy()
        
        # Check if age_group column exists (for soundlife CMV analysis)
        if 'age_group' in obs_ct.columns:
            # Convert age_group from Categorical to string to avoid creating empty combinations
            obs_ct['age_group'] = obs_ct['age_group'].astype(str)
            # Group by donor_id, age_group, and condition - take median of age_residual
            obs_ct = obs_ct.groupby(['donor_id', 'age_group', 'condition'])['age_residual'].median().reset_index()
            # Use existing age_group column and map to pretty names
            age_group_name_mapping = {'young': 'Young', 'old': 'Old'}
            obs_ct['age_bin'] = obs_ct['age_group'].map(age_group_name_mapping)
            # Set correct order: Young, Old
            age_bin_order = ['Young', 'Old']
        else:
            # Standard groupby without age_group
            obs_ct = obs_ct.groupby(['donor_id', 'condition']).agg({
                'age_residual': 'median',
                'age': 'median'
            }).reset_index()
            # Bin ages manually (for SLE, COVID, etc.)
            # Import age cutoff from common configuration
            age_bins = [20, AGE_CUTOFF_SLE, 80]
            obs_ct['age_bin'] = pd.cut(obs_ct['age'], bins=age_bins, right=False)
            obs_ct['age_bin'] = obs_ct['age_bin'].astype(str)
            age_bin_order = [str(b) for b in pd.cut([21, AGE_CUTOFF_SLE + 1], bins=age_bins, right=False).categories]

        plot_data = []
        pvals = []
        positions = []

        print(f"\n{cell_type} - Age residuals by age group:")
        print(f"{'Age Group':<12} {'Mean Difference':<15} {'P-value':<10} {'N_Control':<10} {'N_Disease':<10}")
        print("-" * 70)

        for age_bin in age_bin_order:
            for group in [ctr, cond]:
                vals = obs_ct[(obs_ct['condition'] == group) & (obs_ct['age_bin'] == age_bin)]['age_residual']
                for v in vals:
                    plot_data.append({'age_bin': age_bin, 'condition': group, 'age_residual': v})

            # Store p-values for later FDR correction and calculate mean differences
            cond_vals = obs_ct[(obs_ct['condition'] == cond) & (obs_ct['age_bin'] == age_bin)]['age_residual'].values
            ctr_vals = obs_ct[(obs_ct['condition'] == ctr) & (obs_ct['age_bin'] == age_bin)]['age_residual'].values
            
            if len(cond_vals) >= 3 and len(ctr_vals) >= 3:
                tstat, pval = ttest_ind(cond_vals, ctr_vals, equal_var=False)
                mean_diff = cond_vals.mean() - ctr_vals.mean()
                pvals.append(pval)
                positions.append(age_bin)
                print(f"{age_bin:<12} {mean_diff:>+8.2f} yrs    {pval:<10.3e} {len(ctr_vals):<10} {len(cond_vals):<10}")
            else:
                pvals.append(np.nan)
                positions.append(age_bin)
                if len(cond_vals) > 0 and len(ctr_vals) > 0:
                    mean_diff = cond_vals.mean() - ctr_vals.mean()
                    print(f"{age_bin:<12} {mean_diff:>+8.2f} yrs    {'N/A':<10} {len(ctr_vals):<10} {len(cond_vals):<10} (insufficient data)")
                else:
                    print(f"{age_bin:<12} {'N/A':<15} {'N/A':<10} {len(ctr_vals):<10} {len(cond_vals):<10} (no data)")

        plot_df = pd.DataFrame(plot_data)
        plot_df_c = plot_df.copy()
        plot_df_c['condition'] = plot_df_c['condition'].apply(lambda name: surrogate_names.get(name, name))
        
        # Use None for palette if config is provided (let seaborn auto-generate)
        palette_to_use = palette_disease
        
        sns.barplot(data=plot_df_c, x='age_bin', y='age_residual', hue='condition', width=0.5,
                    ax=ax, palette=palette_to_use, ci='sd', errorbar='sd', capsize=0.1, linewidth=.1, alpha=.8, errcolor='black', errwidth=1.5)

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
                    group_means = group_data.groupby('condition')['age_residual'].mean()
                    group_stds = group_data.groupby('condition')['age_residual'].std()
                    y_max = (group_means + group_stds).max() + 2  # leave more space

                    h = 1  # bracket height
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
            ax.set_ylabel("Age residual (years)\n(predicted - actual age)")
        else:
            ax.set_ylabel("")
        ax.get_legend().remove()
        ax.spines[['top', 'right']].set_visible(False)
        ax.axhline(0, linestyle='--', color='gray', linewidth=1, alpha=0.5)
        ax.margins(x=0.2, y=0.1)

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Condition', frameon=False)
    # plt.suptitle(f"Age shift in {disease_name}", fontsize=13, y=1.05, weight='bold')
    plt.tight_layout()
    # plt.show()
def plot_experiment(test_type, df_all, ctr, treatment, cell_type, pval_map, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(2, 2))
    assert test_type in ['paired', 'unpaired', 'mixed-effect']

    # subset data
    df_sub = df_all[df_all['condition'].isin([ctr, treatment])].copy()

    # map donor IDs to pretty names
    donor_ids = df_sub['donor_id'].unique()
    donor_pretty = {did: f"Donor {i+1}" for i, did in enumerate(donor_ids)}
    df_sub['test_group_pretty'] = df_sub['donor_id'].map(donor_pretty)

    # create donor palette
    donors = df_sub['test_group_pretty'].unique()
    donor_palette = dict(zip(
        donors,
        sns.color_palette("husl", len(donors))
    ))

    if test_type == "paired":
        # pivot to ensure donors have both conditions
        df_pivot = df_sub.pivot_table(index='test_group_pretty', columns='condition', values='predicted_age')
        df_pivot = df_pivot.dropna(subset=[ctr, treatment], how='any')

        df_plot = df_pivot.reset_index().melt(
            id_vars='test_group_pretty',
            value_vars=[ctr, treatment],
            var_name='condition',
            value_name='predicted_age'
        )

        sns.lineplot(
            data=df_plot,
            x='condition', y='predicted_age',
            hue='test_group_pretty', marker='o', alpha=0.6,
            ax=ax, palette=donor_palette, legend=True
        )
    else:
        # donor-colored scatter dots
        sns.stripplot(
            data=df_sub,
            x='condition', y='predicted_age',
            order=[ctr, treatment],
            dodge=False, jitter=True,
            hue='test_group_pretty',
            palette=donor_palette,
            ax=ax, alpha=0.7
        )
        ax.legend_.set_title("Donor")

    # add stats annotation
    pval, slope = pval_map.get((cell_type, ctr, treatment), (1.0, 0))
    star = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''

    y_max = df_sub['predicted_age'].max()
    y_bracket = y_max + 7
    h = 2

    ax.plot([0, 0, 1, 1], [y_bracket - h, y_bracket, y_bracket, y_bracket - h],
            lw=1.5, c='black')
    ax.text(0.5, y_bracket + 1, f"{slope:.2f} yrs {star}\n(p={pval:.3g})",
            ha='center', va='bottom', fontsize=10)

    ax.set_xlabel("")
    ax.set_ylabel("Predicted Age")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.margins(x=0.1, y=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    return ax


def plot_group_strip(df_all, group_exps, group_name, cell_type, pval_map, ctr="Control", 
            figsize=None,  name_mapping={}, max_len=25, highlight_treatments=None):
    """
    Strip plot showing treatment-control differences for all significant treatments.
    Each dot is one donor/sample, colored by donor ID.
    """
    if not group_exps:
        return None

    df_plot = []
    # donors_all = sorted(df_all['donor_id'].unique())
    # donor_map = {d: f"Donor {i+1}" for i, d in enumerate(donors_all)}  # Pretty names
    donor_map = {}
    # Build mapping: treatment -> control
    treatment_to_ctr = {}
    for ctr, treatment in group_exps:
        df_sub = df_all[df_all['condition'].isin([ctr, treatment])].copy()
        df_pivot = df_sub.pivot_table(index='donor_id', columns='condition', values='predicted_age')
        if df_pivot.empty:
            continue

        diff = df_pivot[treatment] - df_pivot[ctr]
        if group_name == "Rejuvenating":
            diff = -diff

        for donor, val in diff.items():
            df_plot.append({
                'treatment': treatment,
                'diff': val,
                'p_value': pval_map.get((cell_type, ctr, treatment), (1.0, 0))[0],
                'donor': donor_map.get(donor, donor)
            })
        treatment_to_ctr[treatment] = ctr
    if not df_plot:
        return None

    df_plot = pd.DataFrame(df_plot)
    order = df_plot.groupby('treatment')['p_value'].mean().sort_values(
        ascending=True
    ).index

    # Plot strip
    extra_space = 1 if len(order) > 4 else 3
    if figsize is None:
        width = .25*len(order)+extra_space+1
        if len(order) < 5:
            width = 3
        if len(order) < 2:
            width = 2.5
        figsize = (width, 3)
    fig, ax = plt.subplots(figsize=figsize)
    # donors = sorted(df_plot['donor'].unique(), key=lambda x: int(x.split(' ')[1]))
    donors = sorted(df_plot['donor'].unique())
    donor_palette = dict(zip(donors, sns.color_palette("husl", len(donors))))

    sns.stripplot(
        data=df_plot,
        x='treatment',
        y='diff',
        order=order,
        size=6,
        hue='donor',
        palette=donor_palette,
        jitter=False,
        ax=ax,
        hue_order=donors 
    )
    
    # Annotate above the dots
    for i, treatment in enumerate(order):
        mean_diff = df_plot[df_plot['treatment']==treatment]['diff'].mean()
        max_diff = df_plot[df_plot['treatment']==treatment]['diff'].max()
        ctr = treatment_to_ctr[treatment]  # get correct control
        pval, slope = pval_map.get((cell_type, ctr, treatment))
        text = f"{pval:.2}"
        y_loc = max_diff + .1*max_diff
        ax.text(i, y_loc, text, ha='center', va='bottom', fontsize=8, rotation=45)

    # Set x-tick labels with color highlighting
    xticklabels = [name_mapping.get(t, t)[:max_len] for t in order]
    ax.set_xticklabels(xticklabels, rotation=45, ha='right')
    
    # Color specific treatments in red if highlight_treatments is provided
    if highlight_treatments:
        for i, (tick_label, treatment) in enumerate(zip(ax.get_xticklabels(), order)):
            if treatment in highlight_treatments:
                tick_label.set_color('red')
                # tick_label.set_weight('bold')
    
    x_margin = .3 - .02 * len(order)
    if x_margin < 0.05:
        x_margin = 0.05
    # print(f"x_margin: {x_margin}")
    ax.margins(x=x_margin, y=.1)
    # Update y-axis label
    ylabel = "Age rejuvenation (yrs)" if group_name=="Rejuvenating" else "Age acceleration (yrs)"
    ax.set_ylabel(ylabel)
    ax.set_xlabel('')
    ax.set_title(f"{cell_type}", fontsize=12, weight='bold', pad=40)
    ax.spines[['top', 'right']].set_visible(False)
    bbox_to_anchor = (1.01, 1) if len(donors) <= 10 else (1.01, 1.2)
    ax.legend(title="", bbox_to_anchor=bbox_to_anchor, loc='upper left', frameon=False, labelspacing=0.2,)
    plt.tight_layout()
    return fig, ax