"""
Analyze vaccination response patterns comparing CMV negative vs positive subjects.

Goal: Identify CMV-specific patterns in TF activity changes during vaccination,
even if not statistically significant, focusing on the ~50 TFs that are significant
when combining all samples at Day 7.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy
import os

# Paths
from ciim.src.common import base_dir, PLOTS_DIR

# Load results
print("="*80)
print("VACCINATION RESPONSE: CMV-SPECIFIC PATTERNS ANALYSIS")
print("="*80)

results = pd.read_csv(f'{base_dir}/output/stats/stats_soundlife_bulk_tf_activity_mixed-effect.csv')

# Get significant TFs from Day 7 - All samples (maximum power)
d7_all = results[results['config_label'] == 'vacc_d7_all'].copy()
d7_sig = d7_all[d7_all['p_value_adj'] < 0.05].copy()

print(f"\nDay 7 (All samples) - Significant TFs: {len(d7_sig)}")
print(f"Cell types: {d7_sig['cell_type'].value_counts().to_dict()}")

# Get the list of significant TFs
sig_tfs = d7_sig['tf'].unique()
print(f"\nUnique TFs significant at Day 7: {len(sig_tfs)}")

# Now extract those TFs from CMV-stratified analyses
d7_cmv_neg = results[
    (results['config_label'] == 'vacc_d7_cmv_neg') &
    (results['tf'].isin(sig_tfs))
].copy()

d7_cmv_pos = results[
    (results['config_label'] == 'vacc_d7_cmv_pos') &
    (results['tf'].isin(sig_tfs))
].copy()

print(f"\nDay 7 (CMV Neg) - TFs extracted: {len(d7_cmv_neg)}")
print(f"Day 7 (CMV Pos) - TFs extracted: {len(d7_cmv_pos)}")

# Merge to compare slopes
comparison = d7_sig[['tf', 'cell_type', 'slope_condition', 'p_value_adj']].merge(
    d7_cmv_neg[['tf', 'cell_type', 'slope_condition', 'p_value']],
    on=['tf', 'cell_type'],
    how='inner',
    suffixes=('_all', '_cmv_neg')
).merge(
    d7_cmv_pos[['tf', 'cell_type', 'slope_condition', 'p_value']],
    on=['tf', 'cell_type'],
    how='inner',
    suffixes=('', '_cmv_pos')
)

comparison.rename(columns={
    'slope_condition': 'slope_cmv_pos',
    'p_value': 'p_value_cmv_pos'
}, inplace=True)

print(f"\nMerged comparison table: {len(comparison)} TF × cell type combinations")

# Add direction columns
comparison['direction_all'] = comparison['slope_condition_all'].apply(
    lambda x: 'Up' if x > 0 else 'Down'
)
comparison['direction_cmv_neg'] = comparison['slope_condition_cmv_neg'].apply(
    lambda x: 'Up' if x > 0 else 'Down'
)
comparison['direction_cmv_pos'] = comparison['slope_cmv_pos'].apply(
    lambda x: 'Up' if x > 0 else 'Down'
)

# Check agreement
comparison['cmv_neg_agrees'] = (
    comparison['direction_all'] == comparison['direction_cmv_neg']
)
comparison['cmv_pos_agrees'] = (
    comparison['direction_all'] == comparison['direction_cmv_pos']
)

# CMV-specific difference in effect size
comparison['cmv_diff'] = comparison['slope_cmv_pos'] - comparison['slope_condition_cmv_neg']
comparison['abs_cmv_diff'] = np.abs(comparison['cmv_diff'])

print("\n" + "="*80)
print("DIRECTIONAL AGREEMENT WITH COMBINED ANALYSIS")
print("="*80)
print(f"CMV Negative agrees: {comparison['cmv_neg_agrees'].sum()} / {len(comparison)} ({comparison['cmv_neg_agrees'].mean()*100:.1f}%)")
print(f"CMV Positive agrees: {comparison['cmv_pos_agrees'].sum()} / {len(comparison)} ({comparison['cmv_pos_agrees'].mean()*100:.1f}%)")

# Cases where CMV status changes the direction
comparison['direction_flips'] = (
    comparison['direction_cmv_neg'] != comparison['direction_cmv_pos']
)

print(f"\nDirection flips (CMV- vs CMV+): {comparison['direction_flips'].sum()}")

if comparison['direction_flips'].sum() > 0:
    print("\nTFs with direction flips:")
    flippers = comparison[comparison['direction_flips']][
        ['tf', 'cell_type', 'slope_condition_cmv_neg', 'slope_cmv_pos', 
         'direction_cmv_neg', 'direction_cmv_pos']
    ]
    print(flippers.to_string(index=False))

# Statistical comparison of CMV neg vs pos effect sizes
print("\n" + "="*80)
print("EFFECT SIZE COMPARISON")
print("="*80)

# Correlation between CMV- and CMV+ slopes
corr, pval = stats.pearsonr(
    comparison['slope_condition_cmv_neg'], 
    comparison['slope_cmv_pos']
)
print(f"\nCorrelation (CMV- vs CMV+ slopes): r = {corr:.3f}, p = {pval:.2e}")

# Mean absolute difference
print(f"Mean absolute difference |CMV+ - CMV-|: {comparison['abs_cmv_diff'].mean():.3f}")

# Top TFs with largest CMV-specific differences
print("\n" + "="*80)
print("TOP 10 TFs WITH LARGEST CMV-SPECIFIC DIFFERENCES (|CMV+ - CMV-|)")
print("="*80)
top_diff = comparison.nlargest(10, 'abs_cmv_diff')[
    ['tf', 'cell_type', 'slope_condition_cmv_neg', 'slope_cmv_pos', 
     'cmv_diff', 'direction_cmv_neg', 'direction_cmv_pos']
]
print(top_diff.to_string(index=False))

# ============================================================================
# VISUALIZATIONS
# ============================================================================

output_dir = os.path.join(PLOTS_DIR, 'vaccination_cmv_patterns')
os.makedirs(output_dir, exist_ok=True)

# 1. Scatter plot: CMV- vs CMV+ slopes
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: All TFs
ax = axes[0]
ax.scatter(
    comparison['slope_condition_cmv_neg'], 
    comparison['slope_cmv_pos'],
    alpha=0.6,
    s=50,
    c='steelblue'
)
ax.axline((0, 0), slope=1, color='red', linestyle='--', alpha=0.5, label='y=x')
ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
ax.axvline(0, color='gray', linestyle='-', linewidth=0.5)
ax.set_xlabel('Slope (CMV Negative)', fontsize=11)
ax.set_ylabel('Slope (CMV Positive)', fontsize=11)
ax.set_title(f'Vaccination Response: CMV- vs CMV+\n(r={corr:.3f}, p={pval:.2e})', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Right: Color by cell type
ax = axes[1]
for ct in comparison['cell_type'].unique():
    df_ct = comparison[comparison['cell_type'] == ct]
    ax.scatter(
        df_ct['slope_condition_cmv_neg'], 
        df_ct['slope_cmv_pos'],
        alpha=0.6,
        s=50,
        label=ct
    )
ax.axline((0, 0), slope=1, color='red', linestyle='--', alpha=0.5)
ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
ax.axvline(0, color='gray', linestyle='-', linewidth=0.5)
ax.set_xlabel('Slope (CMV Negative)', fontsize=11)
ax.set_ylabel('Slope (CMV Positive)', fontsize=11)
ax.set_title('Colored by Cell Type', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/cmv_neg_vs_pos_scatter.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: {output_dir}/cmv_neg_vs_pos_scatter.png")
plt.close()

# 2. Heatmap of slopes across CMV status
# Pivot for heatmap
for cell_type in comparison['cell_type'].unique():
    df_ct = comparison[comparison['cell_type'] == cell_type].copy()
    
    if len(df_ct) < 3:  # Skip if too few TFs
        continue
    
    # Create matrix: TFs x [All, CMV-, CMV+]
    heatmap_data = df_ct[['tf', 'slope_condition_all', 'slope_condition_cmv_neg', 'slope_cmv_pos']].copy()
    heatmap_data = heatmap_data.set_index('tf')
    heatmap_data.columns = ['All Samples', 'CMV Negative', 'CMV Positive']
    
    # Cluster rows
    if len(heatmap_data) > 2:
        linkage = hierarchy.linkage(heatmap_data.values, method='ward')
        dendro = hierarchy.dendrogram(linkage, no_plot=True)
        row_order = dendro['leaves']
        heatmap_data = heatmap_data.iloc[row_order]
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, len(heatmap_data)*0.3 + 2))
    
    sns.heatmap(
        heatmap_data,
        cmap='RdBu_r',
        center=0,
        vmin=-heatmap_data.abs().max().max(),
        vmax=heatmap_data.abs().max().max(),
        cbar_kws={'label': 'Vaccination Effect (Slope)'},
        linewidths=0.5,
        linecolor='white',
        ax=ax
    )
    
    ax.set_title(f'Vaccination Day 7: TF Response Patterns\n{cell_type}', 
                 fontsize=12, weight='bold', pad=15)
    ax.set_xlabel('Sample Stratification', fontsize=11)
    ax.set_ylabel('Transcription Factor', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/heatmap_slopes_{cell_type}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/heatmap_slopes_{cell_type}.png")
    plt.close()

# 3. Volcano-style plot: CMV difference vs significance
fig, ax = plt.subplots(figsize=(8, 6))

# Color by whether direction flips
colors = comparison['direction_flips'].map({True: 'red', False: 'steelblue'})

ax.scatter(
    comparison['cmv_diff'],
    -np.log10(comparison['p_value_cmv_neg']),
    alpha=0.6,
    s=50,
    c=colors,
    label='CMV Neg p-value'
)

ax.axvline(0, color='gray', linestyle='-', linewidth=0.5)
ax.axhline(-np.log10(0.05), color='gray', linestyle='--', alpha=0.5, label='p=0.05')

ax.set_xlabel('CMV-specific difference (CMV+ slope - CMV- slope)', fontsize=11)
ax.set_ylabel('-log10(p-value) in CMV Negative', fontsize=11)
ax.set_title('CMV-Specific Vaccination Response\n(Red = direction flip between CMV- and CMV+)', 
             fontsize=12, weight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/cmv_diff_vs_pvalue.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir}/cmv_diff_vs_pvalue.png")
plt.close()

# 4. Bar plot: Mean effect sizes
mean_effects = pd.DataFrame({
    'All Samples': comparison.groupby('cell_type')['slope_condition_all'].mean(),
    'CMV Negative': comparison.groupby('cell_type')['slope_condition_cmv_neg'].mean(),
    'CMV Positive': comparison.groupby('cell_type')['slope_cmv_pos'].mean()
})

fig, ax = plt.subplots(figsize=(8, 5))
mean_effects.plot(kind='bar', ax=ax, width=0.8, alpha=0.7)
ax.set_xlabel('Cell Type', fontsize=11)
ax.set_ylabel('Mean Vaccination Effect (Slope)', fontsize=11)
ax.set_title('Average Vaccination Response by CMV Status', fontsize=12, weight='bold')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.legend(title='Sample Group', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{output_dir}/mean_effects_by_cmv.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir}/mean_effects_by_cmv.png")
plt.close()

# ============================================================================
# ADDITIONAL ANALYSIS: Time course (D0, D7, D90)
# ============================================================================

print("\n" + "="*80)
print("TIME COURSE ANALYSIS: D0, D7, D90")
print("="*80)

# Extract same TFs across all timepoints
time_course_data = []

for day_config in ['vacc_d0_all', 'vacc_d7_all', 'vacc_d90_all',
                    'vacc_d0_cmv_neg', 'vacc_d7_cmv_neg', 'vacc_d90_cmv_neg',
                    'vacc_d0_cmv_pos', 'vacc_d7_cmv_pos', 'vacc_d90_cmv_pos']:
    
    day_data = results[
        (results['config_label'] == day_config) &
        (results['tf'].isin(sig_tfs))
    ].copy()
    
    if len(day_data) > 0:
        # Parse day and CMV status
        parts = day_config.split('_')
        day_num = parts[1].replace('d', '')
        cmv_status = '_'.join(parts[2:]) if len(parts) > 2 else 'all'
        
        day_data['day'] = int(day_num)
        day_data['cmv_status'] = cmv_status
        time_course_data.append(day_data)

if time_course_data:
    time_course_df = pd.concat(time_course_data, ignore_index=True)
    
    # Calculate mean slopes per day and CMV status
    time_summary = time_course_df.groupby(['day', 'cmv_status', 'cell_type']).agg({
        'slope_condition': ['mean', 'std', 'count']
    }).reset_index()
    
    time_summary.columns = ['day', 'cmv_status', 'cell_type', 'mean_slope', 'std_slope', 'n_tfs']
    
    print("\nTime course summary (mean slopes):")
    print(time_summary.to_string(index=False))
    
    # Plot time course
    fig, axes = plt.subplots(1, len(comparison['cell_type'].unique()), 
                             figsize=(5*len(comparison['cell_type'].unique()), 4))
    
    if len(comparison['cell_type'].unique()) == 1:
        axes = [axes]
    
    for idx, cell_type in enumerate(sorted(comparison['cell_type'].unique())):
        ax = axes[idx]
        
        for cmv_status in ['all', 'cmv_neg', 'cmv_pos']:
            df_plot = time_summary[
                (time_summary['cell_type'] == cell_type) &
                (time_summary['cmv_status'] == cmv_status)
            ].sort_values('day')
            
            if len(df_plot) > 0:
                label = cmv_status.replace('_', ' ').title()
                ax.plot(df_plot['day'], df_plot['mean_slope'], 
                       marker='o', linewidth=2, label=label)
                ax.fill_between(
                    df_plot['day'],
                    df_plot['mean_slope'] - df_plot['std_slope'],
                    df_plot['mean_slope'] + df_plot['std_slope'],
                    alpha=0.2
                )
        
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Day', fontsize=11)
        ax.set_ylabel('Mean Vaccination Effect', fontsize=11)
        ax.set_title(f'{cell_type}', fontsize=12, weight='bold')
        ax.set_xticks([0, 7, 90])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Vaccination Response Time Course by CMV Status', 
                 fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/time_course_by_cmv.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/time_course_by_cmv.png")
    plt.close()

# ============================================================================
# SAVE COMPARISON TABLE
# ============================================================================

comparison_output = comparison[[
    'tf', 'cell_type', 
    'slope_condition_all', 'p_value_adj_all',
    'slope_condition_cmv_neg', 'p_value_cmv_neg',
    'slope_cmv_pos', 'p_value_cmv_pos',
    'cmv_diff', 'direction_all', 'direction_cmv_neg', 'direction_cmv_pos',
    'direction_flips'
]].copy()

comparison_output.to_csv(f'{output_dir}/vaccination_cmv_comparison.csv', index=False)
print(f"\nSaved comparison table: {output_dir}/vaccination_cmv_comparison.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"All results saved to: {output_dir}")
print("\nKey Findings:")
print(f"1. {len(sig_tfs)} TFs significant at Day 7 (combined)")
print(f"2. Direction agreement - CMV Neg: {comparison['cmv_neg_agrees'].mean()*100:.1f}%, CMV Pos: {comparison['cmv_pos_agrees'].mean()*100:.1f}%")
print(f"3. {comparison['direction_flips'].sum()} TFs show direction flip between CMV- and CMV+")
print(f"4. Correlation between CMV- and CMV+ effects: r={corr:.3f}")
print("="*80)
