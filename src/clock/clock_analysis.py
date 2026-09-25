#!/usr/bin/env python
"""
Aging Clock Analysis - Post-prediction analysis script

This script performs post-prediction analysis for aging clock predictions on both 
disease and perturbation datasets, generating various visualizations including:
- Age acceleration in disease datasets
- Age-stratified analysis
- Perturbation effect analysis (rejuvenating vs accelerating)
- Statistical testing with mixed effects models
"""

import argparse
from calendar import c
import os
import sys
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import anndata as ad
from statsmodels.stats.multitest import multipletests

# Import common utilities and configuration
from hira.src.config import (
    CLOCK_PLOTS_DIR as PLOTS_DIR,
    OUTPUT_DIR,
    CLOCK_STATS_DIR,
    MAJOR_CTS as default_cell_types,
    surrogate_names,
    colors_blind
)

from hira.src.config import get_config
from hira.src.clock.helper import save_clock_stats
from hira.src.utils.util import test_mixed_effects, test_paired, test_unpaired
from hira.src.clock.plots import (
    wrapper_age_acceleration_disease,
    wrapper_plot_age_acceleration_disease_bins,
    plot_group_strip,
    plot_experiment,
    plot_scatter_age_vs_predictedAge
)

from hira.src.clock.helper import wrapper_clock_predictions

warnings.filterwarnings("ignore")

# Set matplotlib defaults
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "Arial"
# increase the pd display width
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 100)

def apply_data_filter(obs, config):
    """
    Apply data filters from config to observation dataframe.
    
    Parameters
    ----------
    obs : pd.DataFrame
        Observations dataframe
    config : ConditionConfig
        Configuration with data_filter specifications
        
    Returns
    -------
    pd.DataFrame
        Filtered observations
    """
    if config.data_filter is None:
        return obs
    
    filtered_obs = obs.copy()
    for column, values in config.data_filter.items():
        if column not in filtered_obs.columns:
            print(f"Warning: Filter column '{column}' not found in data. Skipping filter.")
            continue
        
        # Handle both single value and list of values
        if not isinstance(values, list):
            values = [values]
        
        filtered_obs = filtered_obs[filtered_obs[column].isin(values)]
        print(f"  Filtered by {column}: {values} -> {len(filtered_obs)} samples remaining")
    
    return filtered_obs


def get_experiment_setup(obs_pert, config):
    """
    Get experiment setup (control-treatment pairs) for a dataset from config.
    
    Parameters
    ----------
    dataset : str
        Dataset name
    obs_pert : pd.DataFrame
        Observations dataframe
    config : ConditionConfig
        Dataset configuration
        
    Returns
    -------
    tuple
        (experiments, pvalue_show_type, pretty_names, test_type, group_key, mock_names, plot_config)
    """
    # Get settings from config
    pvalue_show_type = config.clock_pvalue_correction if config.clock_pvalue_correction else 'corrected'
    pretty_names = config.clock_pretty_names if config.clock_pretty_names else {}
    test_type = config.clock_test_type  
    group_key = config.clock_group_key 
    mock_names = config.clock_mock_names if hasattr(config, 'clock_mock_names') else False
    plot_config = config.clock_plot_config if config.clock_plot_config else {}
    
    clock_experiments = config.clock_experiments 
    assert clock_experiments is not None, "clock_experiments must be specified in config"
    # Get experiments
    if config.clock_experiments == 'all':
        conditions = obs_pert['condition'].unique().tolist()
        control_mapping = config.control_mapping
        experiments = []
        for cond in conditions:
            if isinstance(control_mapping, str):
                ctr = control_mapping
            else:
                ctr = control_mapping[cond]
            if cond != ctr:
                experiments.append((ctr, cond))
    else:
        experiments = config.clock_experiments
    
    return experiments, pvalue_show_type, pretty_names, test_type, group_key, mock_names, plot_config


def perform_statistical_tests(obs_pert, experiments, dataset, test_type='mixed-effect', 
                              pvalue_show_type='corrected', group_key='donor_id'):
    """
    Perform statistical tests for all experiments.
    
    Parameters
    ----------
    obs_pert : pd.DataFrame
        Observations dataframe
    experiments : list
        List of (control, treatment) tuples
    dataset : str
        Dataset name
    test_type : str
        Type of test (paired, unpaired, mixed_effect)
    pvalue_show_type : str
        Whether to show raw or corrected p-values
    group_key : str
        Column name for random effects grouping
        
    Returns
    -------
    dict
        Dictionary mapping (cell_type, ctr, treatment) to (p_value, slope)
    """
    config = get_config(dataset)
    pval_map = {}
    
    for cell_type in obs_pert['cell_type'].unique():
        obs_t = obs_pert[obs_pert['cell_type'] == cell_type].copy()
        df_all = obs_t.copy()
        
        raw_pvals, slopes, test_labels = [], [], []
        
        for (ctr, treatment) in experiments:
            # Check sufficient data
            if (df_all['condition'] == treatment).sum() < 3:
                print(f"Skipping {cell_type} {ctr} vs {treatment} due to insufficient data.")
                continue
            
            df_sub = df_all[df_all['condition'].isin([ctr, treatment])].copy()
            if len(df_sub[df_sub['condition'] == treatment]) < 3:
                print(f"Skipping {cell_type} {ctr} vs {treatment} due to insufficient data.")
                continue

            # Perform statistical test
            if test_type == 'paired':
                p_value, slope = test_paired(df_sub, ctr, treatment)
            elif test_type == 'unpaired':
                p_value, slope = test_unpaired(df_sub, ctr, treatment)
            elif test_type == 'mixed-effect':
                p_value, slope = test_mixed_effects(
                    df_sub, ctr, treatment, 
                    target_variable='predicted_age', 
                    group_key=group_key,
                    config=config
                )
            else:
                raise ValueError(f"Unknown test_type: {test_type}")

            if np.isnan(p_value):
                print(f"Warning: NaN p-value for {cell_type} {ctr} vs {treatment}. Skipping.")
                continue
            
            raw_pvals.append(p_value)
            slopes.append(slope)
            test_labels.append((cell_type, ctr, treatment))
        
        # Apply FDR correction only if we have tests for this cell type
        if raw_pvals:
            if pvalue_show_type == 'raw':
                corrected_pvals = raw_pvals
            else:
                rejected, corrected_pvals, _, _ = multipletests(raw_pvals, method='fdr_bh')
            
            
            for (label, corr_pval, slope) in zip(test_labels, corrected_pvals, slopes):
                pval_map[label] = (corr_pval, slope)

    save_clock_stats(pd.DataFrame(
        [{'dataset': dataset, 'cell_type': ct, 'ctr': ctr, 'treatment': tr,
          'p_value': p, 'delta': d, 'test_type': test_type, 'p_value_type': pvalue_show_type}
         for (ct, ctr, tr), (p, d) in pval_map.items()]), f'perturbation_{dataset}')
    return pval_map


def prepare_plot_inputs(obs_pert, cell_type, experiments, pval_map, p_value_t=0.05):
    """
    Prepare plotting inputs by categorizing experiments.
    
    Parameters
    ----------
    obs_pert : pd.DataFrame
        Observations dataframe
    cell_type : str
        Cell type
    experiments : list
        List of (control, treatment) tuples
    pval_map : dict
        P-value mapping
    p_value_t : float
        P-value threshold
        
    Returns
    -------
    tuple
        (df_all, rejuvenating, aging)
    """
    df_all = obs_pert[obs_pert['cell_type'] == cell_type][
        ['donor_id', 'condition', 'predicted_age']
    ].copy()
    
    significant_experiments = [
        (ctr, treatment) for (ctr, treatment) in experiments
        if pval_map.get((cell_type, ctr, treatment), (1.0, 0))[0] < p_value_t
    ]
    
    rejuvenating = [
        (ctr, treatment) for (ctr, treatment) in significant_experiments 
        if pval_map.get((cell_type, ctr, treatment), (1.0, 0))[1] < 0
    ]
    
    aging = [
        (ctr, treatment) for (ctr, treatment) in significant_experiments 
        if pval_map.get((cell_type, ctr, treatment), (1.0, 0))[1] > 0
    ]
    
    return df_all, rejuvenating, aging


def analyze_disease(obs, dataset, output_dir, cell_types, config=None):
    """
    Analyze disease dataset for age acceleration.
    
    Parameters
    ----------
    obs : pd.DataFrame
        Observations with predictions
    dataset : str
        Dataset name
    output_dir : str
        Output directory for plots
    cell_types : list
        Cell types to analyze
    config : ConditionConfig, optional
        Dataset configuration (for dynamic condition mapping)
    """
    print("\n" + "="*60)
    print(f"Disease Analysis: {dataset}")
    print("="*60)
    assert config is not None, "Config must be provided"
    control_mapping = config.control_mapping
    if isinstance(control_mapping, dict):
        raise ValueError("Not implemented for dict control_mapping")

    disease_name = config.display_name
    ctr = control_mapping
    cond = config.treatment_groups
    assert len(cond) == 2, "Only two treatment groups supported"
    cond = cond[1]
    
    # Overall age acceleration plot
    print("Generating age acceleration plot...")
    obs_filtered = obs[obs['cell_type'].isin(cell_types)]
    wrapper_age_acceleration_disease(
        obs_filtered, 
        disease_dataset=dataset, 
        ctr=ctr, cond=cond,
        figsize=(3, 2.5)
    )
    plt.title('')
    output_path = os.path.join(output_dir, f'clock_{dataset}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: {output_path}")
    
    # Age-stratified analysis - all cell types
    print("Generating combined age-stratified plot...")
    bin_stats = wrapper_plot_age_acceleration_disease_bins(obs_filtered, disease_dataset=dataset, ctr=ctr, cond=cond)
    save_clock_stats(bin_stats, f'disease_bins_{dataset}')
    output_path = os.path.join(output_dir, f'clock_{dataset}_bins.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: {output_path}")


def analyze_aging(obs, dataset, output_dir, cell_types, config):
    """
    Analyze aging dataset comparing young vs old groups.
    
    Shows scatter plots of predicted age vs actual age for each age group,
    with unpaired t-test comparing predicted ages between groups.
    
    Parameters
    ----------
    obs : pd.DataFrame
        Observations with predictions
    dataset : str
        Dataset name
    output_dir : str
        Output directory for plots
    cell_types : list
        Cell types to analyze
    config : ConditionConfig
        Dataset configuration with age group info
    """
    from scipy.stats import spearmanr
    from scipy import stats
    import seaborn as sns
    import numpy as np
    
    print("\n" + "="*60)
    print(f"Aging Analysis: {dataset}")
    print("="*60)
    
    # Get age group column from config
    age_group_column = config.condition_column  # Should be 'age_group'
    
    # Get pretty names from config
    pretty_names = config.clock_pretty_names if config.clock_pretty_names else {}
    
    # Map age groups to pretty names for plotting
    if pretty_names:
        obs = obs.copy()
        obs['age_group_display'] = obs[age_group_column].map(lambda x: pretty_names.get(x, x))
        hue_column = 'age_group_display'
    else:
        hue_column = age_group_column
    
    # Generate scatter plots for each cell type
    stats_rows = []
    for cell_type in cell_types:
        obs_ct = obs[obs['cell_type'] == cell_type].copy()
        if len(obs_ct) == 0:
            print(f"  No data for {cell_type}, skipping...")
            continue
        
        print(f"\nGenerating aging scatter plot for {cell_type}...")
        
        # Perform unpaired t-test comparing predicted ages between age groups
        # order by actual age, not alphabetically -- 'Old' sorts before 'Young'
        age_groups = list(obs_ct.groupby(age_group_column, observed=True)['age'].mean()
                          .sort_values().index)
        
        if len(age_groups) == 2:
            young_group, old_group = age_groups[0], age_groups[1]
            young_pred = obs_ct[obs_ct[age_group_column] == young_group]['predicted_age'].dropna()
            old_pred = obs_ct[obs_ct[age_group_column] == old_group]['predicted_age'].dropna()
            
            if len(young_pred) >= 2 and len(old_pred) >= 2:
                # Test: old vs young (positive difference means old has higher predicted age)
                t_stat, p_value = stats.ttest_ind(old_pred, young_pred, equal_var=False)
                mean_diff = old_pred.mean() - young_pred.mean()
                
                # Determine significance stars
                if p_value < 0.001:
                    stars = '***'
                elif p_value < 0.01:
                    stars = '**'
                elif p_value < 0.05:
                    stars = '*'
                else:
                    stars = 'ns'
                
                print(f"  Unpaired t-test (predicted age):")
                print(f"    {pretty_names.get(young_group, young_group)}: {young_pred.mean():.1f} ± {young_pred.std():.1f} (n={len(young_pred)})")
                print(f"    {pretty_names.get(old_group, old_group)}: {old_pred.mean():.1f} ± {old_pred.std():.1f} (n={len(old_pred)})")
                print(f"    Difference (old - young): {mean_diff:.1f}, p={p_value:.3e} {stars}")
                stats_rows.append({'dataset': dataset, 'cell_type': cell_type,
                                   'young_group': young_group, 'old_group': old_group,
                                   'delta_predicted_age': mean_diff, 'p_value': p_value,
                                   'n_young': len(young_pred), 'n_old': len(old_pred)})
            else:
                p_value = None
                stars = None
                print(f"  Warning: Not enough data for t-test")
        else:
            p_value = None
            stars = None
            print(f"  Warning: Expected 2 age groups, found {len(age_groups)}")
        
        # Create strip plot showing predicted ages (NOT residuals for aging comparison)
        fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
        
        # Set the correct order for age groups (young first, then old)
        if hue_column == 'age_group_display':
            # Use pretty names in correct order
            age_order = [pretty_names.get('young', 'young'), pretty_names.get('old', 'old')]
        else:
            age_order = ['young', 'old']
        
        # Strip plot with age group coloring - use predicted_age
        sns.stripplot(
            data=obs_ct,
            x=hue_column,
            y='predicted_age',
            hue=hue_column,  # Color by age group
            order=age_order,  # Ensure correct order: young then old
            ax=ax,
            alpha=0.7,
            s=6,
            palette=None,  # Let seaborn auto-generate palette
            legend=False  # Remove legend since x-axis already shows groups
        )
        
        ax.set_ylabel("Predicted age (years)")
        ax.set_xlabel("")
        ax.spines[['top', 'right']].set_visible(False)
        ax.margins(x=0.3, y=0.2)  # Add more margin on x and y axes
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)  # Add zero line
        
        # Add statistical bracket and annotation if t-test was performed
        if p_value is not None and stars is not None:
            # Get y-axis limits after setting margins
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            
            # Position bracket higher with more space
            bracket_y = y_max - y_range * 0.15
            bracket_height = y_range * 0.03
            
            # Get x positions for the two age groups (centers of the categories)
            x_positions = [0, 1]  # Strip plot uses categorical positions 0, 1, etc.
            
            # Draw bracket
            ax.plot([x_positions[0], x_positions[0], x_positions[1], x_positions[1]],
                   [bracket_y, bracket_y + bracket_height, bracket_y + bracket_height, bracket_y],
                   'k-', linewidth=1.5)
            
            # Add p-value and stars
            bracket_text = f'p={p_value:.3e}\n{stars}' if p_value >= 0.001 else f'{stars}'
            ax.text((x_positions[0] + x_positions[1]) / 2, bracket_y + bracket_height + y_range * 0.02,
                   bracket_text, ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Set title
        ax.set_title(cell_type, fontsize=10, pad=10)
        
        # Save plot
        output_path = os.path.join(output_dir, f'clock_{dataset}_aging_{cell_type}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Saved: {output_path}")
    
    # Combined plot with all cell types
    print(f"\nGenerating combined aging scatter plot...")
    n_cell_types = len([ct for ct in cell_types if len(obs[obs['cell_type'] == ct]) > 0])
    
    if n_cell_types > 0:
        fig, axes = plt.subplots(1, n_cell_types, figsize=(3.5 * n_cell_types, 2.5), 
                                sharey=True, squeeze=False)
        axes = axes.flatten()
        
        for idx, cell_type in enumerate(cell_types):
            obs_ct = obs[obs['cell_type'] == cell_type].copy()
            if len(obs_ct) == 0:
                continue
            
            ax = axes[idx]
            
            # Perform unpaired t-test for this cell type
            # order by actual age, not alphabetically -- 'Old' sorts before 'Young'
            age_groups = list(obs_ct.groupby(age_group_column, observed=True)['age'].mean()
                              .sort_values().index)
            
            if len(age_groups) == 2:
                young_group, old_group = age_groups[0], age_groups[1]
                young_pred = obs_ct[obs_ct[age_group_column] == young_group]['predicted_age'].dropna()
                old_pred = obs_ct[obs_ct[age_group_column] == old_group]['predicted_age'].dropna()
                
                if len(young_pred) >= 2 and len(old_pred) >= 2:
                    t_stat, p_value = stats.ttest_ind(old_pred, young_pred, equal_var=False)
                    
                    # Determine significance stars
                    if p_value < 0.001:
                        stars = '***'
                    elif p_value < 0.01:
                        stars = '**'
                    elif p_value < 0.05:
                        stars = '*'
                    else:
                        stars = 'ns'
                else:
                    p_value = None
                    stars = None
            else:
                p_value = None
                stars = None
            
            plot_scatter_age_vs_predictedAge(
                obs_ct, 
                dataset=dataset, 
                ax=ax, 
                hue=hue_column,
                palette=None,  # Let seaborn auto-generate palette
                s=30, 
                alpha=0.7
            )
            
            # Add statistical bracket and annotation if t-test was performed
            if p_value is not None and stars is not None:
                # Get y-axis limits
                y_min, y_max = ax.get_ylim()
                y_range = y_max - y_min
                
                # Position bracket at top of plot
                bracket_y = y_max - y_range * 0.05
                bracket_height = y_range * 0.02
                
                # Get x positions for the two age groups
                x_min, x_max = ax.get_xlim()
                x_center_young = x_min + (x_max - x_min) * 0.25
                x_center_old = x_min + (x_max - x_min) * 0.75
                
                # Draw bracket
                ax.plot([x_center_young, x_center_young, x_center_old, x_center_old],
                       [bracket_y, bracket_y + bracket_height, bracket_y + bracket_height, bracket_y],
                       'k-', linewidth=1.5)
                
                # Add p-value and stars
                bracket_text = f'p={p_value:.3e}\n{stars}' if p_value >= 0.001 else f'{stars}'
                ax.text((x_center_young + x_center_old) / 2, bracket_y + bracket_height + y_range * 0.01,
                       bracket_text, ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_title(cell_type, fontsize=10, pad=10)
            
            # Only show legend on last plot
            if idx < len(cell_types) - 1:
                if ax.get_legend():
                    ax.get_legend().remove()
        
        save_clock_stats(pd.DataFrame(stats_rows), f'aging_{dataset}')
        output_path = os.path.join(output_dir, f'clock_{dataset}_aging_combined.png')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Saved: {output_path}")


def analyze_perturbation(obs_pert, dataset, output_dir, experiments, pval_map, 
                         p_value_t, config, pretty_names=None, mock_names=False):
    print("\n" + "="*60)
    print(f"Perturbation Analysis: {dataset}")
    print("="*60)

    for cell_type in obs_pert['cell_type'].unique():
        df_all, rejuvenating, aging = prepare_plot_inputs(
            obs_pert, cell_type, experiments, pval_map, p_value_t
        )
        print(f"\n{cell_type}:")
        print(f"  Rejuvenating: {len(rejuvenating)}")
        print(f"  Aging: {len(aging)}")
        # nothing significant -> no figure is written, so drop any stale one from an earlier run
        for group, name in [(rejuvenating, 'rejuvenating'), (aging, 'acceleration')]:
            if not group:
                suffix = '_mocked' if (mock_names and name == 'rejuvenating') else ''
                stale = os.path.join(output_dir, f'clock_{dataset}_{cell_type}_{name}{suffix}.png')
                if os.path.exists(stale):
                    os.remove(stale)
                    print(f"  Removed stale plot: {stale}")
        # Plot rejuvenating effects
        if len(rejuvenating) > 0:
            name_mapping = pretty_names if pretty_names else {}
            highlight_treatments = []  # Track which treatments to highlight
            
            # Handle mock names for OP dataset
            if mock_names:
                rejuvenating_sorted = sorted(
                    rejuvenating, 
                    key=lambda x: pval_map.get((cell_type, x[0], x[1]), (1.0, 0))[1]
                )
                name_mapping = {}
                # Get target treatments from config
                target_treatments = config.treatment_groups if hasattr(config, 'treatment_groups') and config.treatment_groups != 'all' else []
                
                compound_counter = 1
                for rank, (ctr, treatment) in enumerate(rejuvenating_sorted, start=1):
                    # Show real name if in target_treatments, otherwise mock it
                    if treatment in target_treatments:
                        name_mapping[treatment] = treatment
                        highlight_treatments.append(treatment)  # Mark for red highlighting
                    else:
                        name_mapping[treatment] = f"Compound {compound_counter}"
                        compound_counter += 1
                rejuvenating = rejuvenating_sorted
            
            plot_group_strip(
                df_all, rejuvenating, "Rejuvenating", cell_type, 
                pval_map,
                name_mapping=name_mapping,
                highlight_treatments=highlight_treatments if mock_names else None
            )
            plt.title('')
            
            suffix = '_mocked' if mock_names else ''
            output_path = os.path.join(
                output_dir, 
                f'clock_{dataset}_{cell_type}_rejuvenating{suffix}.png'
            )
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"  Saved rejuvenating plot: {output_path}")
        
        # Plot aging/acceleration effects
        if len(aging) > 0:
            name_mapping_aging = pretty_names if pretty_names else {}
            plot_group_strip(
                df_all, aging, "Acceleration", cell_type, pval_map, 
                name_mapping=name_mapping_aging,
                highlight_treatments=None
            )
            # plt.title(cell_type, pad=15)
            output_path = os.path.join(
                output_dir, 
                f'clock_{dataset}_{cell_type}_acceleration.png'
            )
            plt.title('')
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"  Saved acceleration plot: {output_path}")


def plot_specific_perturbations(obs_pert, dataset, cell_types, pval_map, 
                                pretty_names, output_dir, test_type='mixed-effect'):
    
    for cell_type in cell_types:
        df_all = obs_pert[obs_pert['cell_type'] == cell_type]
        if len(df_all) == 0:
            continue
        
        if dataset == 'CXCL9':
            # Rejuvenating effects
            exp = [
                ('RPMI', 'RPMI + ruxolitinib'), 
                ('LPS', 'LPS + ruxolitinib')
            ]
            plot_group_strip(
                df_all, exp, "Rejuvenating", cell_type, pval_map, 
                name_mapping=pretty_names
            )
            plt.title('')
            output_path = os.path.join(
                output_dir, 
                f'clock_{dataset}_{cell_type}_rejuv.png'
            )
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"  Saved: {output_path}")
            
            # Acceleration effects
            exp = [('RPMI', 'LPS')]
            plot_group_strip(
                df_all, exp, "Acceleration", cell_type, pval_map, 
                name_mapping=pretty_names
            )
            plt.title('')
            output_path = os.path.join(
                output_dir, 
                f'clock_{dataset}_{cell_type}_acc.png'
            )
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"  Saved: {output_path}")
            
        elif dataset == 'op':
            # Rejuvenating effects
            exp = [
                ('DMSO', 'Ruxolitinib')
            ]
            plot_group_strip(
                df_all, exp, "Rejuvenating", cell_type, pval_map, 
                name_mapping=pretty_names
            )
            plt.title('')
            output_path = os.path.join(
                output_dir, 
                f'clock_{dataset}_{cell_type}_rejuv.png'
            )
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"  Saved: {output_path}")
            
            
        elif dataset == 'parsebioscience':
            plot_experiment(
                test_type, df_all, 
                ctr='PBS', 
                treatment='IL-10', 
                cell_type=cell_type, 
                pval_map=pval_map
            )
            plt.legend(loc=(1.2, -.1), frameon=False)
            plt.suptitle(f'{cell_type}', y=1.1, weight='bold', fontsize=12)
            
            output_path = os.path.join(
                output_dir, 
                f'clock_{dataset}_{cell_type}_IL10.png'
            )
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (e.g., SLE_European, CXCL9, op, parsebioscience, soundlife)'
    )
    parser.add_argument(
        '--analysis-type',
        type=str,
        choices=['disease', 'perturbation', 'aging'],
        required=True,
        help='Type of analysis: disease, perturbation, or aging'
    )
    parser.add_argument(
        '--config-label',
        type=str,
        default=None,
        help='Config label for datasets with multiple configurations (unused, kept for CLI compatibility)'
    )

    # Analysis parameters
    parser.add_argument(
        '--cell-types',
        type=str,
        nargs='+',
        default=['CD4T', 'CD8T'],
        help='Cell types to analyze (default: CD4T CD8T)'
    )
    parser.add_argument(
        '--from-stats',
        action='store_true',
        help='Plot from the predictions_<dataset>.csv an earlier run saved, instead of predicting (no bulk data needed)'
    )
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = os.path.join(PLOTS_DIR, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration
    try:
        config = get_config(args.dataset)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    p_value_threshold = config.clock_pvalue_threshold if config.clock_pvalue_threshold else 0.05

    # Get predictions
    print("\nLoading predictions...")
    if args.from_stats:
        obs = pd.read_csv(f'{CLOCK_STATS_DIR}/predictions_{args.dataset}.csv', index_col=0)
        obs = obs[obs['cell_type'].isin(args.cell_types)]
    else:
        obs = wrapper_clock_predictions(
            args.cell_types,
            [args.dataset],
            only_sig_genes=True
        )
    
    if len(obs) == 0:
        raise ValueError("Error: No predictions loaded. Check that data files exist.")

    if not args.from_stats:
        save_clock_stats(obs.reset_index(), f'predictions_{args.dataset}')
    
    if args.analysis_type == 'disease':
        # obs = obs[obs['age_group'] == 'young']  # Filter out young samples for disease analysis
        analyze_disease(obs, args.dataset, output_dir, args.cell_types, config)
    
    elif args.analysis_type == 'aging':
        analyze_aging(obs, args.dataset, output_dir, args.cell_types, config)
    
    elif args.analysis_type == 'perturbation':
        # Prepare perturbation data
        obs_pert = obs.copy()        
        # # Filter out specific conditions if needed (e.g., Belinostat for CXCL9)
        if args.dataset == 'op':
            obs_pert = obs_pert[~obs_pert['condition'].isin(['Belinostat'])]
        
        # Get experiment setup from config
        experiments, pvalue_show_type, pretty_names, test_type, group_key, mock_names, plot_config = get_experiment_setup(
            obs_pert, config
        )
        # Perform statistical tests
        print("\nPerforming statistical tests...")
        pval_map = perform_statistical_tests(
            obs_pert, 
            experiments, 
            args.dataset,
            test_type=test_type,
            pvalue_show_type=pvalue_show_type,
            group_key=group_key
        )
        
        # Generate plots
        analyze_perturbation(
            obs_pert, 
            args.dataset, 
            output_dir, 
            experiments, 
            pval_map,
            p_value_threshold,
            config,
            pretty_names=pretty_names,
            mock_names=mock_names
        )
        
        # Generate specific perturbation plots
        
        plot_specific_perturbations(
            obs_pert, 
            args.dataset, 
            args.cell_types, 
            pval_map,
            pretty_names, 
            output_dir,
            test_type=test_type
        )
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"All plots saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
