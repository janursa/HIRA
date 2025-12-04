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
import os
import sys
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import anndata as ad
from statsmodels.stats.multitest import multipletests

# Import common utilities and configuration
from ciim.src.common import (
    PLOTS_DIR, 
    SAVE_DIR,
    cell_types as default_cell_types,
    surrogate_names,
    colors_blind
)
from ciim.src.feature_association.config import get_config
from ciim.src.utils.util import test_mixed_effects, test_paired, test_unpaired
from ciim.src.clock.plots import (
    wrapper_age_acceleration_disease,
    wrapper_plot_age_acceleration_disease_bins,
    plot_group_strip,
    plot_experiment,
    plot_scatter_age_vs_predictedAge
)

# Import from GRNimmuneClock package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../GRNimmuneClock'))
from grnimmuneclock import predict_age

warnings.filterwarnings("ignore")

# Set matplotlib defaults
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "Arial"


def get_all_predictions(cell_types, datasets, feature_type='gene_expression', 
                       data_type='bulk', reg_type='ridge', version='all_data'):
    """
    Get predictions for all cell types and datasets.
    
    Parameters
    ----------
    cell_types : list
        List of cell types to analyze
    datasets : list
        List of dataset names
    feature_type : str
        Feature type (gene_expression or tf_activity)
    data_type : str
        Data type (bulk or sc)
    reg_type : str
        Regression type (ridge, lasso, etc.)
    version : str
        Model version
        
    Returns
    -------
    pd.DataFrame
        DataFrame with predictions and metadata
    """
    obs_store = []
    for cell_type in cell_types:
        for dataset in datasets:
            adata_path = f"{SAVE_DIR}/{feature_type}_smoothed/{dataset}_{cell_type}_{data_type}.h5ad"
            
            if not os.path.exists(adata_path):
                print(f"Warning: File not found {adata_path}, skipping...")
                continue
                
            adata = ad.read_h5ad(adata_path)
            adata.obs['condition'] = adata.obs['condition'].apply(
                lambda name: surrogate_names.get(name, name)
            )
            conds = adata.obs['condition'].unique()
            
            for cond in conds:
                adata_c = adata[adata.obs['condition'] == cond]
                adata_c = predict_age(
                    adata_c, cell_type, 
                    feature_type=feature_type, 
                    data_type=data_type, 
                    reg_type=reg_type, 
                    version=version
                )
                obs = adata_c.obs
                obs['dataset'] = dataset
                obs['cell_type'] = cell_type
                obs['condition'] = cond
                obs_store.append(obs)
    
    obs = pd.concat(obs_store, axis=0)
    return obs


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


def get_experiment_setup(dataset, obs_pert, config):
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
    test_type = config.clock_test_type if config.clock_test_type else 'mixed_effect'
    group_key = config.clock_group_key if config.clock_group_key else 'donor_id'
    mock_names = config.clock_mock_names if hasattr(config, 'clock_mock_names') else False
    plot_config = config.clock_plot_config if config.clock_plot_config else {}
    
    # Get experiments
    if config.clock_experiments == 'auto' or config.clock_experiments is None:
        # Auto-generate experiments from control group
        control = config.control_group
        experiments = [
            (control, treatment) 
            for treatment in obs_pert['condition'].unique() 
            if treatment != control
        ]
    else:
        experiments = config.clock_experiments
    
    return experiments, pvalue_show_type, pretty_names, test_type, group_key, mock_names, plot_config


def perform_statistical_tests(obs_pert, experiments, dataset, test_type='mixed_effect', 
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
            if df_sub[df_sub['condition'] == treatment]['donor_age'].nunique() < 3:
                print(f"Skipping {cell_type} {ctr} vs {treatment} due to insufficient data.")
                continue

            # Perform statistical test
            if test_type == 'paired':
                p_value, slope = test_paired(df_sub, ctr, treatment)
            elif test_type == 'unpaired':
                p_value, slope = test_unpaired(df_sub, ctr, treatment)
            elif test_type == 'mixed_effect':
                p_value, slope = test_mixed_effects(
                    dataset, df_sub, ctr, treatment, 
                    target_variable='predicted_age', 
                    group_key=group_key
                )

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
            
            print(f"{cell_type}: {pvalue_show_type} p-values - {corrected_pvals}")
            
            for (label, corr_pval, slope) in zip(test_labels, corrected_pvals, slopes):
                pval_map[label] = (corr_pval, slope)
    
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
        ['test_group', 'condition', 'predicted_age']
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
    
    # If config specifies a different condition_column, use that instead of 'condition'
    if config and config.condition_column != 'condition':
        if config.condition_column in obs.columns:
            print(f"Using {config.condition_column} as condition column")
            obs = obs.copy()
            obs['condition'] = obs[config.condition_column].astype(str)
            # Apply name mapping if available
            if config.name_mapping:
                obs['condition'] = obs['condition'].map(lambda x: config.name_mapping.get(x, x))
        else:
            print(f"Warning: Condition column '{config.condition_column}' not found in data")
    
    # Overall age acceleration plot
    print("Generating age acceleration plot...")
    obs_filtered = obs[obs['cell_type'].isin(cell_types)]
    wrapper_age_acceleration_disease(
        obs_filtered, 
        disease_dataset=dataset, 
        figsize=(3, 2.5),
        config=config
    )
    plt.title('')
    output_path = os.path.join(output_dir, f'clock_{dataset}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
    plt.close()
    print(f"  Saved: {output_path}")
    
    # Age-stratified analysis - per cell type
    for cell_type in cell_types:
        obs_ct = obs[obs['cell_type'] == cell_type]
        if len(obs_ct) == 0:
            continue
            
        print(f"Generating age-stratified plot for {cell_type}...")
        wrapper_plot_age_acceleration_disease_bins(obs_ct, disease_dataset=dataset, config=config)
        output_path = os.path.join(output_dir, f'clock_{dataset}_bins_{cell_type}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
        plt.close()
        print(f"  Saved: {output_path}")
    
    # Age-stratified analysis - all cell types
    print("Generating combined age-stratified plot...")
    wrapper_plot_age_acceleration_disease_bins(obs_filtered, disease_dataset=dataset, config=config)
    output_path = os.path.join(output_dir, f'clock_{dataset}_bins.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
    plt.close()
    print(f"  Saved: {output_path}")


def analyze_aging(obs, dataset, output_dir, cell_types, config):
    """
    Analyze aging dataset comparing young vs old groups.
    
    Shows scatter plots of predicted age vs actual age for each age group,
    with Spearman correlation calculated for each.
    
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
    
    print("\n" + "="*60)
    print(f"Aging Analysis: {dataset}")
    if config.config_label:
        print(f"Configuration: {config.config_label}")
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
    for cell_type in cell_types:
        obs_ct = obs[obs['cell_type'] == cell_type].copy()
        if len(obs_ct) == 0:
            print(f"  No data for {cell_type}, skipping...")
            continue
        
        print(f"\nGenerating aging scatter plot for {cell_type}...")
        
        # Calculate Spearman correlation for each age group
        correlations = {}
        for age_group in obs_ct[age_group_column].unique():
            obs_group = obs_ct[obs_ct[age_group_column] == age_group]
            if len(obs_group) > 2:  # Need at least 3 points for correlation
                corr, pval = spearmanr(obs_group['age'], obs_group['predicted_age'])
                display_name = pretty_names.get(age_group, age_group)
                correlations[display_name] = {'r': corr, 'p': pval, 'n': len(obs_group)}
        
        # Print correlations
        print(f"  Spearman correlations for {cell_type}:")
        for group_name, stats in correlations.items():
            print(f"    {group_name}: r={stats['r']:.3f}, p={stats['p']:.3e}, n={stats['n']}")
        
        # Create scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))
        plot_scatter_age_vs_predictedAge(
            obs_ct, 
            dataset=dataset, 
            ax=ax, 
            hue=hue_column,
            palette=None,  # Let seaborn auto-generate palette
            s=30, 
            alpha=0.7
        )
        
        # Add correlation info to legend or title
        if len(correlations) > 0:
            corr_text = ' | '.join([f"{name}: r={s['r']:.2f}" for name, s in correlations.items()])
            ax.set_title(f"{cell_type}\n{corr_text}", fontsize=9, pad=10)
        else:
            ax.set_title(cell_type, fontsize=10, pad=10)
        
        # Save plot
        config_suffix = f"_{config.config_label}" if config.config_label else ""
        output_path = os.path.join(output_dir, f'clock_{dataset}{config_suffix}_aging_{cell_type}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
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
            plot_scatter_age_vs_predictedAge(
                obs_ct, 
                dataset=dataset, 
                ax=ax, 
                hue=hue_column,
                palette=None,  # Let seaborn auto-generate palette
                s=30, 
                alpha=0.7
            )
            
            # Calculate and add correlation
            correlations = {}
            for age_group in obs_ct[age_group_column].unique():
                obs_group = obs_ct[obs_ct[age_group_column] == age_group]
                if len(obs_group) > 2:
                    corr, pval = spearmanr(obs_group['age'], obs_group['predicted_age'])
                    display_name = pretty_names.get(age_group, age_group)
                    correlations[display_name] = corr
            
            if len(correlations) > 0:
                corr_text = ' | '.join([f"{name}: r={r:.2f}" for name, r in correlations.items()])
                ax.set_title(f"{cell_type}\n{corr_text}", fontsize=9, pad=10)
            else:
                ax.set_title(cell_type, fontsize=10, pad=10)
            
            # Only show legend on last plot
            if idx < len(cell_types) - 1:
                if ax.get_legend():
                    ax.get_legend().remove()
        
        config_suffix = f"_{config.config_label}" if config.config_label else ""
        output_path = os.path.join(output_dir, f'clock_{dataset}{config_suffix}_aging_combined.png')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
        plt.close()
        print(f"  Saved: {output_path}")


def analyze_perturbation(obs_pert, dataset, output_dir, experiments, pval_map, 
                         p_value_t, config, pretty_names=None, mock_names=False):
    """
    Analyze perturbation dataset for rejuvenating/accelerating effects.
    
    Parameters
    ----------
    obs_pert : pd.DataFrame
        Observations with predictions
    dataset : str
        Dataset name
    output_dir : str
        Output directory for plots
    experiments : list
        List of (control, treatment) tuples
    pval_map : dict
        P-value mapping
    p_value_t : float
        P-value threshold
    config : ConditionConfig
        Dataset configuration
    pretty_names : dict, optional
        Name mapping for display
    mock_names : bool
        Whether to mock compound names (keep top 1, rename others)
    """
    print("\n" + "="*60)
    print(f"Perturbation Analysis: {dataset}")
    print("="*60)
    
    # Get plot config
    plot_config = config.clock_plot_config if config.clock_plot_config else {}
    
    for cell_type in obs_pert['cell_type'].unique():
        df_all, rejuvenating, aging = prepare_plot_inputs(
            obs_pert, cell_type, experiments, pval_map, p_value_t
        )
        
        print(f"\n{cell_type}:")
        print(f"  Rejuvenating: {len(rejuvenating)}")
        print(f"  Aging: {len(aging)}")
        
        # Get plot parameters from config or use defaults
        if cell_type in plot_config:
            cell_plot_config = plot_config[cell_type]
        elif 'default' in plot_config:
            cell_plot_config = plot_config['default']
        else:
            cell_plot_config = {
                'rejuvenating': {'figsize': (7, 3), 'margins': (0.1, 0.2), 'ha': 'right', 'bbox_to_anchor': (1, 1)},
                'aging': {'figsize': (7, 3), 'margins': (0.1, 0.2), 'ha': 'right', 'bbox_to_anchor': (1, 1)},
            }
        
        rej_config = cell_plot_config.get('rejuvenating', {})
        age_config = cell_plot_config.get('aging', {})
        
        rej_figsize = rej_config.get('figsize', (7, 3))
        rej_margins = rej_config.get('margins', (0.1, 0.2))
        rej_ha = rej_config.get('ha', 'right')
        rej_bbox = rej_config.get('bbox_to_anchor', (1, 1))
        
        age_figsize = age_config.get('figsize', (7, 3))
        age_margins = age_config.get('margins', (0.1, 0.2))
        age_ha = age_config.get('ha', 'right')
        age_bbox = age_config.get('bbox_to_anchor', (1, 1))
        
        # Plot rejuvenating effects
        if len(rejuvenating) > 0:
            name_mapping = pretty_names if pretty_names else {}
            
            # Handle mock names for OP dataset
            if mock_names:
                rejuvenating_sorted = sorted(
                    rejuvenating, 
                    key=lambda x: pval_map.get((cell_type, x[0], x[1]), (1.0, 0))[1]
                )
                name_mapping = {}
                for rank, (ctr, treatment) in enumerate(rejuvenating_sorted, start=1):
                    if rank == 1:
                        name_mapping[treatment] = treatment
                    else:
                        name_mapping[treatment] = f"Compound {rank}"
                rejuvenating = rejuvenating_sorted
            
            plot_group_strip(
                df_all, rejuvenating, "Rejuvenating", cell_type, pval_map,
                figsize=rej_figsize, ha=rej_ha, bbox_to_anchor=rej_bbox,
                margins=rej_margins, name_mapping=name_mapping
            )
            plt.title('')
            
            suffix = '_mocked' if mock_names else ''
            output_path = os.path.join(
                output_dir, 
                f'clock_{dataset}_{cell_type}_rejuvenating{suffix}.png'
            )
            plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
            plt.close()
            print(f"  Saved rejuvenating plot: {output_path}")
        
        # Plot aging/acceleration effects
        if len(aging) > 0:
            plot_group_strip(
                df_all, aging, "Acceleration", cell_type, pval_map,
                figsize=age_figsize, margins=age_margins, ha=age_ha, 
                bbox_to_anchor=age_bbox, name_mapping=pretty_names if pretty_names else {}
            )
            plt.title(cell_type, pad=15)
            output_path = os.path.join(
                output_dir, 
                f'clock_{dataset}_{cell_type}_acceleration.png'
            )
            plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
            plt.close()
            print(f"  Saved acceleration plot: {output_path}")


def plot_specific_perturbations(obs_pert, dataset, cell_types, pval_map, 
                                pretty_names, output_dir, test_type='mixed_effect'):
    """
    Plot specific perturbation comparisons (dataset-specific).
    
    Parameters
    ----------
    obs_pert : pd.DataFrame
        Observations with predictions
    dataset : str
        Dataset name
    cell_types : list
        Cell types to plot
    pval_map : dict
        P-value mapping
    pretty_names : dict
        Name mapping
    output_dir : str
        Output directory
    test_type : str
        Type of statistical test
    """
    print("\nGenerating specific perturbation plots...")
    
    for cell_type in cell_types:
        df_all = obs_pert[obs_pert['cell_type'] == cell_type]
        
        if len(df_all) == 0:
            continue
        
        if dataset == 'CXCL9':
            # Rejuvenating effects
            exp = [
                ('24 h RPMI', '24 h RPMI + ruxolitinib'), 
                ('24 h LPS', '24 h LPS + ruxolitinib')
            ]
            plot_group_strip(
                df_all, exp, "Rejuvenating", cell_type, pval_map, 
                figsize=(3, 3), margins=(0.3, 0.3),
                name_mapping=pretty_names, bbox_to_anchor=(1, 1), 
                max_len=25, ha='center'
            )
            plt.title('')
            output_path = os.path.join(
                output_dir, 
                f'clock_{dataset}_{cell_type}_rejuv.png'
            )
            plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
            plt.close()
            print(f"  Saved: {output_path}")
            
            # Acceleration effects
            exp = [('24 h RPMI', '24 h LPS')]
            plot_group_strip(
                df_all, exp, "Acceleration", cell_type, pval_map, 
                figsize=(2.5, 3), margins=(0.3, 0.3),
                name_mapping=pretty_names, bbox_to_anchor=(1, 1), 
                max_len=25, ha='center'
            )
            plt.title('')
            output_path = os.path.join(
                output_dir, 
                f'clock_{dataset}_{cell_type}_acc.png'
            )
            plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
            plt.close()
            print(f"  Saved: {output_path}")
            
        elif dataset == 'op':
            fig, axes = plt.subplots(1, 2, figsize=(4, 1.6), sharey=True)
            
            ax = axes[0]
            plot_experiment(
                test_type, df_all, 
                ctr='Dimethyl Sulfoxide', 
                treatment='Ruxolitinib', 
                cell_type=cell_type, 
                pval_map=pval_map, 
                ax=ax
            )
            ax.get_legend().remove()
            
            ax = axes[1]
            plot_experiment(
                test_type, df_all, 
                ctr='Dimethyl Sulfoxide', 
                treatment='LY2090314', 
                cell_type=cell_type, 
                pval_map=pval_map, 
                ax=ax
            )
            plt.legend(loc=(1.1, .2), frameon=False)
            plt.suptitle(f'{cell_type}', y=1.1, weight='bold', fontsize=12)
            
            output_path = os.path.join(
                output_dir, 
                f'clock_{dataset}_{cell_type}_examples.png'
            )
            plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
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
            plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
            plt.close()
            print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Aging Clock Analysis - Post-prediction analysis for disease, perturbation, and aging datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Disease analysis (CMV effect)
  python clock_analysis.py --dataset soundlife --config-label cmv_young --analysis-type disease
  
  # Aging analysis (young vs old)
  python clock_analysis.py --dataset soundlife --config-label aging_cmv_neg --analysis-type aging
  
  # Perturbation analysis (all settings from config)
  python clock_analysis.py --dataset CXCL9 --analysis-type perturbation
  
  # Custom cell types
  python clock_analysis.py --dataset op --analysis-type perturbation --cell-types CD4T CD8T
        """
    )
    
    # Required arguments
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
        help='Config label for datasets with multiple configurations (e.g., aging_cmv_neg, cmv_young)'
    )
    
    # Model configuration
    parser.add_argument(
        '--feature-type',
        type=str,
        default='gene_expression',
        help='Feature type: gene_expression or tf_activity (default: gene_expression)'
    )
    parser.add_argument(
        '--data-type',
        type=str,
        default='bulk',
        help='Data type: bulk or sc (default: bulk)'
    )
    parser.add_argument(
        '--reg-type',
        type=str,
        default='ridge',
        help='Regression type (default: ridge)'
    )
    parser.add_argument(
        '--version',
        type=str,
        default='all_data',
        help='Model version (default: all_data)'
    )
    
    # Analysis parameters
    parser.add_argument(
        '--cell-types',
        type=str,
        nargs='+',
        default=['CD4T', 'CD8T'],
        help='Cell types to analyze (default: CD4T CD8T)'
    )
    
    # Optional overrides (if not specified, use config)
    parser.add_argument(
        '--p-value-threshold',
        type=float,
        default=None,
        help='P-value threshold for significance (default: from config)'
    )
    
    # Optional flags
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for plots (default: PLOTS_DIR from common.py)'
    )
    parser.add_argument(
        '--skip-specific-plots',
        action='store_true',
        help='Skip dataset-specific detailed plots'
    )
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else PLOTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration
    try:
        configs = get_config(args.dataset, config_label=args.config_label)
        config = configs[0]  # Use first config (most datasets have only one)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Override config with command-line arguments if provided
    p_value_threshold = args.p_value_threshold if args.p_value_threshold else config.clock_pvalue_threshold
    
    print("="*60)
    print(f"Aging Clock Analysis - {args.analysis_type.capitalize()}")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    if args.config_label:
        print(f"Config label: {args.config_label}")
    print(f"Analysis type: {args.analysis_type}")
    print(f"Feature type: {args.feature_type}")
    print(f"Data type: {args.data_type}")
    print(f"Model version: {args.version}")
    print(f"Cell types: {', '.join(args.cell_types)}")
    print(f"P-value threshold: {p_value_threshold}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    # Get predictions
    print("\nLoading predictions...")
    obs = get_all_predictions(
        args.cell_types, 
        [args.dataset],
        feature_type=args.feature_type,
        data_type=args.data_type,
        reg_type=args.reg_type,
        version=args.version
    )
    
    if len(obs) == 0:
        print("Error: No predictions loaded. Check that data files exist.")
        return
    
    print(f"Loaded {len(obs)} predictions")
    print(f"Conditions: {obs['condition'].unique()}")
    
    # Apply data filters if specified in config
    if config.data_filter:
        print("\nApplying data filters...")
        obs = apply_data_filter(obs, config)
        print(f"After filtering: {len(obs)} predictions")
        if len(obs) == 0:
            print("Error: No data remaining after filtering.")
            return
    
    print()
    
    # Run analysis based on type
    if args.analysis_type == 'disease':
        analyze_disease(obs, args.dataset, output_dir, args.cell_types, config)
    
    elif args.analysis_type == 'aging':
        analyze_aging(obs, args.dataset, output_dir, args.cell_types, config)
    
    elif args.analysis_type == 'perturbation':
        # Prepare perturbation data
        obs_pert = obs.copy()
        obs_pert['test_group'] = obs_pert['donor_age'].copy()
        
        # Filter out specific conditions if needed (e.g., Belinostat for CXCL9)
        if args.dataset == 'CXCL9':
            obs_pert = obs_pert[~obs_pert['condition'].isin(['Belinostat'])]
        
        # Get experiment setup from config
        experiments, pvalue_show_type, pretty_names, test_type, group_key, mock_names, plot_config = get_experiment_setup(
            args.dataset, obs_pert, config
        )
        
        print(f"Number of experiments: {len(experiments)}")
        print(f"P-value correction: {pvalue_show_type}")
        print(f"Test type: {test_type}")
        print(f"Group key: {group_key}")
        print(f"Mock names: {mock_names}")
        
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
        if not args.skip_specific_plots:
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
