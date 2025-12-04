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
from ciim.src.utils.util import test_mixed_effects, test_paired, test_unpaired
from ciim.src.clock.plots import (
    wrapper_age_acceleration_disease,
    wrapper_plot_age_acceleration_disease_bins,
    plot_group_strip,
    plot_experiment
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


def get_experiment_setup(dataset, obs_pert):
    """
    Get experiment setup (control-treatment pairs) for a dataset.
    
    Parameters
    ----------
    dataset : str
        Dataset name
    obs_pert : pd.DataFrame
        Observations dataframe
        
    Returns
    -------
    tuple
        (experiments, pvalue_show_type, pretty_names)
    """
    pretty_names = {}
    
    if dataset == 'CXCL9':
        pvalue_show_type = 'raw'
        experiments = [
            ('24 h RPMI', '24 h LPS'),
            ('24 h RPMI', '24 h RPMI + ruxolitinib'),
            ('24 h LPS', '24 h LPS + ruxolitinib'),
        ]
        pretty_names = {
            '24 h LPS': 'LPS \n (ctr: RPMI)',
            '24 h LPS + ruxolitinib': 'Ruxolitinib \n (ctr: LPS)',
            '24 h RPMI': 'RPMI',
            '24 h RPMI + ruxolitinib': 'Ruxolitinib \n (ctr: RPMI)',
        }
    elif dataset == 'op':
        pvalue_show_type = 'corrected'
        experiments = [
            ('Dimethyl Sulfoxide', treatment) 
            for treatment in obs_pert['condition'].unique() 
            if treatment != 'Dimethyl Sulfoxide'
        ]
    elif dataset == 'parsebioscience':
        pvalue_show_type = 'corrected'
        ctr = 'PBS'
        experiments = [
            (ctr, treatment) 
            for treatment in obs_pert['condition'].unique() 
            if treatment != ctr
        ]
    else:
        raise ValueError(f"Dataset {dataset} not recognized for perturbation experiments.")
    
    return experiments, pvalue_show_type, pretty_names


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


def analyze_disease(obs, dataset, output_dir, cell_types):
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
    """
    print("\n" + "="*60)
    print(f"Disease Analysis: {dataset}")
    print("="*60)
    
    # Overall age acceleration plot
    print("Generating age acceleration plot...")
    obs_filtered = obs[obs['cell_type'].isin(cell_types)]
    wrapper_age_acceleration_disease(
        obs_filtered, 
        disease_dataset=dataset, 
        figsize=(3, 2.5)
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
        wrapper_plot_age_acceleration_disease_bins(obs_ct, disease_dataset=dataset)
        output_path = os.path.join(output_dir, f'clock_{dataset}_bins_{cell_type}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
        plt.close()
        print(f"  Saved: {output_path}")
    
    # Age-stratified analysis - all cell types
    print("Generating combined age-stratified plot...")
    wrapper_plot_age_acceleration_disease_bins(obs_filtered, disease_dataset=dataset)
    output_path = os.path.join(output_dir, f'clock_{dataset}_bins.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
    plt.close()
    print(f"  Saved: {output_path}")


def analyze_perturbation(obs_pert, dataset, output_dir, experiments, pval_map, 
                         p_value_t, pretty_names=None, mock_names=False):
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
    pretty_names : dict, optional
        Name mapping for display
    mock_names : bool
        Whether to mock compound names (keep top 1, rename others)
    """
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
        
        # Configure plot parameters based on dataset
        if dataset == 'op':
            rej_figsize = (7.5, 3)
            rej_margins = (0.05, 0.2)
            rej_ha = 'right'
            rej_bbox = (1, 1)
            
            age_figsize = (5, 3)
            age_margins = (0.05, 0.2)
            age_ha = 'right'
            age_bbox = (1, 1)
        elif dataset == 'CXCL9':
            rej_figsize = (4, 3)
            rej_margins = (0.12, 0.2)
            rej_ha = 'right'
            rej_bbox = (1, 1.2)
            
            age_figsize = (7, 3)
            age_margins = (0.12, 0.2)
            age_ha = 'right'
            age_bbox = (1, 1.2)
        elif dataset == 'parsebioscience':
            if cell_type == 'CD4T':
                rej_figsize = (4, 3)
                rej_margins = (0.12, 0.2)
                age_figsize = (10, 3)
                age_margins = (0.12, 0.2)
            else:
                rej_figsize = (7, 3)
                rej_margins = (0.12, 0.2)
                age_figsize = (10, 3)
                age_margins = (0.12, 0.2)
            rej_ha = 'right'
            rej_bbox = (1, 1.2)
            age_ha = 'right'
            age_bbox = (1, 1.2)
        else:
            rej_figsize = age_figsize = (7, 3)
            rej_margins = age_margins = (0.1, 0.2)
            rej_ha = age_ha = 'right'
            rej_bbox = age_bbox = (1, 1)
        
        # Plot rejuvenating effects
        if len(rejuvenating) > 0:
            name_mapping = pretty_names if pretty_names else None
            
            # Handle mock names for OP dataset
            if mock_names and dataset == 'op':
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
                bbox_to_anchor=age_bbox, name_mapping=pretty_names
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
        description='Aging Clock Analysis - Post-prediction analysis for disease and perturbation datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Disease analysis
  python clock_analysis.py --dataset SLE_European --analysis-type disease
  
  # Perturbation analysis
  python clock_analysis.py --dataset CXCL9 --analysis-type perturbation --test-type mixed_effect
  
  # Custom cell types
  python clock_analysis.py --dataset op --analysis-type perturbation --cell-types CD4T CD8T
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (e.g., SLE_European, CXCL9, op, parsebioscience)'
    )
    parser.add_argument(
        '--analysis-type',
        type=str,
        choices=['disease', 'perturbation'],
        required=True,
        help='Type of analysis: disease or perturbation'
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
    parser.add_argument(
        '--test-type',
        type=str,
        default='mixed_effect',
        choices=['paired', 'unpaired', 'mixed_effect'],
        help='Statistical test type for perturbations (default: mixed_effect)'
    )
    parser.add_argument(
        '--group-key',
        type=str,
        default='donor_id',
        help='Column name for random effects grouping in mixed effects (default: donor_id)'
    )
    parser.add_argument(
        '--p-value-threshold',
        type=float,
        default=0.05,
        help='P-value threshold for significance (default: 0.05)'
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
    parser.add_argument(
        '--mock-names',
        action='store_true',
        help='Mock compound names (OP dataset only: keep top 1, rename others)'
    )
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else PLOTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print(f"Aging Clock Analysis - {args.analysis_type.capitalize()}")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Analysis type: {args.analysis_type}")
    print(f"Feature type: {args.feature_type}")
    print(f"Data type: {args.data_type}")
    print(f"Model version: {args.version}")
    print(f"Cell types: {', '.join(args.cell_types)}")
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
    print()
    
    # Run analysis based on type
    if args.analysis_type == 'disease':
        analyze_disease(obs, args.dataset, output_dir, args.cell_types)
    
    elif args.analysis_type == 'perturbation':
        # Prepare perturbation data
        obs_pert = obs.copy()
        obs_pert['test_group'] = obs_pert['donor_age'].copy()
        
        # Filter out specific conditions if needed
        if args.dataset == 'CXCL9':
            obs_pert = obs_pert[~obs_pert['condition'].isin(['Belinostat'])]
        
        # Get experiment setup
        experiments, pvalue_show_type, pretty_names = get_experiment_setup(
            args.dataset, obs_pert
        )
        
        print(f"Number of experiments: {len(experiments)}")
        print(f"P-value correction: {pvalue_show_type}")
        
        # Perform statistical tests
        print("\nPerforming statistical tests...")
        pval_map = perform_statistical_tests(
            obs_pert, 
            experiments, 
            args.dataset,
            test_type=args.test_type,
            pvalue_show_type=pvalue_show_type,
            group_key=args.group_key
        )
        
        # Generate plots
        analyze_perturbation(
            obs_pert, 
            args.dataset, 
            output_dir, 
            experiments, 
            pval_map,
            args.p_value_threshold,
            pretty_names=pretty_names,
            mock_names=args.mock_names
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
                test_type=args.test_type
            )
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"All plots saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
