#!/usr/bin/env python
"""
Unified Condition Analysis - Post-run visualization script

This script performs post-run analysis for both disease and perturbation experiments,
generating various visualizations including:
- Overview heatmap of cell types
- Overlap analysis with aging TFs
- Age-stratified analysis (disease) / Aging-perturbation comparison
- Case TF trends / Donor-level effects (perturbations)
- Pathway analysis
"""

import argparse
import os
import sys
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype

# Import common utilities and configuration
from ciim.src.common import (
    PLOTS_DIR, 
    SAVE_DIR,
    cell_types, 
    datasets_all,
    palette_trend,
    palette_disease_effect,
    palette_treatment,
    surrogate_names,
    mapping_minor_2_major
)
from ciim.src.feature_association.config import get_config
from ciim.src.feature_association.helper import retrieve_sig_stats
from ciim.src.feature_association.plots import (
    heamap_plot_minor_cell_types,
    plot_overlap,
    plot_analysis_and_centrality,
    plot_donor_level_perturbation_effect
)
from ciim.src.feature_association.disease import plot_healthy_disease_trend
from ciim.src.utils.util import retrieve_net_consensus
from ciim.src.pathway_analysis.util import pathway_kde_func
from ciim.src.pathway_analysis.plots import plot_pathway_kde

warnings.filterwarnings("ignore")

# Set matplotlib defaults
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "Arial"


def load_stats(dataset, data_type, feature_type, config_label=None):
    """Load statistics from saved CSV file."""
    cfg_list = get_config(dataset)
    
    # Special handling for 'cmv' label - combine cmv_young and cmv_old
    if config_label == 'cmv' and dataset == 'soundlife':
        print("Combining cmv_young and cmv_old configs...")
        
        # Load both configs
        cfg_young = next((c for c in cfg_list if c.config_label == 'cmv_young'), None)
        cfg_old = next((c for c in cfg_list if c.config_label == 'cmv_old'), None)
        
        if cfg_young is None or cfg_old is None:
            raise ValueError(f"Cannot find both cmv_young and cmv_old configs for dataset '{dataset}'")
        
        # Load stats file
        stats_path = f'{SAVE_DIR}/stats/stats_{dataset}_{data_type}_{feature_type}_{cfg_young.test_type}.csv'
        
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Stats file not found: {stats_path}")
        
        stats = pd.read_csv(stats_path)
        
        # Filter to cmv_young and cmv_old, then combine
        if 'config_label' in stats.columns:
            stats_young = stats[stats['config_label'] == 'cmv_young'].copy()
            stats_old = stats[stats['config_label'] == 'cmv_old'].copy()
            
            # Combine both age groups
            stats = pd.concat([stats_young, stats_old], ignore_index=True)
            
            # Update config_label to 'cmv' for combined data
            stats['config_label'] = 'cmv'
            
            print(f"  Loaded {len(stats_young)} rows from cmv_young")
            print(f"  Loaded {len(stats_old)} rows from cmv_old")
            print(f"  Combined: {len(stats)} rows")
        else:
            raise ValueError("Stats file does not contain 'config_label' column for filtering")
        
        # Return the young config as template (they have same structure)
        cfg = cfg_young
        
        return stats, cfg
    
    # Standard config handling
    if config_label:
        cfg = next((c for c in cfg_list if c.config_label == config_label), None)
        if cfg is None:
            raise ValueError(f"Config with label '{config_label}' not found for dataset '{dataset}'")
    else:
        cfg = cfg_list[0]  # Use first config (most datasets have only one)
    
    # For soundlife, all configs are in the same stats file (no config_label in filename)
    stats_path = f'{SAVE_DIR}/stats/stats_{dataset}_{data_type}_{feature_type}_{cfg.test_type}.csv'
    
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    
    stats = pd.read_csv(stats_path)
    
    # For soundlife with config_label, filter stats to the specific config
    if config_label and dataset == 'soundlife':
        # The stats file contains all configs, filter by the condition specified in the config
        # This will be handled by the config object's condition_column, control, treatment
        pass  # Stats filtering happens later based on config properties
    
    return stats, cfg


def get_condition_palette(analysis_type):
    """Get appropriate color palette based on analysis type."""
    if analysis_type == 'disease':
        return palette_disease_effect
    elif analysis_type == 'perturbation':
        return palette_treatment
    else:
        return palette_disease_effect  # Default


def plot_overview_heatmap(stats, dataset, analysis_type, output_dir, config_label=None):
    """Generate overview heatmap of minor cell types."""
    print("Generating overview heatmap...")
    
    palette = get_condition_palette(analysis_type)
    
    if analysis_type == 'disease' or analysis_type == 'aging':
        # Special handling for combined CMV (has young and old separately)
        if config_label == 'cmv' and dataset == 'soundlife':
            # Plot separate heatmaps for young and old
            for age_group in ['young', 'old']:
                stats_filtered = stats[stats['age_group'] == age_group].copy()
                if len(stats_filtered) > 0:
                    _plot_single_heatmap(stats_filtered, palette, output_dir, 
                                       suffix=f"{dataset}_cmv_{age_group}")
        else:
            # For disease/aging, filter to both age groups
            stats_filtered = stats[stats['age_group'] == 'Both age groups'].copy()
            _plot_single_heatmap(stats_filtered, palette, output_dir, 
                               suffix=dataset if (analysis_type == 'disease' or analysis_type == 'aging') else None)
    else:
        # For perturbations, process each condition separately
        cfg = get_config(dataset)[0]
        target_treatments = cfg.target_treatments
        
        if target_treatments is None:
            print("  Warning: No target treatments configured, using all conditions")
            target_treatments = stats['condition'].unique()
        
        for condition in target_treatments:
            stats_filtered = stats[stats['condition'] == condition].copy()
            _plot_single_heatmap(stats_filtered, palette, output_dir, suffix=condition)
        return


def _plot_single_heatmap(stats, palette, output_dir, suffix=None):
    """Helper function to plot a single heatmap."""
    stats['cell_type'] = pd.Categorical(stats['cell_type'], categories=cell_types, ordered=True)
    stats['major_cell_type'] = stats['cell_type'].astype(
        CategoricalDtype(categories=['CD4T', 'CD8T', 'NK', 'MONO', 'B'], ordered=True)
    )
    
    heamap_plot_minor_cell_types(
        stats, 
        slope_col='slope_condition', 
        palette=palette, 
        figsize=(2, 3), 
        sig_dots_y_offset=3, 
        annotate_x_ticks=False, 
        map_names={'cell_type': 'Sub type', 'major_cell_type': 'Cell type'},
        dendrogram_visible=False, 
        show_legend=False
    )
    
    suffix_str = f"_{suffix.replace(' ', '_').replace('(', '_').replace(')', '_').replace(':', '_')}" if suffix else ""
    output_path = os.path.join(output_dir, f'overview{suffix_str}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_aging_overlap(stats_sig, dataset, included_cell_types, analysis_type, agreement, output_dir):
    """Plot overlap between condition (disease/perturbation) and aging TFs."""
    print("Generating aging overlap plot...")
    
    cfg = get_config(dataset)[0]
    aging_stats_sig = retrieve_sig_stats(type='bulk').drop_duplicates(subset=['cell_type', 'tf'])
    aging_stats_sig = aging_stats_sig[['tf', 'cell_type', 'slope']].copy()
    
    # Determine which conditions to process
    if analysis_type == 'perturbation':
        target_treatments = cfg.target_treatments
        if target_treatments is None:
            target_treatments = stats_sig['condition'].unique()
        conditions_to_plot = target_treatments
    else:
        conditions_to_plot = [None]  # For disease, process all together
    
    for cell_type in included_cell_types:
        print(f"  Processing cell type: {cell_type}")
        
        for condition in conditions_to_plot:
            # Filter stats
            if condition is not None:
                df_sub = stats_sig[(stats_sig['condition'] == condition) & (stats_sig['cell_type'] == cell_type)].reset_index(drop=True)
            else:
                df_sub = stats_sig[stats_sig['cell_type'] == cell_type].reset_index(drop=True)
            
            aging_stats_sig_sub = aging_stats_sig[aging_stats_sig['cell_type'] == cell_type].reset_index(drop=True)
            
            if len(df_sub) == 0:
                print(f"    Warning: No data for {'condition ' + condition if condition else 'this cell type'}")
                continue
            
            # Calculate overlap statistics (aging TFs as base)
            # Use aging TFs as base (left merge) to calculate overlap
            merged = aging_stats_sig_sub[['tf', 'slope']].merge(
                df_sub[['tf', 'slope_condition']], 
                on='tf', 
                how='left'
            )
            
            # Filter to only those that have condition data (inner join equivalent)
            merged_inner = merged[merged['slope_condition'].notna()]
            
            if len(merged_inner) > 0:
                same_direction = (np.sign(merged_inner['slope']) == np.sign(merged_inner['slope_condition'])).sum()
                opposite_direction = (np.sign(merged_inner['slope']) == -np.sign(merged_inner['slope_condition'])).sum()
                total_overlap = len(merged_inner)
                
                print(f"    Overlap with aging TFs: {total_overlap}")
                print(f"      Same direction: {same_direction} ({same_direction/total_overlap*100:.1f}%)")
                print(f"      Opposite direction: {opposite_direction} ({opposite_direction/total_overlap*100:.1f}%)")
            
            plot_overlap(
                aging_stats_sig_sub[['tf', 'cell_type', 'slope']].reset_index(drop=True), 
                df_sub[['tf', 'cell_type', 'slope_condition']].reset_index(drop=True), 
                col='cell_type', 
                how='left', 
                agreement=agreement, 
                legend=True, 
                figsize=(1.5, 2), 
                legend_loc=(1, 0.5)
            )
            
            if analysis_type == 'perturbation':
                plt.title('')
                plt.legend().remove()
            
            name_suffix = f'{dataset}_{condition}_{cell_type}' if condition else f'{dataset}_{cell_type}'
            output_path = os.path.join(output_dir, f'condition_aging_overlap_{name_suffix}.png')
            output_path = output_path.replace(' ', '_').replace('(', '_').replace(')', '_').replace(':', '_')
            plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
            plt.close()
            print(f"    Saved: {output_path}")


def plot_aging_disease_overlap_heatmap(stats_sig, dataset, cell_types, output_dir):
    """
    Plot heatmap comparing TF activity/directions between aging and disease/aging condition.
    (Disease/Aging-specific visualization showing side-by-side slopes)
    Creates separate heatmaps for each cell type with TFs on y-axis.
    """
    print("Generating aging vs condition overlap heatmap...")
    
    import seaborn as sns
    from matplotlib.patches import Rectangle
        
    # Load aging stats
    aging_stats_sig = retrieve_sig_stats(type='bulk').drop_duplicates(subset=["cell_type", "tf"])
    
    # Process each cell type separately
    for cell_type in cell_types:
        if cell_type not in stats_sig['cell_type'].unique():
            print(f"  Warning: No data for {cell_type}")
            continue
            
        print(f"  Processing cell type: {cell_type}")
        
        # Get condition data for this cell type
        condition_df = stats_sig[
            stats_sig['cell_type'] == cell_type
        ][['tf', 'slope_condition']].copy()
        
        # Get aging data for same cell type
        aging_df = aging_stats_sig[
            aging_stats_sig['cell_type'] == cell_type
        ][['tf', 'slope']].copy()
        
        # Get union of TFs from both condition and aging
        all_tfs = sorted(list(set(condition_df['tf'].tolist() + aging_df['tf'].tolist())))
        
        # Create a dataframe with all TFs
        merged = pd.DataFrame({'tf': all_tfs})
        merged = merged.merge(condition_df, on='tf', how='left')
        merged = merged.merge(aging_df, on='tf', how='left')
        
        if len(merged) == 0:
            print(f"    Warning: No TFs found for {cell_type}")
            continue
        
        # Sort by aging slope (Age-associated TFs column), put NaNs at the end
        merged['sort_key'] = merged['slope'].fillna(999)  # NaNs go to bottom
        merged = merged.sort_values('sort_key', ascending=False).drop(columns=['sort_key'])
        
        n_tfs = len(merged)
        print(f"    Total TFs (union): {n_tfs}")
        print(f"      Condition TFs: {condition_df['tf'].nunique()}")
        print(f"      Aging TFs: {aging_df['tf'].nunique()}")
        
        # Prepare data matrix: rows are TFs, columns are [Aging, Condition]
        data_matrix = merged[['slope', 'slope_condition']].values
        
        # Convert to binary: +1 for positive slope, -1 for negative slope
        binary_matrix = np.sign(data_matrix)
        
        # Create heatmap with TFs on y-axis
        figsize = (2.5, max(6, n_tfs * 0.05))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap with binary colormap using palette_trend colors
        from matplotlib.colors import ListedColormap
        # Decrease in aging: '#B0BF1A' (greenish), Increase in aging: '#E52B50' (red)
        colors = [palette_trend['Decrease in aging'], '#f7f7f7', palette_trend['Increase in aging']]  # decrease, white (for NaN), increase
        cmap = ListedColormap(colors)
        
        im = ax.imshow(
            binary_matrix,
            aspect='auto',
            cmap=cmap,
            vmin=-1,
            vmax=1,
            interpolation='nearest'
        )
        
        # Add vertical line to separate aging from condition
        ax.axvline(0.5, color='black', linewidth=2)
        
        # Set column labels
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Age-associated\nTFs', 'SoundLife'], fontsize=9, rotation=45, ha='right')
        ax.set_xlabel('')
        
        # Hide y-axis labels (TF names) but show count
        ax.set_yticks([])
        ax.set_ylabel(f'TFs (n={n_tfs})', fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[-1, 1])
        cbar.ax.set_yticklabels(['Decrease', 'Increase'], fontsize=8)
        cbar.set_label('Direction', rotation=270, labelpad=15, fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        
        # Title
        ax.set_title(f'{cell_type}', fontsize=10, weight='bold', pad=10)
        
        # Save (removed agreement statistics annotation)
        output_path = os.path.join(output_dir, f'aging_condition_overlap_heatmap_{dataset}_{cell_type}.png')
        output_path = output_path.replace(' ', '_').replace('(', '_').replace(')', '_').replace(':', '_')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
        plt.close()
        print(f"    Saved: {output_path}")


def plot_aging_experiment_heatmap(stats_sig, dataset, output_dir):
    """
    Plot heatmap comparing TF activity/directions between aging and experimental conditions.
    (Perturbation-specific visualization)
    """
    print("Generating aging vs experiment heatmap...")
    
    import seaborn as sns
    from matplotlib.patches import Rectangle
        
    # Load aging stats
    aging_stats_sig = retrieve_sig_stats(type='bulk').drop_duplicates(subset=["cell_type", "tf"])
    
    cfg = get_config(dataset)[0]
    target_treatments = cfg.target_treatments
    
    if target_treatments is None:
        print("  Warning: No target treatments configured")
        return
    
    for cell_type in stats_sig['cell_type'].unique():
        print(f"  Processing cell type: {cell_type}")
        
        for treatment in target_treatments:
            # Get experiment data
            exp_df = stats_sig[
                (stats_sig['condition'] == treatment) & 
                (stats_sig['cell_type'] == cell_type)
            ][['tf', 'slope_condition']].copy()
            
            # Get aging data for same cell type
            aging_df = aging_stats_sig[
                aging_stats_sig['cell_type'] == cell_type
            ][['tf', 'slope']].copy()
            
            # Merge on TFs that are in the experiment (left join)
            merged = exp_df.merge(aging_df, on='tf', how='left')
            
            if len(merged) == 0:
                print(f"    Warning: No data for {treatment} in {cell_type}")
                continue
            
            # Sort by experiment slope for better visualization
            merged = merged.sort_values('slope_condition', ascending=False)
            
            # Prepare data matrix
            data_matrix = merged[['slope', 'slope_condition']].values
            
            # Create heatmap
            n_tfs = len(merged)
            figsize = (2.5, max(6, n_tfs * 0.03))  
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot heatmap with diverging colormap
            im = ax.imshow(
                data_matrix,
                aspect='auto',
                vmin=-max(abs(data_matrix.min()), abs(data_matrix.max())),
                vmax=max(abs(data_matrix.min()), abs(data_matrix.max())),
                interpolation='nearest'
            )
            
            # Add vertical line to separate aging from experiment
            ax.axvline(0.5, color='black', linewidth=2)
            
            # Set column labels
            ax.set_xticks([0, 1])
            ax.set_xlabel('')
            
            # Remove y-axis labels (too many TFs)
            ax.set_yticks([])
            ax.set_ylabel(f'TFs (n={n_tfs})', fontsize=9)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Slope (effect size)', rotation=270, labelpad=15, fontsize=9)
            cbar.ax.tick_params(labelsize=8)
            
            # Title
            ax.set_title(f'{cell_type}', fontsize=10, weight='bold', pad=10)
            
            # Add agreement statistics as text
            same_direction = (np.sign(merged['slope']) == np.sign(merged['slope_condition'])).sum()
            total_with_aging = merged['slope'].notna().sum()
            if total_with_aging > 0:
                agreement_pct = (same_direction / total_with_aging) * 100
                ax.text(
                    0.02, 0.98, 
                    f'Same direction:\n{same_direction}/{total_with_aging} ({agreement_pct:.0f}%)',
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                )
            
            # Save
            name = f'{dataset}_{treatment}_{cell_type}'
            output_path = os.path.join(output_dir, f'aging_experiment_heatmap_{name}.png')
            output_path = output_path.replace(' ', '_').replace('(', '_').replace(')', '_').replace(':', '_')
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
            plt.close()
            print(f"    Saved: {output_path}")


def plot_age_stratified_or_comparison(stats, stats_sig, dataset, analysis_type, target_cell_types, top_aging_tfs, output_dir):
    """
    Plot age-stratified analysis (disease/aging) or aging-perturbation comparison.
    This function handles disease, aging, and perturbation cases.
    """
    print(f"Generating {'age-stratified analysis' if (analysis_type == 'disease' or analysis_type == 'aging') else 'aging vs perturbation comparison'}...")
    
    palette_all = {**palette_trend, **palette_disease_effect, **palette_treatment}
    
    aging_stats_sig = retrieve_sig_stats(type='bulk').drop_duplicates(subset=['cell_type', 'tf'])
    aging_stats_sig = aging_stats_sig[['tf', 'cell_type', 'slope']]
    
    if analysis_type == 'disease' or analysis_type == 'aging':
        _plot_age_stratified_disease(stats, stats_sig, dataset, target_cell_types, top_aging_tfs, aging_stats_sig, palette_all, output_dir)
    else:
        _plot_perturbation_comparison(stats, dataset, target_cell_types, top_aging_tfs, aging_stats_sig, palette_all, output_dir)


def _plot_age_stratified_disease(stats, stats_sig, disease_name, target_cell_types, top_aging_tfs, aging_stats_sig, palette_all, output_dir):
    """Plot age-stratified analysis for disease data."""
    age_groups = stats['age_group'].unique()
    stats['dataset'] = stats['dataset'].apply(lambda name: surrogate_names.get(name, name))
    
    # Map age_group names for better display
    age_group_display_map = {
        'Both age groups': 'SoundLife cohort' if disease_name == 'soundlife' else 'Both age groups'
    }
    
    for cell_type in target_cell_types:
        print(f"  Processing cell type: {cell_type}")
        
        # --- Aging TFs ---
        aging_df = aging_stats_sig[aging_stats_sig['cell_type'] == cell_type]
        net = retrieve_net_consensus(datasets_all, cell_type)
        c_df = net.groupby('source').size().reset_index(name='degree')
        aging_df = aging_df.merge(c_df, left_on='tf', right_on='source', how='left')[['tf', 'slope', 'degree']]
        aging_df['degree'] = aging_df['degree'].div(aging_df['degree'].max())  # Normalize degree
        aging_df['trend'] = ['Increase in aging' if x > 0 else 'Decrease in aging' for x in aging_df['slope']]
        aging_df['analysis'] = 'Age-associated'
        

        # --- Disease/Aging Condition TFs ---
        stats_store = []
        for age_group in age_groups:
            stats_d = stats[
                (stats['age_group'] == age_group) &
                (stats['cell_type'] == cell_type) &
                (~stats['slope_condition'].isna())
            ].copy()
            stats_d = stats_d[['tf', 'cell_type', 'slope_condition', 'p_value_adj']].drop_duplicates()
            # Map age group name for display
            display_name = age_group_display_map.get(age_group, age_group)
            stats_d['analysis'] = display_name
            # For soundlife aging analysis, use aging trend labels; otherwise use disease labels
            if disease_name == 'soundlife':
                stats_d['trend'] = ['Increase in aging' if x > 0 else 'Decrease in aging' for x in stats_d['slope_condition']]
            else:
                stats_d['trend'] = ['Increase in disease' if x > 0 else 'Decrease in disease' for x in stats_d['slope_condition']]
            stats_store.append(stats_d)
        
        stats_condition = pd.concat(stats_store)
        
        # Use display names for categorical ordering
        display_age_groups = [age_group_display_map.get(ag, ag) for ag in age_groups]
        stats_condition['analysis'] = stats_condition['analysis'].astype(
            pd.CategoricalDtype(categories=display_age_groups, ordered=True)
        )
        
        # --- Restrict to overlapping TFs ---
        common_tfs = set(aging_df['tf']) & set(stats_condition['tf'])
        aging_df = aging_df[aging_df['tf'].isin(common_tfs)]
        
        # Order TFs by degree
        top_tfs = aging_df.sort_values('degree', ascending=False).head(top_aging_tfs)['tf'].unique()
        tf_order = list(top_tfs)
        
        # --- Combine and filter ---
        aging_df = aging_df[aging_df['tf'].isin(top_tfs)]
        stats_combined = pd.concat([aging_df, stats_condition])
        df = stats_combined[stats_combined['tf'].isin(top_tfs)].copy()
        
        # Set TF categorical order
        df['tf'] = pd.Categorical(df['tf'], categories=tf_order, ordered=True)
        
        plot_analysis_and_centrality(
            df, 
            all_groups=display_age_groups,  # Use display names instead of original age_groups
            figsize=(2.1, 4), 
            palette_all=palette_all, 
            plot_centrality=False, 
            show_legend=False
        )
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'central_aging_tfs_overlap_{cell_type}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
        plt.close()
        print(f"    Saved: {output_path}")


def _plot_perturbation_comparison(stats, dataset, target_cell_types, top_aging_tfs, aging_stats_sig, palette_all, output_dir):
    """Plot aging vs perturbation comparison."""
    stats_df = stats.copy()
    
    cfg = get_config(dataset)[0]
    target_treatments = cfg.target_treatments
    
    if target_treatments is None:
        target_treatments = stats_df['condition'].unique()
    
    for cell_type in target_cell_types:
        print(f"  Processing cell type: {cell_type}")
        
        # Aging TFs
        stats_aging = aging_stats_sig[aging_stats_sig['cell_type'] == cell_type].copy()
        stats_aging['analysis'] = 'Age-associated'
        stats_aging['trend'] = ['Increase in aging' if x > 0 else 'Decrease in aging' for x in stats_aging['slope']]
        
        # Perturbation TFs
        comparisons = target_treatments
        stats_store = []
        for comparison in comparisons:
            stats_d = stats_df[
                (stats_df['condition'] == comparison) & 
                (stats_df['cell_type'] == cell_type)
            ].copy()
            stats_d['is_significant'] = stats_d['p_value_adj'] < 0.05
            stats_d['analysis'] = comparison
            stats_d['trend'] = [
                'Increase after treatment' if x > 0 else 'Decrease after treatment' 
                for x in stats_d['slope_condition']
            ]
            stats_store.append(stats_d)
        
        if not stats_store:
            print(f"    Warning: No data for {cell_type}")
            continue
            
        stats_condition = pd.concat(stats_store)
        
        # Subset to overlapping TFs
        drug_tfs = stats_condition['tf'].unique()
        aging_tfs = stats_aging['tf'].unique()
        common_tfs = np.intersect1d(drug_tfs, aging_tfs)
        
        if len(common_tfs) == 0:
            print(f"    Warning: No common TFs for {cell_type}")
            continue
        
        # Merge
        df = pd.concat([stats_aging, stats_condition])
        df = df[df['tf'].isin(common_tfs)]
        
        df['analysis'] = pd.Categorical(
            df['analysis'], 
            categories=['Age-associated'] + list(comparisons), 
            ordered=True
        )
        
        # Add centrality information
        net = retrieve_net_consensus(datasets_all, cell_type)
        c_df = net.groupby('source').size().reset_index(name='degree')
        df = df.merge(c_df, left_on='tf', right_on='source', how='left')
        
        top_tfs = df.drop_duplicates(subset=['cell_type', 'tf']).sort_values(
            'degree', ascending=False
        ).head(top_aging_tfs)['tf'].unique()
        df = df[df['tf'].isin(top_tfs)].sort_values('degree', ascending=False)
        df['degree'] = df['degree'].div(df['degree'].max())  # Normalize degree
        
        # Filter palette to only include trends that exist in the data
        available_trends = df['trend'].unique() if 'trend' in df.columns else []
        filtered_palette = {k: v for k, v in palette_all.items() if k in available_trends}
        
        plot_analysis_and_centrality(
            df, 
            all_groups=comparisons, 
            palette_all=filtered_palette if filtered_palette else palette_all, 
            figsize=(2.5, 4), 
            ax2_margins={'x': 0.2, 'y': 0.02}, 
            hide_ylabels=False, 
            plot_centrality=True, 
            show_legend=True
        )
        plt.suptitle(f'{cell_type}', y=.98, weight='bold')
        plt.tight_layout()
        plt.title('')
        
        output_path = os.path.join(output_dir, f'aging_condition_trend_{cell_type}_{dataset}.png')
        output_path = output_path.replace(' ', '_').replace('(', '_').replace(')', '_').replace(':', '_')
        plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
        plt.close()
        print(f"    Saved: {output_path}")


def plot_case_studies(dataset, data_type, analysis_type, case_tfs, case_cell_type, target_cell_types, output_dir):
    """
    Plot case studies: healthy vs disease trends (disease/aging) or donor-level effects (perturbation).
    """
    print(f"Generating {'case TF trends' if (analysis_type == 'disease' or analysis_type == 'aging') else 'donor-level perturbation plots'}...")
    
    if analysis_type == 'disease' or analysis_type == 'aging':
        _plot_disease_case_tfs(dataset, data_type, case_tfs, case_cell_type, output_dir)
    else:
        _plot_perturbation_donor_level(dataset, target_cell_types, output_dir)


def _plot_disease_case_tfs(dataset, data_type, case_tfs, cell_type, output_dir):
    """Plot healthy vs disease trends for specific TFs."""
    # Determine condition column based on dataset
    if dataset == 'SLE_European':
        condition_col = 'condition'
    elif dataset == 'soundlife':
        condition_col = 'condition'  # Will use 'CMV-' vs 'CMV+' or aging groups
    else:
        condition_col = 'Max_WHO_Group'
    
    for i, case_tf in enumerate(case_tfs):
        fig, ax = plt.subplots(1, 1, figsize=(2, .6), sharey=False, sharex=False)
        
        plot_healthy_disease_trend(
            dataset=dataset, 
            data_type=data_type, 
            cell_type=cell_type, 
            case_tf=case_tf, 
            condition_col=condition_col, 
            ax=ax
        )
        
        ax.set_title(f'{case_tf}', fontsize=10, pad=10, weight='bold')
        if i == 0:
            ax.set_xlabel('')
        
        output_path = os.path.join(output_dir, f'healthy_disease_trend_{case_tf}_{cell_type}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
        plt.close()
        print(f"  Saved: {output_path}")


def _plot_perturbation_donor_level(dataset, target_cell_types, output_dir):
    """Plot donor-level perturbation effects for case TFs."""
    cfg = get_config(dataset)[0]
    stats_path = f'{SAVE_DIR}/stats/stats_{dataset}_*_tf_activity_{cfg.test_type}.csv'
    
    # Try to load stats
    import glob
    matching_files = glob.glob(stats_path.replace('*', 'sc'))
    if not matching_files:
        matching_files = glob.glob(stats_path.replace('*', 'bulk'))
    
    if not matching_files:
        print(f"  Warning: No stats files found, skipping donor-level plots")
        return
    
    stats = pd.read_csv(matching_files[0])
    
    # Dataset-specific configurations
    if dataset == 'CXCL9':
        comparisons = ['Ruxolitinib (ctr: RPMI)', 'Ruxolitinib (ctr: LPS)']
        bbox_to_anchor = (1, 1)
    elif dataset == 'op':
        comparisons = ['Ruxolitinib']
        bbox_to_anchor = (1, 1)
    elif dataset == 'parsebioscience':
        comparisons = ['IL-10']
        bbox_to_anchor = (1, 1)
    else:
        print(f"  Warning: Unknown dataset {dataset}, skipping donor-level plots")
        return
    
    for comparison in comparisons:
        for cell_type in target_cell_types:
            # Cell type specific TFs
            if cell_type == 'CD4T':
                case_tfs = ['KLF6', 'PRDM1']
            elif cell_type == 'CD8T':
                case_tfs = ['LEF1', 'ZEB2']
            else:
                print(f"    Warning: No case TFs defined for {cell_type}")
                continue
            
            fig, axes = plt.subplots(1, 2, figsize=(2.5, 1.2), sharex=False, sharey=False)
            
            for i, case_tf in enumerate(case_tfs):
                stats_case = stats[
                    (stats['condition'] == comparison) & 
                    (stats['tf'] == case_tf) & 
                    (stats['cell_type'] == cell_type)
                ].copy()
                
                if len(stats_case) == 0:
                    print(f"    Warning: No data for {case_tf} in {cell_type}")
                    continue
                
                ax = axes[i]
                
                # Map comparison to control and treatment labels
                if comparison == 'Ruxolitinib (ctr: LPS)':
                    ctr = '24 h LPS'
                    treatment = '24 h LPS + ruxolitinib'
                elif comparison == 'Ruxolitinib (ctr: RPMI)':
                    ctr = '24 h RPMI'
                    treatment = '24 h RPMI + ruxolitinib'
                elif comparison == 'Ruxolitinib':
                    ctr = 'Dimethyl Sulfoxide'
                    treatment = 'Ruxolitinib'
                elif comparison == 'IL-10':
                    ctr = 'PBS'
                    treatment = 'IL-10'
                else:
                    print(f"    Warning: Unknown comparison {comparison}")
                    continue
                
                p_value_adj = stats_case['p_value_adj'].values[0]
                plot_donor_level_perturbation_effect(
                    case_tf, ctr, treatment, cell_type, p_value_adj, 
                    dataset=dataset, ax=ax, bbox_to_anchor=bbox_to_anchor
                )
                
                if i == 0:
                    ax.get_legend().remove()
                else:
                    ax.set_ylabel('', fontsize=10)
                    ax.spines[['left']].set_visible(False)
            
            output_path = os.path.join(
                output_dir, 
                f'donor_level_perturbation_effect_{comparison}_{cell_type}.png'
            )
            output_path = output_path.replace(' ', '_').replace('(', '_').replace(')', '_').replace(':', '_')
            plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
            plt.close()
            print(f"    Saved: {output_path}")


def plot_pathway_analysis(stats, dataset, data_type, feature_type, analysis_type, pathway_cell_types, output_dir):
    """Perform pathway analysis comparing aging and condition (disease or perturbation)."""
    print("Generating pathway analysis...")
    
    cfg = get_config(dataset)[0]
    
    # Determine feature column based on feature type
    feature_col = 'target' if feature_type == 'gene_expression' else 'tf'
    
    # Get aging stats with the same feature type
    # Note: Aging stats are always 'bulk' type regardless of condition data type
    aging_stats_sig = retrieve_sig_stats(
        type='bulk',  # Always use bulk for aging stats
        race="both", 
        feature_type=feature_type
    ).drop_duplicates(subset=["cell_type", feature_col])
    aging_stats_sig = aging_stats_sig[[feature_col, "cell_type", "slope", 'p_value_adj']]
    
    # Filter significant condition stats
    stats_sig = stats[(stats['p_value_adj'] < 0.05)].copy()
    
    if analysis_type == 'disease' or analysis_type == 'aging':
        # For disease/aging, filter to younger age group
        stats_sig = stats_sig[
            (~stats_sig['slope_condition'].isna()) & 
            (stats_sig['age_group'] == 'Younger than 50')
        ]
    else:
        # For perturbation, just filter non-null slopes
        stats_sig = stats_sig[~stats_sig['slope_condition'].isna()]
    
    if len(stats_sig) == 0:
        print("  Warning: No significant data for pathway analysis")
        return
    
    # Pathway enrichment with appropriate feature column
    res_aging = pathway_kde_func(aging_stats_sig, min_genes=10, feature_col=feature_col)
    res_aging_sig = res_aging[res_aging['p_adj'] < 0.05]
    
    stats_sig['slope'] = stats_sig['slope_condition']
    res_condition = pathway_kde_func(stats_sig, min_genes=10, feature_col=feature_col)
    res_condition_sig = res_condition[res_condition['p_adj'] < 0.05]
    
    sets = np.concatenate([res_aging_sig['gene_set'].unique(), res_condition_sig['gene_set'].unique()])
    
    if len(sets) == 0:
        print("  No significant pathways found")
        return
    
    plot_pathway_kde(
        df_aging=aging_stats_sig,
        res_aging=res_aging,
        df_condition=stats_sig,
        res_cond=res_condition,
        cell_types=pathway_cell_types,
        sets=sets,
        min_genes=10,
        max_height=0.1,
        row_spacing=0.4,
        cell_spacing=1.5,
        feature_col=feature_col,
    )
    
    output_path = os.path.join(output_dir, f'{dataset}_aging_pathway.png')
    plt.savefig(output_path, bbox_inches="tight", dpi=300, transparent=True)
    plt.close()
    print(f"  Saved: {output_path}")
    
    # Additional GSEA for perturbations
    if analysis_type == 'perturbation':
        _plot_perturbation_pathway_gsea(stats_sig, dataset, output_dir, feature_col)


def _plot_perturbation_pathway_gsea(stats_sig, dataset, output_dir, feature_col):
    """Generate GSEA pathway analysis for perturbations."""
    from ciim.src.pathway_analysis.util import gsea_func
    from ciim.src.feature_association.plots import dotplot_category_color
    
    print("\n  Running GSEA enrichment analysis...")
    try:
        pathway_scores = gsea_func(
            stats_sig,
            pvalue_col='p_value_adj',
            gene_sets=['MSigDB_Hallmark_2020'],
            feature_col=feature_col
        )
        
        if pathway_scores is not None and len(pathway_scores) > 0:
            print(f"    Found {len(pathway_scores)} significant pathway enrichments")
            
            # Plot GSEA dotplot
            n_terms = pathway_scores['Term'].nunique()
            cell_types = pathway_scores['cell_type'].unique()
            
            fig, ax = plt.subplots(
                1, 1, 
                figsize=(len(cell_types) * 0.12 + 1, 1 + 0.15 * n_terms), 
                sharey=True, 
                sharex=True
            )
            
            pathway_scores['cell_type'] = pd.Categorical(
                pathway_scores['cell_type'], 
                categories=cell_types, 
                ordered=True
            )
            
            dotplot_category_color(
                pathway_scores,
                ax,
                color_col='trend',
                size_col='neg_log10_adj_pval',
                x='cell_type',
                y='Term',
                palette=palette_treatment,
                sizes=(50, 250),
                show_color_legend=True,
                show_size_legend=True,
                size_legend_title='Significance',
                color_legend_title='Pathway activity',
                size_legend_loc=(1.2, 0.35),
                color_legend_loc=(1.02, 0.75),
                y_label='',
                alpha=0.5
            )
            
            output_path = os.path.join(output_dir, f'{dataset}_pathway_gsea.png')
            plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
            plt.close()
            print(f"    Saved GSEA plot: {output_path}")
        else:
            print("    No significant pathway enrichments found")
    
    except Exception as e:
        print(f"    Warning: GSEA analysis failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Unified Condition Analysis - Post-run visualization for disease and perturbation experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Disease analysis
  python post_condition_analysis.py --dataset SLE_European --analysis-type disease --data-type bulk --feature-type tf_activity
  
  # Perturbation analysis
  python post_condition_analysis.py --dataset CXCL9 --analysis-type perturbation --data-type sc --feature-type tf_activity
  
  # Soundlife aging analysis (CMV-negative cohort)
  python post_condition_analysis.py --dataset soundlife --config-label aging_cmv_neg --analysis-type aging --data-type bulk --feature-type tf_activity
  
  # Soundlife CMV disease analysis (combines cmv_young and cmv_old)
  python post_condition_analysis.py --dataset soundlife --config-label cmv --analysis-type disease --data-type bulk --feature-type tf_activity
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
        '--data-type',
        type=str,
        default='bulk',
        help='Data type: bulk or sc (default: bulk)'
    )
    parser.add_argument(
        '--feature-type',
        type=str,
        default='tf_activity',
        help='Feature type: tf_activity or gene_expression (default: tf_activity)'
    )
    parser.add_argument(
        '--config-label',
        type=str,
        default=None,
        help='Configuration label for datasets with multiple configs (e.g., aging_cmv_neg, cmv for soundlife). Use "cmv" to combine cmv_young and cmv_old.'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for plots (default: PLOTS_DIR from common.py)'
    )
    parser.add_argument(
        '--cell-types',
        type=str,
        nargs='+',
        default=['CD4T', 'CD8T'],
        help='Cell types to analyze (default: CD4T CD8T)'
    )
    parser.add_argument(
        '--case-tfs',
        type=str,
        nargs='+',
        default=['LEF1', 'ZEB2'],
        help='Case TFs for trend plotting (disease only, default: LEF1 ZEB2)'
    )
    parser.add_argument(
        '--case-cell-type',
        type=str,
        default='CD8T',
        help='Cell type for case TF analysis (disease only, default: CD8T)'
    )
    parser.add_argument(
        '--top-aging-tfs',
        type=int,
        default=15,
        help='Number of top aging TFs to show (default: 15)'
    )
    parser.add_argument(
        '--agreement',
        type=str,
        default='same',
        choices=['same', 'opposite'],
        help='Agreement type for overlap analysis (default: same)'
    )
    parser.add_argument(
        '--sig-threshold',
        type=float,
        default=0.05,
        help='Significance threshold for p-values (default: 0.05)'
    )
    parser.add_argument(
        '--skip-pathway',
        action='store_true',
        help='Skip pathway analysis'
    )
    parser.add_argument(
        '--skip-case-studies',
        action='store_true',
        help='Skip case studies (TF trends for disease or donor-level for perturbations)'
    )
    parser.add_argument(
        '--skip-heatmap',
        action='store_true',
        help='Skip aging-experiment heatmap (perturbation only)'
    )
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else PLOTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print(f"Unified Condition Analysis - {args.analysis_type.capitalize()}")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    if args.config_label:
        print(f"Config label: {args.config_label}")
    print(f"Analysis type: {args.analysis_type}")
    print(f"Data type: {args.data_type}")
    print(f"Feature type: {args.feature_type}")
    print(f"Output directory: {output_dir}")
    print(f"Cell types: {', '.join(args.cell_types)}")
    print(f"Significance threshold: {args.sig_threshold}")
    print("="*60)
    
    # Load statistics
    print("\nLoading statistics...")
    stats, cfg = load_stats(args.dataset, args.data_type, args.feature_type, args.config_label)
    
    # For soundlife with config_label, filter to the specific config
    # Skip filtering if config_label is 'cmv' (already combined in load_stats)
    if args.config_label and args.dataset == 'soundlife' and 'config_label' in stats.columns and args.config_label != 'cmv':
        print(f"Filtering stats to config_label: {args.config_label}")
        stats = stats[stats['config_label'] == args.config_label].copy()
        print(f"  Filtered to {len(stats)} rows")
    
    # Standardize column names - ensure slope_condition exists
    if 'slope' in stats.columns and 'slope_condition' not in stats.columns:
        stats['slope_condition'] = stats['slope']
    
    # Filter based on analysis type
    if args.analysis_type == 'disease' or args.analysis_type == 'aging':
        stats = stats[~stats['slope_condition'].isna()].copy()
        stats['major_cell_type'] = stats['cell_type'].apply(
            lambda x: mapping_minor_2_major.get(x, x)
        )
    else:
        stats['cell_type'] = pd.Categorical(stats['cell_type'], categories=cell_types, ordered=True)
        stats['trend'] = [
            'Increase after treatment' if x > 0 else 'Decrease after treatment' 
            for x in stats['slope_condition']
        ]
    
    stats_sig = stats[stats['p_value_adj'] < args.sig_threshold].copy()
    
    print(f"Total significant TFs per cell type:")
    if len(stats_sig) > 0:
        print(stats_sig.groupby(['cell_type'])['tf'].nunique())
    else:
        print("  No significant TFs found!")
    print()
    
    # 1. Overview heatmap
    plot_overview_heatmap(stats_sig if (args.analysis_type == 'disease' or args.analysis_type == 'aging') else stats, 
                         args.dataset, args.analysis_type, output_dir, args.config_label)
    
    # 2. Aging overlap
    if len(stats_sig) > 0:
        plot_aging_overlap(
            stats_sig, 
            args.dataset,
            args.cell_types, 
            args.analysis_type,
            args.agreement,
            output_dir
        )
    
    # 3. Aging-experiment heatmap
    if not args.skip_heatmap and len(stats_sig) > 0:
        if args.analysis_type == 'perturbation':
            plot_aging_experiment_heatmap(stats_sig, args.dataset, output_dir)
        elif args.analysis_type == 'disease' or args.analysis_type == 'aging':
            # For disease/aging, also generate the overlap heatmap
            plot_aging_disease_overlap_heatmap(stats_sig, args.dataset, args.cell_types, output_dir)
    
    # 4. Age-stratified or comparison analysis
    plot_age_stratified_or_comparison(
        stats, 
        stats_sig, 
        args.dataset,
        args.analysis_type,
        args.cell_types,
        args.top_aging_tfs,
        output_dir
    )
    
    # 5. Case studies (TF trends or donor-level)
    if not args.skip_case_studies:
        plot_case_studies(
            args.dataset,
            args.data_type,
            args.analysis_type,
            args.case_tfs,
            args.case_cell_type,
            args.cell_types,
            output_dir
        )
    
    # 6. Pathway analysis (optional)
    if not args.skip_pathway:
        plot_pathway_analysis(
            stats,
            args.dataset,
            args.data_type,
            args.feature_type,
            args.analysis_type,
            args.cell_types,
            output_dir
        )
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"All plots saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
