
#!/usr/bin/env python
"""
Unified Condition Analysis - Post-run visualization script
This script performs post-run analysis for both disease and perturbation experiments,
generating various visualizations including:
- Overview heatmap of cell types
- Overlap analysis with aging genes
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
import seaborn as sns
from pandas.api.types import CategoricalDtype
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from hiara.src.config import surrogate_names
from hiara import retrieve_feature_data, retrieve_sig_stats, retrieve_features_stats
from hiara.src.feature_association.plots import heatplot_age_trend
# Import common utilities and configuration
from hiara.src.config import (
    PLOTS_DIR, 
    OUTPUT_DIR,
    CELL_TYPES, 
    DISCOVERY_COHORTS,
    palette_trend,
    palette_disease_effect,
    palette_treatment,
    surrogate_names
)
from hiara.src.config import get_config
from hiara.src.feature_association.plots import (
    heamap_plot_minor_cell_types,
    plot_overlap,
    plot_tf_act_central_tfs,
)
from hiara.src.utils.util import retrieve_net_consensus
from hiara.src.pathway_analysis.util import pathway_kde_func
from hiara.src.pathway_analysis.plots import plot_pathway_kde
warnings.filterwarnings("ignore")
# Set matplotlib defaults
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "Arial"


def format_tf_activity_for_disease_trend_plot(dataset, data_type, cell_type, tf, age_limit=[20, 75], condition_col='disease'):
    """
    Format TF activity data for disease trend plotting.
    
    Parameters
    ----------
    dataset : str
        Dataset name
    type : str
        Data type (e.g., 'bulk', 'sc')
    cell_type : str
        Cell type to analyze
    tf : str
        Transcription factor name
    age_limit : list, optional
        Age range [min, max], default [20, 75]
    condition_col : str, optional
        Column name for condition (e.g., 'disease', 'Max_WHO_Group'), default 'disease'
    
    Returns
    -------
    pd.DataFrame or None
        Binned dataframe with disease/condition as rows, age bins as columns
    """
    tf_acts = retrieve_feature_data(dataset=dataset, cell_type=cell_type, data_type=data_type, condition=None)
    tf_acts = tf_acts[(tf_acts.obs['age'] >= age_limit[0]) & (tf_acts.obs['age'] <= age_limit[1])]
    tf_acts = tf_acts[:, tf_acts.var_names == tf]

    
    if tf_acts.shape[1] == 0:
        return None

    expr = tf_acts.to_df()
    expr = expr.merge(tf_acts.obs[['age', condition_col]], left_index=True, right_index=True, how='left')
    expr[condition_col] = expr[condition_col].astype('category')
    expr['age'] = expr['age'].astype(int)

    # Pivot table: rows = disease, columns = age, values = expression
    expr_table = expr.pivot_table(index=condition_col, columns='age', values=tf)

    # Bin ages into 5-year intervals
    df = expr_table.copy()
    age_columns = df.columns
    age_bins = defaultdict(list)

    for col in age_columns:
        bin_start = (col // 5) * 5
        age_bins[bin_start].append(col)

    # Average across bins
    binned_means = {bin_start: df[bin_ages].mean(axis=1) for bin_start, bin_ages in age_bins.items()}
    binned_df = pd.DataFrame(binned_means)

    # Sort bins by age
    binned_df = binned_df[sorted(binned_df.columns)]
    
    return binned_df


def plot_healthy_disease_trend(dataset, data_type, cell_type, case_tf, condition_col, ax=None, normalize=False):
    """
    Plot healthy vs disease trend for a TF across age.
    
    Parameters
    ----------
    dataset : str
        Dataset name
    type : str
        Data type (e.g., 'bulk', 'sc')
    cell_type : str
        Cell type to analyze
    case_tf : str
        Transcription factor name
    condition_col : str
        Column name for condition (e.g., 'disease', 'Max_WHO_Group')
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, creates new if None
    normalize : bool, optional
        Whether to normalize values, default False
    
    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot
    """
    binned_df = format_tf_activity_for_disease_trend_plot(
        dataset=dataset, data_type=data_type, cell_type=cell_type, tf=case_tf, condition_col=condition_col
    )
    
    if binned_df is None:
        return None
    
    binned_df.index = [surrogate_names.get(name, name) for name in binned_df.index]
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(2.5, .5), sharey=False, sharex=True)
    
    if normalize:
        binned_df = binned_df.sub(binned_df.min(axis=1), axis=0)
        binned_df = binned_df.div(binned_df.abs().max(axis=1), axis=0)
    
    heatplot_age_trend(
        binned_df, 
        cmap='magma', 
        cbar_title="TF activity\n(normalized)" if normalize else "TF activity", 
        y_label="Condition", 
        ax=ax, 
        show_cbar=True,
        cbar_kws={
            "shrink": 1.2,
            "aspect": 3,
            "fraction": 0.1
        }
    )
    ax.set_xlabel('Age')
    
    return ax


def get_condition_palette(analysis_type):
    """Get appropriate color palette based on analysis type."""
    if analysis_type == 'disease':
        return palette_disease_effect
    elif analysis_type == 'perturbation':
        return palette_treatment
    else:
        return palette_disease_effect  # Default

def plot_overview_heatmap(stats, args):
    """Generate overview heatmap of minor cell types."""
    print("Generating overview heatmap...")
    
    dataset = args.dataset
    analysis_type = args.analysis_type
    output_dir = args.output_dir
    config_label = args.config_label
    slope_col = 'slope'  # Could be parameterized if needed
    
    palette = get_condition_palette(analysis_type)
    
    if analysis_type == 'disease' or analysis_type == 'aging':
        # Special handling for combined CMV (has young and old separately)
        
        if 'age_group' in stats.columns:
            unique_age_groups = stats['age_group'].unique()
            if 'Both age groups' in unique_age_groups:
                stats_filtered = stats[stats['age_group'] == 'Both age groups'].copy()
            else:
                # Data already filtered by config (e.g., cmv_young has age_group='young')
                stats_filtered = stats.copy()
        else:
            # No age_group column (e.g., SLE_European), use data as-is
            stats_filtered = stats.copy()
        _plot_single_heatmap(stats_filtered, palette, output_dir, 
                            suffix=dataset if (analysis_type == 'disease' or analysis_type == 'aging') else None, slope_col=slope_col)
    else:
        # For perturbations, process each condition separately
        cfg = get_config(dataset)
        target_treatments = cfg.target_treatments
        
        if target_treatments is None:
            print("  Warning: No target treatments configured, using all conditions")
            target_treatments = stats['condition'].unique()
        
        for condition in target_treatments:
            stats_filtered = stats[stats['condition'] == condition].copy()
            _plot_single_heatmap(stats_filtered, palette, output_dir, suffix=condition, slope_col=slope_col)
        return

def _plot_single_heatmap(stats, palette, output_dir, suffix=None, slope_col='slope'):
    """Helper function to plot a single heatmap."""
    stats['cell_type'] = pd.Categorical(stats['cell_type'], categories=CELL_TYPES, ordered=True)
    stats['major_cell_type'] = stats['cell_type'].astype(
        CategoricalDtype(categories=['CD4T', 'CD8T', 'NK', 'MONO', 'B'], ordered=True)
    )
    
    heamap_plot_minor_cell_types(
        stats, 
        slope_col=slope_col, 
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

def plot_aging_overlap(stats_sig, args):
    """Plot overlap between condition (disease/perturbation) and aging genes."""
    print("Generating aging overlap plot...")
    
    dataset = args.dataset
    included_cell_types = args.cell_types
    analysis_type = args.analysis_type
    agreement = args.agreement
    output_dir = args.output_dir
    feature_type = args.feature_type
    
    cfg = get_config(dataset)
    aging_stats_sig = retrieve_sig_stats(data_type='bulk', feature_type=feature_type).drop_duplicates(subset=['cell_type', 'gene'])
    aging_stats_sig = aging_stats_sig[['gene', 'cell_type', 'slope']].copy()

    comparisons = stats_sig['comparison'].unique()
    
    print('Stats of comparison:', stats_sig.groupby(['cell_type', 'comparison'])['gene'].nunique())
    for cell_type in included_cell_types:
        print(f"  Processing cell type: {cell_type}")
        for comparison in comparisons:
            # Filter stats
            stats_sig_sub = stats_sig[(stats_sig['comparison'] == comparison) & (stats_sig['cell_type'] == cell_type)].reset_index(drop=True)
            
            if len(stats_sig_sub) == 0:
                print(f"    Warning: No data for {'comparison ' + comparison + ' -- ' + cell_type}")
                continue
            
            # Rename slope to slope_condition to avoid conflicts when merging
            stats_sig_sub_renamed = stats_sig_sub[['gene', 'slope']].rename(columns={'slope': 'slope_condition'})
            
            # Get aging data for this cell type (keep cell_type column for plot_overlap)
            aging_stats_sig_ct = aging_stats_sig[aging_stats_sig['cell_type'] == cell_type][['gene', 'cell_type', 'slope']].copy()
            
            # Merge aging slopes with condition slopes
            merged = aging_stats_sig_ct[['gene', 'slope']].merge(
                stats_sig_sub_renamed, 
                on='gene', 
                how='inner'
            )
              
            if len(merged) > 0:
                same_direction = (np.sign(merged['slope']) == np.sign(merged['slope_condition'])).sum()
                opposite_direction = (np.sign(merged['slope']) != np.sign(merged['slope_condition'])).sum()
                total_overlap = len(merged)
                
                print(f"{comparison} Aging genes {len(aging_stats_sig_ct)}, Condition sig genes {len(stats_sig_sub)}, Overlap {total_overlap}")
                print(f"      Same direction: {same_direction} ({same_direction/total_overlap*100:.1f}%)")
                print(f"      Opposite direction: {opposite_direction} ({opposite_direction/total_overlap*100:.1f}%)")

            # Prepare data for plot_overlap (rename slope to slope_condition for consistency with plot function)
            stats_sig_sub_for_plot = stats_sig_sub[['gene', 'cell_type', 'slope']].reset_index(drop=True).rename(columns={'slope': 'slope_condition'})

            plot_overlap(
                stats_sig_sub_for_plot,  # RIGHT side (Sound Life sig genes)
                aging_stats_sig_ct.reset_index(drop=True),  # LEFT side (Reference aging genes - baseline)
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
            comparison = comparison.replace('(', '_').replace(')', '_').replace(':', '_').replace(' ', '_')
            name_suffix = f'{dataset}_{comparison}_{cell_type}'
            output_path = os.path.join(output_dir, f'condition_aging_overlap_{name_suffix}.png')
            output_path = output_path.replace(' ', '_').replace('(', '_').replace(')', '_').replace(':', '_')
            plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
            plt.close()
            print(f"    Saved: {output_path}")

def plot_directional_consistency_scatter(stats_sig, args):
    """
    Generate directional consistency scatter plots comparing Sound Life vs Reference aging genes.
    Shows signed -log10(p-values) with direction concordance.
    """
    print("Generating directional consistency scatter plots...")
    
    feature_type = args.feature_type
    dataset = args.dataset
    included_cell_types = args.cell_types
    output_dir = args.output_dir
    
    import seaborn as sns
    
    # Load reference aging genes
    aging_stats_sig = retrieve_sig_stats(data_type='bulk', feature_type=feature_type, filter_inconsistent=True)
    aging_stats_sig = aging_stats_sig.drop_duplicates(subset=['cell_type', 'gene'])
    
    # Prepare data
    sl_sig = (
        stats_sig[stats_sig['p_value_adj'] < 0.05]
        .sort_values("p_value_adj")
        .drop_duplicates(["cell_type", 'gene'], keep="first")
    )
    
    ref_aging_unique = (
        aging_stats_sig
        .sort_values("p_value_adj")
        .drop_duplicates(["cell_type", 'gene'], keep="first")
    )
    
    sns.set_style('whitegrid')
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    
    for cell_type in included_cell_types:
        if cell_type not in sl_sig['cell_type'].unique():
            print(f"  Warning: No significant genes for {cell_type}")
            continue
            
        print(f"  Processing {cell_type}")
        
        sl_ct = sl_sig[sl_sig['cell_type'] == cell_type].copy()
        ref_ct = ref_aging_unique[ref_aging_unique['cell_type'] == cell_type].copy()
        
        print(f"    Sound Life significant genes: {len(sl_ct)}")
        print(f"    Reference aging genes: {len(ref_ct)}")
        
        # Rename slope to slope_condition in sl_ct to avoid conflicts
        sl_ct_renamed = sl_ct[['gene', 'slope', 'p_value_adj']].rename(columns={'slope': 'slope_condition'})
        
        # Inner merge to get overlap
        merged = sl_ct_renamed.merge(
            ref_ct[['gene', 'slope', 'meta_p_adj']],
            on='gene',
            how='inner'
        )
        
        if len(merged) < 10:
            print(f"    Overlap too small (n={len(merged)}), skipping")
            continue
        
        print(f"    Overlap genes: {len(merged)}")
        
        # Calculate directional metrics
        merged['same_direction'] = (
            np.sign(merged['slope_condition']) == np.sign(merged['slope'])
        )
        
        merged['neg_log_p_sl'] = -np.log10(merged['p_value_adj']) * np.sign(merged['slope_condition'])
        merged['neg_log_p_ref'] = -np.log10(merged['meta_p_adj']) * np.sign(merged['slope'])
        
        same_dir = merged[merged['same_direction']]
        opp_dir = merged[~merged['same_direction']]
        
        print(f"    Same direction: {len(same_dir)} ({len(same_dir)/len(merged)*100:.1f}%)")
        print(f"    Opposite direction: {len(opp_dir)} ({len(opp_dir)/len(merged)*100:.1f}%)")
        
        # Create scatter plot
        fig = plt.figure(figsize=(3, 3))
        ax_main = plt.subplot(1, 1, 1)
        s = 20
        
        if len(opp_dir) > 0:
            ax_main.scatter(
                opp_dir['neg_log_p_sl'],
                opp_dir['neg_log_p_ref'],
                c='red',
                s=s,
                alpha=0.6,
                label=f'Opposite direction (n={len(opp_dir)})',
                edgecolors='darkred',
                linewidths=.1
            )
        
        if len(same_dir) > 0:
            ax_main.scatter(
                same_dir['neg_log_p_sl'],
                same_dir['neg_log_p_ref'],
                c='green',
                s=s,
                alpha=0.6,
                label=f'Same direction (n={len(same_dir)})',
                edgecolors='darkred',
                linewidths=.1
            )
        
        ax_main.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax_main.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        
        ax_main.set_xlabel('Sound Life aging\n-log10(p)', fontsize=10)
        ax_main.set_ylabel('Reference aging\n-log10(p)', fontsize=10)
        
        ax_main.legend(loc=(1.01, 0.5), framealpha=0.9, fontsize=10, frameon=False)
        
        output_path = os.path.join(output_dir, f'directional_consistency_{dataset}_{cell_type}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {output_path}")

def plot_aging_disease_overlap_heatmap(stats_sig, args):
    """
    Plot heatmap comparing TF activity/directions between aging and disease/aging condition.
    (Disease/Aging-specific visualization showing side-by-side slopes)
    Creates separate heatmaps for each cell type with genes on y-axis.
    """
    print("Generating aging vs condition overlap heatmap...")
    
    dataset = args.dataset
    cell_types = args.cell_types
    output_dir = args.output_dir
    feature_type = args.feature_type
    
    import seaborn as sns
    from matplotlib.patches import Rectangle
        
    # Load aging stats
    aging_stats_sig = retrieve_sig_stats(data_type='bulk', feature_type=feature_type).drop_duplicates(subset=["cell_type", 'gene'])
    
    # Process each cell type separately
    for cell_type in cell_types:
        if cell_type not in stats_sig['cell_type'].unique():
            print(f"  Warning: No data for {cell_type}")
            continue
            
        print(f"  Processing cell type: {cell_type}")
        
        # Get condition data for this cell type (rename slope to slope_condition)
        condition_df = stats_sig[
            stats_sig['cell_type'] == cell_type
        ][['gene', 'slope']].copy().rename(columns={'slope': 'slope_condition'})
        
        # Get aging data for same cell type
        aging_df = aging_stats_sig[
            aging_stats_sig['cell_type'] == cell_type
        ][['gene', 'slope']].copy()
        
        # Get union of genes from both condition and aging
        all_tfs = sorted(list(set(condition_df['gene'].tolist() + aging_df['gene'].tolist())))
        
        # Create a dataframe with all genes
        merged = pd.DataFrame({'gene': all_tfs})
        merged = merged.merge(condition_df, on='gene', how='left')
        merged = merged.merge(aging_df, on='gene', how='left')
        
        if len(merged) == 0:
            print(f"    Warning: No genes found for {cell_type}")
            continue
        
        # Sort by aging slope (Age-associated genes column), put NaNs at the end
        merged['sort_key'] = merged['slope'].fillna(999)  # NaNs go to bottom
        merged = merged.sort_values('sort_key', ascending=False).drop(columns=['sort_key'])
        
        n_tfs = len(merged)
        print(f"    Total genes (union): {n_tfs}")
        print(f"      Condition genes: {condition_df['gene'].nunique()}")
        print(f"      Aging genes: {aging_df['gene'].nunique()}")
        
        # Prepare data matrix: rows are genes, columns are [Condition, Aging]
        data_matrix = merged[['slope_condition', 'slope']].values
        
        # Convert to binary: +1 for positive slope, -1 for negative slope
        binary_matrix = np.sign(data_matrix)
        
        # Create heatmap with genes on y-axis
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
        ax.set_xticklabels(['Age-associated\ngenes', 'SoundLife'], fontsize=9, rotation=45, ha='right')
        ax.set_xlabel('')
        
        # Hide y-axis labels (TF names) but show count
        ax.set_yticks([])
        ax.set_ylabel(f'genes (n={n_tfs})', fontsize=9)
        
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

def plot_aging_experiment_heatmap(stats_sig, args):
    """
    Plot heatmap comparing TF activity/directions between aging and experimental conditions.
    (Perturbation-specific visualization)
    """
    print("Generating aging vs experiment heatmap...")
    
    dataset = args.dataset
    output_dir = args.output_dir
    feature_type = args.feature_type
    
    import seaborn as sns
    from matplotlib.patches import Rectangle
        
    # Load aging stats
    aging_stats_sig = retrieve_sig_stats(data_type='bulk', feature_type=feature_type).drop_duplicates(subset=["cell_type", 'gene'])
    
    cfg = get_config(dataset)
    target_treatments = cfg.target_treatments
    
    if target_treatments is None:
        print("  Warning: No target treatments configured")
        return
    
    for cell_type in stats_sig['cell_type'].unique():
        print(f"  Processing cell type: {cell_type}")
        
        for treatment in target_treatments:
            # Get experiment data (rename slope to slope_condition)
            exp_df = stats_sig[
                (stats_sig['condition'] == treatment) & 
                (stats_sig['cell_type'] == cell_type)
            ][['gene', 'slope']].copy().rename(columns={'slope': 'slope_condition'})
            
            # Get aging data for same cell type
            aging_df = aging_stats_sig[
                aging_stats_sig['cell_type'] == cell_type
            ][['gene', 'slope']].copy()
            
            # Merge on genes that are in the experiment (left join)
            merged = exp_df.merge(aging_df, on='gene', how='left')
            
            if len(merged) == 0:
                print(f"    Warning: No data for {treatment} in {cell_type}")
                continue
            
            # Sort by experiment slope for better visualization
            merged = merged.sort_values('slope_condition', ascending=False)
            
            # Prepare data matrix
            data_matrix = merged[['slope_condition', 'slope']].values
            
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
            
            # Remove y-axis labels (too many genes)
            ax.set_yticks([])
            ax.set_ylabel(f'genes (n={n_tfs})', fontsize=9)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Slope (effect size)', rotation=270, labelpad=15, fontsize=9)
            cbar.ax.tick_params(labelsize=8)
            
            # Title
            ax.set_title(f'{cell_type}', fontsize=10, weight='bold', pad=10)
            
            # Add agreement statistics as text
            same_direction = (np.sign(merged['slope_condition']) == np.sign(merged['slope'])).sum()
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

def plot_central_tf_act_disease(stats, args):
    """Plot age-stratified analysis for disease data."""
    aging_stats_sig = retrieve_sig_stats(data_type='bulk', feature_type=args.feature_type).drop_duplicates(subset=['cell_type', 'gene'])
    
    # Check if we have cmv_young and cmv_old configs
    if 'age_group' not in stats.columns:
        print("  Error: 'age_group' column not found in stats for age-stratified analysis")
        return
    age_groups = stats['age_group'].unique()
    group_col = 'age_group'
    
    stats['dataset'] = stats['dataset'].apply(lambda name: surrogate_names.get(name, name))
    
    for cell_type in args.cell_types:
        print(f"  Processing cell type: {cell_type}")
        # --- Aging genes ---
        aging_df = aging_stats_sig[aging_stats_sig['cell_type'] == cell_type]
        net = retrieve_net_consensus(cell_type=cell_type)
        c_df = net.groupby('source').size().reset_index(name='degree')
        aging_df = aging_df.merge(c_df, left_on='gene', right_on='source', how='left')[['gene', 'slope', 'degree']]
        aging_df['degree'] = aging_df['degree'].div(aging_df['degree'].max())  # Normalize degree
        aging_df['trend'] = ['Increase in aging' if x > 0 else 'Decrease in aging' for x in aging_df['slope']]
        aging_df['analysis'] = 'Age-associated'
        
        # --- Disease/Aging Condition genes ---
        stats_store = []
        for age_group in age_groups:
            stats_d = stats[
                (stats[group_col] == age_group) &
                (stats['cell_type'] == cell_type) &
                (~stats['slope'].isna())
            ].copy()
            stats_d = stats_d[['gene', 'cell_type', 'slope', 'p_value_adj']].drop_duplicates()
            # Map age group name for display
            # display_name = age_group_display_map.get(age_group, age_group)
            stats_d['analysis'] = age_group
            # For soundlife aging or CMV analysis, use aging trend labels; otherwise use disease labels
            stats_d['trend'] = ['Increase in disease' if x > 0 else 'Decrease in disease' for x in stats_d['slope']]
            # stats_d['age_group'] = age_group
            stats_store.append(stats_d)
        
        stats_condition = pd.concat(stats_store)
        stats_condition['analysis'] = stats_condition['analysis'].astype(
            pd.CategoricalDtype(categories=age_groups, ordered=True)
        )
        # --- Restrict to overlapping genes ---
        common_tfs = set(aging_df['gene']) & set(stats_condition['gene'])
        aging_df = aging_df[aging_df['gene'].isin(common_tfs)]
        
        # Order genes by degree
        top_aging_tfs = args.top_aging_tfs
        top_tfs = aging_df.sort_values('degree', ascending=False).head(top_aging_tfs)['gene'].unique()
        tf_order = list(top_tfs)
        
        # --- Combine and filter ---
        aging_df = aging_df[aging_df['gene'].isin(top_tfs)]
        stats_combined = pd.concat([aging_df, stats_condition])
        df = stats_combined[stats_combined['gene'].isin(top_tfs)].copy()
        
        # Set TF categorical order
        df['gene'] = pd.Categorical(df['gene'], categories=tf_order, ordered=True)
        palette_all = {**palette_trend, **palette_disease_effect}
        plot_tf_act_central_tfs(
            df, 
            all_groups=age_groups,  # Use display names instead of original age_groups
            figsize=(2.1, 4), 
            palette_all=palette_all, 
            plot_centrality=False, 
            show_legend=False
        )
        
        plt.tight_layout()
        
        output_path = os.path.join(args.output_dir, f'central_tfs_{cell_type}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
        plt.close()
        print(f"    Saved: {output_path}")

def plot_central_tf_act_perturbation(stats, args):
    """Plot aging vs perturbation comparison."""
    stats_df = stats.copy()
    aging_stats_sig = retrieve_sig_stats(data_type='bulk', feature_type=args.feature_type).drop_duplicates(subset=['cell_type', 'gene'])
    
    cfg = get_config(args.dataset)
    
    # Use comparison column if available, otherwise fall back to condition
    comparisons = stats_df[stats_df['cell_type'].isin(args.cell_types)]['comparison'].unique()
    
    for cell_type in args.cell_types:
        print(f"  Processing cell type: {cell_type}")
        
        # Aging genes
        stats_aging = aging_stats_sig[aging_stats_sig['cell_type'] == cell_type].copy()
        stats_aging['analysis'] = 'Age-associated'
        stats_aging['trend'] = ['Increase in aging' if x > 0 else 'Decrease in aging' for x in stats_aging['slope']]
        
        # Perturbation genes - use comparison column if available
        stats_store = []
        for comparison in comparisons:
            stats_d = stats_df[
                (stats_df['comparison'] == comparison) & 
                (stats_df['cell_type'] == cell_type)
            ].copy()
           
            stats_d['is_significant'] = stats_d['p_value_adj'] < 0.05
            stats_d['analysis'] = comparison
            stats_d['trend'] = [
                'Increase after treatment' if x > 0 else 'Decrease after treatment' 
                for x in stats_d['slope']
            ]
            stats_store.append(stats_d)
        
        if not stats_store:
            print(f"    Warning: No data for {cell_type}")
            continue
            
        stats_condition = pd.concat(stats_store)
        
        # Subset to overlapping genes
        drug_tfs = stats_condition['gene'].unique()
        aging_tfs = stats_aging['gene'].unique()
        common_tfs = np.intersect1d(drug_tfs, aging_tfs)
        
        if len(common_tfs) == 0:
            print(f"    Warning: No common genes for {cell_type}")
            continue
        
        # Merge
        df = pd.concat([stats_aging, stats_condition])
        df = df[df['gene'].isin(common_tfs)]
        
        df['analysis'] = pd.Categorical(
            df['analysis'], 
            categories=['Age-associated'] + list(comparisons), 
            ordered=True
        )
        
        # Add centrality information
        net = retrieve_net_consensus(cell_type=cell_type)
        c_df = net.groupby('source').size().reset_index(name='degree')
        df = df.merge(c_df, left_on='gene', right_on='source', how='left')
        
        top_tfs = df.drop_duplicates(subset=['cell_type', 'gene']).sort_values(
            'degree', ascending=False
        ).head(args.top_aging_tfs)['gene'].unique()
        df = df[df['gene'].isin(top_tfs)].sort_values('degree', ascending=False)
        df['degree'] = df['degree'].div(df['degree'].max())  # Normalize degree
        palette_all = {**palette_trend, **palette_treatment}
        plot_tf_act_central_tfs(
            df, 
            all_groups=comparisons, 
            palette_all=palette_all, 
            figsize=(2.5, 4), 
            ax2_margins={'x': 0.2, 'y': 0.02}, 
            hide_ylabels=False, 
            plot_centrality=True, 
            show_legend=True
        )
        plt.suptitle(f'{cell_type}', y=.98, weight='bold')
        plt.tight_layout()
        plt.title('')
        
        output_path = os.path.join(args.output_dir, f'central_tfs_{cell_type}_{args.dataset}.png')
        output_path = output_path.replace(' ', '_').replace('(', '_').replace(')', '_').replace(':', '_')
        plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
        plt.close()
        print(f"    Saved: {output_path}")

def plot_disease_case_tfs(args):
    """Plot healthy vs disease trends for specific genes."""
    # Determine condition column based on dataset
    condition_col = 'condition'
   
    for i, case_tf in enumerate(args.case_tfs):
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

def plot_ctr_condition_donor_level(args):
    """Plot donor-level perturbation effects for case genes."""    
    stats = retrieve_features_stats(
        dataset=args.dataset, data_type=args.data_type, feature_type=args.feature_type, multi_cohort=False)
    comparisons = stats['comparison'].unique()

    bbox_to_anchor = (1.01, 1)
    case_tfs = args.case_tfs
    
    for cell_type in args.cell_types:
        adata = retrieve_feature_data(dataset=args.dataset, data_type=args.data_type, feature_type=args.feature_type, cell_type=cell_type)
        adata_df = adata.to_df()
        adata_df[['donor_id', 'condition']] = adata.obs[['donor_id', 'condition']].values

        for comparison in comparisons:
            fig, axes = plt.subplots(1, len(case_tfs), figsize=(1.5 * len(case_tfs), 1.2), sharex=False, sharey=False)
            for i, case_tf in enumerate(case_tfs):
                ax = axes[i] if len(case_tfs) > 1 else axes
                print(f"    Plotting {case_tf} in {cell_type} for {comparison}...")
                
                stats_case = stats[
                    (stats['comparison'] == comparison) & 
                    (stats['gene'] == case_tf) & 
                    (stats['cell_type'] == cell_type)
                ].copy()
                
                if len(stats_case) == 0:
                    print(f"    Warning: No data for {case_tf} in {cell_type}")
                    continue
                
                ctr = stats_case['ctrl'].unique()
                assert len(ctr) == 1, "Multiple control groups found"
                ctr = ctr[0]
                treatment = stats_case['condition'].unique()
                assert len(treatment) == 1, "Multiple treatment groups found"
                treatment = treatment[0]
                p_value_adj = stats_case['p_value_adj'].values[0]
                
                # Filter data for control and treatment groups
                plot_data = adata_df[
                    (adata_df['condition'].isin([ctr, treatment])) & 
                    (adata_df[case_tf].notna())
                ][['donor_id', 'condition', case_tf]].copy()

                # Create strip plot colored by donor
                plot_data['condition'] = plot_data['condition'].astype(
                    pd.CategoricalDtype(categories=[ctr, treatment], ordered=True)
                )
                plot_data['condition'] = plot_data['condition'].apply(lambda x: surrogate_names.get(x, x))
                
                donors = plot_data['donor_id'].unique()
                donor_map = {donor: f'Donor {i+1}' for i, donor in enumerate(donors)}
                plot_data['donor_id'] = plot_data['donor_id'].map(donor_map)
                
                sns.stripplot(
                    data=plot_data,
                    x='condition',
                    y=case_tf,
                    hue='donor_id',
                    ax=ax,
                    dodge=False,
                    jitter=True,
                    alpha=0.7,
                    size=4,
                    palette='tab10'
                )

                # rotate x-axis labels
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.margins(x=0.3, y=0.3)
                # Add p-value annotation
                y_max = plot_data[case_tf].max()
                y_min = plot_data[case_tf].min()
                y_range = y_max - y_min
                y_pos = y_max + 0.1 * y_range
                
                # if p_value_adj < 0.001:
                #     sig_text = '***'
                # elif p_value_adj < 0.01:
                #     sig_text = '**'
                # elif p_value_adj < 0.05:
                #     sig_text = '*'
                # else:
                #     sig_text = 'ns'
                sig_text = f'p={p_value_adj:.3f}'
                ax.text(0.5, y_pos, sig_text, ha='center', va='bottom', fontsize=7)
                ax.plot([0, 1], [y_pos - 0.02 * y_range, y_pos - 0.02 * y_range], 'k-', linewidth=1)
                
                ax.set_xlabel('')
                ax.set_ylabel(case_tf if i == 0 else '', fontsize=10)
                ax.set_title(case_tf, fontsize=10, pad=5)
                if i == len(case_tfs) - 1:
                    ax.legend(bbox_to_anchor=bbox_to_anchor, loc='upper left', fontsize=7, title_fontsize=8, frameon=False, 
                            labelspacing=0.2,
                            handletextpad=0.4,
                            borderpad=0.3,
                            columnspacing=0.6
                            )
                else:
                    ax.get_legend().remove()

                if i != 0:
                    ax.set_ylabel('')
                    ax.set_yticklabels([])
               

            output_path = os.path.join(
                args.output_dir, 
                f'{comparison}_{cell_type}.png'
            )
            output_path = output_path.replace(' ', '_').replace('(', '_').replace(')', '_').replace(':', '_')
            plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
            plt.close()
            print(f"    Saved: {output_path}")

def plot_pathway_analysis(stats, args):
    """Perform pathway analysis comparing aging and condition (disease or perturbation)."""
    print("Generating pathway analysis...")
    
    dataset = args.dataset
    data_type = args.data_type
    feature_type = args.feature_type
    analysis_type = args.analysis_type
    pathway_cell_types = args.cell_types
    output_dir = args.output_dir
    
    cfg = get_config(dataset)
    
    # Determine feature column based on feature type
    feature_col = 'target' if feature_type == 'gene_expression' else 'gene'
    
    # Get aging stats with the same feature type
    # Note: Aging stats are always 'bulk' type regardless of condition data type
    aging_stats_sig = retrieve_sig_stats(
        feature_type=feature_type
    ).drop_duplicates(subset=["cell_type", feature_col])
    aging_stats_sig = aging_stats_sig[[feature_col, "cell_type", "slope", 'p_value_adj']]
    
    # Filter significant condition stats
    stats_sig = stats[(stats['p_value_adj'] < 0.05)].copy()
    
    stats_sig = stats_sig[~stats_sig['slope'].isna()]
    
    if len(stats_sig) == 0:
        print("  Warning: No significant data for pathway analysis")
        return
    
    # Choose pathway analysis method based on analysis type
    if analysis_type == 'disease' or analysis_type == 'aging':
        # Use GSEA for disease/aging analysis
        _plot_disease_pathway_gsea(stats_sig, dataset, output_dir, feature_col, pathway_cell_types)
    else:
        # Use KDE for perturbation analysis (original behavior)
        # Pathway enrichment with appropriate feature column
        res_aging = pathway_kde_func(aging_stats_sig, min_genes=10, feature_col=feature_col)
        res_aging_sig = res_aging[res_aging['p_adj'] < 0.05]
        
        stats_sig['slope'] = stats_sig['slope']
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
        _plot_perturbation_pathway_gsea(stats_sig, dataset, output_dir, feature_col)

def _plot_disease_pathway_gsea(stats_sig, dataset, output_dir, feature_col, pathway_cell_types):
    """Generate GSEA pathway analysis for disease/aging."""
    from hiara.src.pathway_analysis.util import gsea_func
    from hiara.src.pathway_analysis.plots import plot_pathway_gsea
    
    print("\n  Running GSEA enrichment analysis...")
    
    # Prepare data for GSEA: add trend column based on slope direction
    stats_sig = stats_sig.copy()
    stats_sig['trend'] = stats_sig['slope'].apply(
        lambda x: 'Increase in disease' if x > 0 else 'Decrease in disease'
    )
    
    # Run GSEA
    pathway_scores = gsea_func(
        stats_sig,
        pvalue_col='p_value_adj',
        gene_sets=['MSigDB_Hallmark_2020'],
        feature_col=feature_col
    )
    
    if pathway_scores is not None and len(pathway_scores) > 0:
        print(f"  Found {len(pathway_scores)} significant pathways")
        
        # Plot combined GSEA results
        plot_pathway_gsea(pathway_scores, palette=palette_disease_effect)
        
        output_path = os.path.join(output_dir, f'{dataset}_pathway_gsea.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
        plt.close()
        print(f"  Saved GSEA plot: {output_path}")
    else:
        print("  No significant pathways found")

def _plot_perturbation_pathway_gsea(stats_sig, dataset, output_dir, feature_col):
    """Generate GSEA pathway analysis for perturbations."""
    from hiara.src.pathway_analysis.util import gsea_func
    from hiara.src.feature_association.plots import dotplot_category_color
    
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
def parse_args():
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
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=PLOTS_DIR,
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
        help='Case genes for trend plotting (disease only, default: LEF1 ZEB2)'
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
        help='Number of top aging genes to show (default: 15)'
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
    parser.add_argument(
        '--skip-overview',
        action='store_true',
        help='Skip overview heatmap (perturbation only)'
    )
    parser.add_argument(
        '--association-type',
        type=str,
        default='grouped',
        choices=['grouped', 'continous'],
        help='Association type used in run_analysis: grouped (uses slope_condition) or continous (uses slope). Default: grouped'
    )
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print(f"Dataset: {args.dataset}")
    if args.config_label:
        print(f"Config label: {args.config_label}")
    print(f"Analysis type: {args.analysis_type}")
    print(f"Data type: {args.data_type}")
    print(f"Feature type: {args.feature_type}")
    print("="*60)

    args.case_tfs = ['KLF6', 'PRDM1' ,'LEF1', 'ZEB2']
    
    # Load statistics
    stats = retrieve_features_stats(
        multi_cohort=False,
        dataset=args.dataset,
        data_type=args.data_type,
        feature_type=args.feature_type
    )
    stats_sig = retrieve_sig_stats(
        multi_cohort=False, 
        dataset=args.dataset,
        data_type=args.data_type,
        feature_type=args.feature_type
    )
    if args.dataset == 'soundlife':
        plot_directional_consistency_scatter(
            stats_sig,
            args
        )
    if args.dataset == 'perez_sle':
        plot_aging_disease_overlap_heatmap(stats_sig, args)
        plot_central_tf_act_disease(stats, args)
        plot_disease_case_tfs(args)

    if args.dataset == 'op':
        if not args.skip_overview:
                plot_overview_heatmap(stats_sig, args)
        # plot_aging_overlap(
        #     stats_sig, 
        #     args
        # )     
        plot_central_tf_act_perturbation(stats, args)
        plot_ctr_condition_donor_level(args) 
    if args.dataset == 'CXCL9':
        # if not args.skip_overview:
        #         plot_overview_heatmap(stats_sig, args)
        # plot_aging_overlap(
        #     stats_sig, 
        #     args
        # )     
        
        plot_central_tf_act_perturbation(stats, args)
        # plot_ctr_condition_donor_level(args) 
  
    
if __name__ == '__main__':
    main()
