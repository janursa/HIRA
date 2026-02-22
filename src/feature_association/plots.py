import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, linregress
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.cluster.hierarchy import linkage
from matplotlib.patches import Patch


from hiara.src.config import FEATURES_DIR, MAJOR_CT_LABEL, PRIOR_DIR, MAJOR_CTS, SUB_CTS , PLOTS_DIR, colors_blind, DISCOVERY_COHORTS, \
    surrogate_names, palette_datasets, palette_trend, palette_datasets_pretty, mapping_minor_2_major, \
    palette_trend_2, palette_major_cts, palette_sub_cts, palette_datasets, palette_trend_2, colors_blind, \
        get_config_fa, cmap_trend
from hiara.src.feature_association.helper import bin_feature_values, retrieve_feature_data, \
                                                    retrieve_sig_stats, retrieve_stats
from hiara.src.utils.util import retrieve_net, retrieve_adata, retrieve_net_consensus
from hiara.src.utils.plots import dotplot

def categorize_granularity(df, granularity):
    categories = MAJOR_CTS if granularity == MAJOR_CT_LABEL else SUB_CTS
    categories = [c for c in categories if c in df['cell_type'].unique()]
    df['cell_type'] = pd.Categorical(
                                df['cell_type'], 
                                categories=categories, 
                                ordered=True
                                )
    
    return df
    
def wrapper_sig_features_counts(args):
    analysis_name = args.analysis_name
    granularity = get_config_fa(analysis_name)['granularity']
    aging_stats_sig = retrieve_sig_stats(analysis_name=analysis_name).drop_duplicates(subset=['cell_type', 'gene'])
    aging_stats_sig = categorize_granularity(aging_stats_sig, granularity)

    aging_stats_sig['cell_type'] = aging_stats_sig['cell_type'].apply(lambda x: surrogate_names.get(x, x))
    sig_features_counts(aging_stats_sig, palette=palette_trend_2)
    
    feature_type = get_config_fa(args.analysis_name)['feature_type']
    plt.ylabel('Significant TFs' if feature_type == 'tf_activity' else 'Significant features')
    file_name = f"{PLOTS_DIR}/aging_features_count_{args.analysis_name}.png"
    print(f"Saving figure to {file_name}")
    plt.savefig(file_name, bbox_inches='tight', dpi=300, transparent=True)


def plot_aging_overlap(analysis_name, stats_sig, cell_types, args):
    """Plot overlap between condition (disease/perturbation) and aging genes."""
    print("Generating aging overlap plot...")
    
    dataset = args.dataset
    included_cell_types = cell_types
    agreement = args.agreement
    output_dir = args.output_dir
    
    aging_stats_sig = retrieve_sig_stats(analysis_name=analysis_name).drop_duplicates(subset=['cell_type', 'gene'])
    aging_stats_sig = aging_stats_sig[['gene', 'cell_type', 'slope']].copy()

    comparisons = stats_sig['comparison'].unique()
    
    print('Stats of comparison:', stats_sig.groupby(['cell_type', 'comparison'])['gene'].nunique())
    # fig, axes = plt.subplots(1, len)
    for comparison in comparisons:
        fig, axes = plt.subplots(1, len(included_cell_types), figsize=(1.5*len(included_cell_types)+1, 2))
        for i, cell_type in enumerate(included_cell_types):
            ax = axes[i] if len(included_cell_types) > 1 else axes  
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
                legend_loc=(1, 0.5),
                ax=ax
            )
            
            if i != len(included_cell_types) - 1:
                ax.legend_.remove()
            if i != 0:
                ax.set_ylabel('')
        comparison = comparison.replace('(', '_').replace(')', '_').replace(':', '_').replace(' ', '_')
        name_suffix = f'{dataset}_{comparison}'
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'ref_overlap_{name_suffix}_{args.feature_type}.png')
        output_path = output_path.replace(' ', '_').replace('(', '_').replace(')', '_').replace(':', '_')
        plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
        plt.close()
        print(f"    Saved: {output_path}")

def plot_directional_consistency_scatter(
                                        stats, 
                                        stats_ref,
                                        x_label = 'Validation analysis \n(significance)',
                                        y_label = 'Discovery analysis \n(significance)',
                                        association_col = '-log10_p_adj',
                                        agreement='same',
                                        label_consistent = 'Consistent',
                                        label_opposing = 'Opposing',
                                        output_dir = FEATURES_DIR,
                                        save_suffix = '',
                                        pvalue_col='p_value_adj'
                                        ):
    

    if stats['comparison'].nunique() != 1:
        raise ValueError(f"should only have one comparision but has : {stats['comparison'].unique()}")

    if agreement == 'same':
        opposing_color = 'indianred'
        consistent_color = 'darkseagreen' 
    elif agreement == 'opposite':
        opposing_color = 'darkseagreen'
        consistent_color = 'indianred'

    else:
        raise ValueError(f"Unsupported agreement type: {agreement}")

    label_consistent = label_consistent + ' \n ({} TFs)'
    label_opposing = label_opposing + ' \n ({} TFs)'

    stats = stats.groupby(['cell_type', 'gene', 'comparison']).agg({'slope': 'mean', pvalue_col: 'max'}).reset_index()
    # Load reference aging genes (only significant)
    stats_ref = stats_ref.groupby(['cell_type', 'gene', 'comparison']).agg({'slope': 'mean', 'meta_p_adj': 'max'}).reset_index()
    included_cell_types = stats['cell_type'].unique()
    all_cell_data = []
    for cell_type in included_cell_types:
        if cell_type not in stats['cell_type'].unique():
            print(f"  Warning: No data for {cell_type}")
            continue
        ref_ct = stats_ref[stats_ref['cell_type'] == cell_type].copy()

        sl_ct = stats[(stats['cell_type'] == cell_type) & (stats['gene'].isin(ref_ct['gene']))].copy()
        if len(sl_ct) == 0:
            print(f"  Warning: No overlapping genes for {cell_type}")
            print(f"available cell types in stats: {stats['cell_type'].unique()}")
            raise ValueError(f"No overlapping genes or cell types for {cell_type}")

    
        # Merge: keep only TFs that are significant in reference aging
        if pvalue_col == 'meta_p_adj':
            pvalue_col = f'meta_p_adj_sl'
            sl_ct.rename({'meta_p_adj': 'meta_p_adj_sl'}, axis=1, inplace=True)
        if True:
            
            sl_ct['abs_log10_p_adj_sl'] = sl_ct[pvalue_col].apply(lambda x: -np.log10(x + 1e-300))
            # print(f"\n  Top 5 TFs with positive slope:")
            # top_5 = sl_ct[sl_ct['slope'] > 0].nlargest(5, 'abs_log10_p_adj_sl')
            # names = ', '.join(top_5['gene'].tolist())
            # print(f"    {cell_type}: {names}")
            # print(f"\n  Top 5 TFs with negative slope:")
            # top_5 = sl_ct[sl_ct['slope'] < 0].nlargest(5, 'abs_log10_p_adj_sl')
            # names = ', '.join(top_5['gene'].tolist())
            # print(f"    {cell_type}: {names}")
        merged = ref_ct.merge(
            sl_ct[['gene', 'slope', 'comparison' ,pvalue_col]],
            on='gene',
            how='left',
            suffixes=('_ref', '_sl')
        )
        
        missing_merged = merged[merged['slope_sl'].isna()]
        if len(missing_merged) == 0:
            aging_genes = ref_ct['gene'].unique()
            c_genes = sl_ct['gene'].unique()
            overlap = set(aging_genes).intersection(set(c_genes))
            print(f"  {cell_type}: All {len(overlap)} aging TFs found in SL data.")
        else:
            print(missing_merged['gene'].nunique(), ' missing genes in the condition data for ', cell_type)
            # Filter out missing genes before continuing
            merged = merged[~merged['slope_sl'].isna()].copy()
        
        merged['consistent'] = np.sign(merged['slope_ref']) == np.sign(merged['slope_sl'])
        merged['cell_type'] = cell_type

        merged['-log10_p_adj_sl'] = -np.log10(merged[pvalue_col] + 1e-300) * np.sign(merged['slope_sl'])
        merged['-log10_p_adj_ref'] = -np.log10(merged['meta_p_adj'] + 1e-300) * np.sign(merged['slope_ref'])
        all_cell_data.append(merged)
        
    if not all_cell_data:
        print("  Warning: No data to plot")
        return
    
    # Combine all cell types
    combined_data = pd.concat(all_cell_data, ignore_index=True)
    
    # Create grouped plot
    fig, axes = plt.subplots(1, len(included_cell_types), figsize=(2 * len(included_cell_types) + 1, 2.7), sharey=False)
    
    if len(included_cell_types) == 1:
        axes = [axes]
    
    for idx, cell_type in enumerate(included_cell_types):
        ax = axes[idx]
        cell_data = combined_data[combined_data['cell_type'] == cell_type]
        if len(cell_data) == 0:
            continue
        consistent = cell_data[cell_data['consistent']]
        inconsistent = cell_data[~cell_data['consistent']]
        
        # Plot inconsistent 
        s=10
        linewidths=0.1
        if len(inconsistent) > 0:
            ax.scatter(
                inconsistent[f'{association_col}_sl'],
                inconsistent[f'{association_col}_ref'],
                c=opposing_color,
                s=s,
                alpha=0.6,
                label=label_opposing.format(len(inconsistent)),
                edgecolors='darkred',
                linewidths=linewidths
            )
        
        # Plot consistent 
        if len(consistent) > 0:
            ax.scatter(
                consistent[f'{association_col}_sl'],
                consistent[f'{association_col}_ref'],
                c=consistent_color,
                s=s,
                alpha=0.6,
                label=label_consistent.format(len(consistent)),
                edgecolors='darkgreen',
                linewidths=linewidths
            )
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        
        # Set symmetric axis limits around zero
        # Check for NaN values
        if cell_data[f'{association_col}_sl'].isna().any():
            raise ValueError(f"NaN values found in {association_col}_sl for cell type {cell_type}")
        if cell_data[f'{association_col}_ref'].isna().any():
            raise ValueError(f"NaN values found in {association_col}_ref for cell type {cell_type}")
        
        x_max = max(abs(cell_data[f'{association_col}_sl'].min()), abs(cell_data[f'{association_col}_sl'].max()))
        y_max = max(abs(cell_data[f'{association_col}_ref'].min()), abs(cell_data[f'{association_col}_ref'].max()))
        
        # Add some padding
        x_max *= 1.05
        y_max *= 1.2  # Increased from 1.05 to 1.5 for more vertical space
        
        ax.set_xlim(-x_max, x_max)
        ax.set_ylim(-y_max, y_max)
        
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        # if idx == 0:
        #     ax.set_ylabel(y_label, fontsize=10)
        # else:
        #     ax.set_ylabel('')
        
        ax.set_title(f'{cell_type}', fontsize=12, pad=30)
        ax.grid(False)

        from matplotlib.lines import Line2D
        legend_elements = []
        markersize=5
        if len(inconsistent) > 0:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', 
                       markerfacecolor=opposing_color, markersize=markersize,
                       markeredgecolor='darkred', markeredgewidth=0.5,
                       label=label_opposing.format(len(inconsistent)))
            )
        if len(consistent) > 0:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', 
                       markerfacecolor=consistent_color, markersize=markersize,
                       markeredgecolor='darkgreen', markeredgewidth=0.5,
                       label=label_consistent.format(len(consistent)))
            )
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.35), 
                 frameon=False, fontsize=8, ncol=2, columnspacing=-.2)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'consistency_scatter{save_suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def _plot_scatter_feature_vs_age(features, analysis_name, cell_type, datasets=DISCOVERY_COHORTS):
    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_features == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        all_ages = []
        all_values = []
        all_datasets = []
        for dataset in datasets:
            try:
                feature_data = retrieve_feature_data(
                    analysis_name=analysis_name,
                    dataset=dataset,
                    cell_type=cell_type
                )
                # Get feature index
                if feature not in feature_data.var_names:
                    continue
                
                feature_idx = list(feature_data.var_names).index(feature)
                feature_values = feature_data.X[:, feature_idx]
                
                if hasattr(feature_values, 'toarray'):
                    feature_values = feature_values.toarray().flatten()
                elif hasattr(feature_values, 'flatten'):
                    feature_values = feature_values.flatten()
                
                ages = feature_data.obs['age'].values
                
                all_ages.extend(ages)
                all_values.extend(feature_values)
                all_datasets.extend([dataset] * len(ages))
                
            except Exception as e:
                print(f"Could not {dataset}/{cell_type}: {e}")
                continue
        
        if not all_ages:
            ax.text(0.5, 0.5, f'No data for {feature}', 
                    ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Create dataframe for plotting
        plot_df = pd.DataFrame({
            'age': all_ages,
            'value': all_values,
            'dataset': all_datasets
        })
        plot_df['dataset'] = plot_df['dataset'].apply(lambda x: surrogate_names.get(x, x))
        
        # Plot scatter points colored by dataset
        for dataset in plot_df['dataset'].unique():
            dataset_data = plot_df[plot_df['dataset'] == dataset]
            color = palette_datasets_pretty.get(dataset, 'gray')
            ax.scatter(dataset_data['age'], dataset_data['value'], 
                        alpha=0.5, s=20, color=color, label=dataset)
        
        ax.set_xlabel('Age (years)', fontsize=10)
        y_label = get_config_fa(analysis_name)['feature_type']
        ax.set_ylabel(surrogate_names.get(y_label, y_label), fontsize=10)
        feature = surrogate_names.get(feature, feature)
        ax.set_title(f'{feature}', fontsize=11, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        
        if idx == (n_features-1):
            ax.legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Remove empty subplots
    for idx in range(n_features, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle(f'{cell_type}', fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
def plot_scatter_feature_vs_age(
        analysis_name, 
        cell_types=None, 
        features=None, 
        filter_for_sig=True,
        feature_selection_mode='top_central', 
        top_features=5, 
        datasets=DISCOVERY_COHORTS
        ):
    assert feature_selection_mode in ['top_central', 'top_sig']
    if filter_for_sig:
        stats_all = retrieve_sig_stats(analysis_name=analysis_name)
    else:
        stats_all = retrieve_stats(analysis_name=analysis_name)
    cell_types = cell_types if cell_types is not None else stats_all['cell_type'].unique()
    for cell_type in cell_types:
        stats = stats_all[stats_all['cell_type'] == cell_type]
        if stats.empty:
            print(f"No significant features for {cell_type}, skipping...")
            continue
        if features is not None:
            selected_features = features
        elif feature_selection_mode == 'top_central':
            net = retrieve_net_consensus(cell_type=cell_type)
            net = net[net['source'].isin(stats['gene'])]
            centrality = net['source'].value_counts().reset_index()
            centrality.columns = ['gene', 'degree']
            centrality = centrality.sort_values(by='degree', ascending=False)
            selected_features = centrality['gene'].head(top_features).tolist()
        elif feature_selection_mode == 'top_sig':
            selected_features = stats.drop_duplicates(subset='gene').sort_values(by='meta_p_adj', ascending=True).head(top_features)['gene'].tolist()
        
        if not selected_features:
            raise ValueError(f"No features selected for {cell_type}, skipping...")
        
        _plot_scatter_feature_vs_age(
            features=selected_features, analysis_name=analysis_name, cell_type=cell_type, datasets=datasets)
        tag = 'custom' if features is not None else ('top_central' if feature_selection_mode == 'top_central' else 'top_sig') 
        file_name = f"{PLOTS_DIR}/scatter_feature_vs_age_{analysis_name}_{cell_type}_{tag}.png"
        print(f"Saving figure to {file_name}")
        plt.savefig(file_name, bbox_inches='tight', dpi=300, transparent=True)
        plt.close()
def plot_young_vs_aging(analysis_name, cell_type, 
                        young_age_threshold=30, annotate_top_n=10, plots_dir=PLOTS_DIR):
    """
    Plot mean feature values in young adults (< age threshold) vs aging slope.

    """

    # Get significant aging associations
    stats_sig = retrieve_sig_stats(analysis_name=analysis_name, cell_type=cell_type)
    feature_type = get_config_fa(analysis_name)['feature_type']
    datasets = stats_sig['dataset'].unique()
    
    # Get average slope per feature
    feature_slopes = stats_sig.groupby('gene').agg({
        'slope': 'mean',
        'meta_p_adj': 'first'
    }).reset_index()
    
    # Collect young adult feature values from all datasets
    young_values_list = []
    
    for dataset in datasets:
        feature_data = retrieve_feature_data(
            analysis_name=analysis_name,
            dataset=dataset,
            cell_type=cell_type
        )
        
        young_mask = feature_data.obs['age'] < young_age_threshold
        if young_mask.sum() == 0:
            print(f'Skipping young group for {dataset} {cell_type}')
            continue
        
        feature_data_young = feature_data[young_mask]
        
        # Get mean values for each feature
        X = feature_data_young.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        mean_values = np.mean(X, axis=0)
        
        # Create dataframe
        young_df = pd.DataFrame({
            'gene': feature_data_young.var_names,
            'mean_young_value': mean_values,
            'dataset': dataset
        })
        
        young_values_list.append(young_df)
        
    
    assert len(young_values_list) > 0, f"No young adult data for {cell_type}"
    
    # Combine all datasets and get overall mean
    all_young_values = pd.concat(young_values_list, ignore_index=True)
    young_means = all_young_values.groupby('gene')['mean_young_value'].mean().reset_index()
    
    # Merge with slopes
    plot_df = feature_slopes.merge(young_means, on='gene', how='inner')
    
    assert len(plot_df) > 0, f"No overlapping features between young values and slopes for {cell_type}"
    
    # Calculate -log10 p-value for sizing
    plot_df['-log10_p'] = -np.log10(plot_df['meta_p_adj'] + 1e-300)
    
    # Determine alignment: aligned if same sign (both positive or both negative), orthogonal if different signs
    plot_df['aligned'] = (plot_df['mean_young_value'] * plot_df['slope']) > 0
    
    # Create plot
    fig, ax = plt.subplots(figsize=(5, 3))
    
    # Separate aligned and orthogonal
    aligned_data = plot_df[plot_df['aligned']]
    orthogonal_data = plot_df[~plot_df['aligned']]
    
    # Plot orthogonal first (so aligned appears on top)
    s_factor = 1
    if len(orthogonal_data) > 0:

        ax.scatter(orthogonal_data['slope'], orthogonal_data['mean_young_value'], 
                    c='indianred', alpha=0.6, 
                    s=orthogonal_data['-log10_p'] * s_factor,
                    edgecolors='darkred', linewidths=0.5,
                    label=f'Converge ({len(orthogonal_data)})')
    
    if len(aligned_data) > 0:
        ax.scatter(aligned_data['slope'], aligned_data['mean_young_value'], 
                    c='darkseagreen', alpha=0.6, 
                    s=aligned_data['-log10_p'] * s_factor,
                    edgecolors='darkgreen', linewidths=0.5,
                    label=f'Diverge ({len(aligned_data)})')

    # Labels and styling
    feature_type = surrogate_names.get(feature_type, feature_type)
    ax.set_xlabel(f'{feature_type} aging slope', fontsize=10)
    ax.set_ylabel(f'{feature_type} in young adults\n(<{young_age_threshold} years)', fontsize=10)
    ax.set_title(f'{cell_type}', fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    
    # Add legend
    ax.legend(frameon=False, fontsize=9, loc=[1.1, 0.5], title='Naive to Effector')
    
    plt.tight_layout()
    
    # Save plot
    file_name = f"{plots_dir}/young_vs_aging_{analysis_name}_{cell_type}.png"
    print(f"Saving figure to {file_name}")
    plt.savefig(file_name, bbox_inches='tight', dpi=300, transparent=True)
    plt.close()

def gsea_analysis(stats_sig):
    from hiara.src.pathway_analysis.util import get_genesets, pathway_kde_func, get_hallmark, gsea_func, wrapper_gsea

    wrapper_gsea(stats_sig)
    file_name = f"{PLOTS_DIR}/gsea_tf_activity.png"
    print(f"Saving figure to {file_name}")
    plt.savefig(file_name, bbox_inches='tight', dpi=200)
def plot_heatmap_overal(stats_aging, analysis_name):
    granularity = get_config_fa(analysis_name)['granularity']
    stats_aging = categorize_granularity(stats_aging, granularity)
    stats_aging['dataset'] = pd.Categorical(stats_aging['dataset'], categories=DISCOVERY_COHORTS, ordered=True)
    trends = get_config_fa(analysis_name)['trend_labels'] if 'trend_labels' in get_config_fa(analysis_name) else ['Decrease in aging', 'Increase in aging']
    plot_overall_heatmap(stats_aging, 
                        sig_dots_y_offset=3, 
                        first_col='cell_type',  
                        first_col_palette=palette_major_cts if granularity == MAJOR_CT_LABEL else palette_sub_cts,
                        second_col='dataset', 
                        second_col_palette=palette_datasets,
                        bbox_to_anchor=(1.07, 1.07),
                        bbox_to_anchor_col2=(1.05, .9),
                        bbox_to_anchor_col1=(1.05, 0.61),
                        trend_colors = ['#B0BF1A', '#E52B50'],
                        trend_names = trends,
                        figsize=(4, 7),
                        map_names={**{'cell_type':'Cell type', 'dataset': 'Dataset'}, **surrogate_names})
    file_name = f"{PLOTS_DIR}/overall_heatmap_{analysis_name}.png"
    print(f"Saving figure to {file_name}")
    plt.savefig(file_name, bbox_inches='tight', dpi=300, transparent=True)
def plot_central_features(stats_aging, cell_types):
    if True:
        # Parameters
        n_top = 10
        focus = 'source'  # can be 'source' or 'target'
        palette = palette_trend_2  # assume this dict is defined

        # Collect top TFs across cell types
        rows = []
        for ct in cell_types:
            stats_ct = stats_aging[stats_aging['cell_type'] == ct].drop_duplicates(subset='gene')[['gene', 'slope', 'trend', 'cell_type']]
            aging_genes = stats_ct['gene'].unique()
            net_ct = retrieve_net_consensus(cell_type=ct)
            net_ct = net_ct[net_ct[focus].isin(aging_genes)]
            centrality = net_ct[focus].value_counts().reset_index(name='degree')    
            max_degree = centrality['degree'].max()
            centrality['normalized_centrality'] = centrality['degree'] / max_degree if max_degree > 0 else 0 
            centrality = centrality.head(n_top) 
            
            # Merge with TF stats to get trend
            merged = centrality.merge(stats_ct, left_on=focus, right_on='gene', how='left')
            rows.append(merged)
        plot_df = pd.concat(rows, ignore_index=True)
        col_c = 'normalized_centrality'
        plot_df[focus] = plot_df[focus].astype(str)
        plot_df = plot_df.sort_values(by=['cell_type', col_c], ascending=[True, False])
        n_cols = len(cell_types)
        fig, axes = plt.subplots(1, n_cols, figsize=(1.5*n_cols, 2.1 ), sharex=False)

        if n_cols == 1:
            axes = [axes]
        i = 0
        for ax, ct in zip(axes, cell_types):
            df_ct = plot_df[plot_df['cell_type'] == ct]
            colors = df_ct['trend'].map(palette)
            ax.barh(df_ct[focus], df_ct[col_c], color=colors, alpha=0.5)
            ax.set_title(f'{ct}', fontsize=10)
            ax.invert_yaxis()
            if i==0:
                ax.set_xlabel('Out-degree centrality' if focus == 'source' else 'In-degree centrality', fontsize=8)
            else:
                ax.set_xlabel('')
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
            ax.margins(x=0.1, y=0.1)
            ax.spines[['top', 'right']].set_visible(False)
            i+=1
        fig.tight_layout()
        file_name = os.path.join(PLOTS_DIR, f'central_aging_{focus}.png')
        print(f"Saving figure to {file_name}")
        plt.savefig(file_name, bbox_inches='tight', dpi=300, transparent=True)
def plot_interaction_of_features_between_cell_types(args):
    from geneRNBI.src.exp_analysis.helper import plot_interactions, create_interaction_df

    stats_sig = retrieve_sig_stats(analysis_name=args.analysis_name).drop_duplicates(subset=['cell_type', 'gene'])
    if stats_sig['cell_type'].nunique() <2:
        print("Not enough cell types for interaction analysis.")
        return
    df_dict = stats_sig.groupby(['cell_type'])['gene'].apply(list).to_dict()
    interaction_main_df = create_interaction_df(df_dict)
    aa = plot_interactions(interaction_main_df, min_subset_size=5, min_degree=1, color_map=palette_major_cts)
    file_name = f"{PLOTS_DIR}/interactions.png"
    print(f"Saving figure to {file_name}")
    plt.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')
    # plt.title(surrogate_names[race], pad=40, fontsize=10, fontweight='bold')

    ttypes = ['CD8T', 'CD4T', 'NK']
    mask = interaction_main_df[ttypes].sum(axis=1)==len(ttypes)
    features = mask[mask].index.unique()
    print(len(features))
    for cell_type in ttypes:
        plot_features_vs_datasets(cell_type=cell_type, datasets=DISCOVERY_COHORTS, features=features, 
                                    sizes=(90, 100), analysis_name=args.analysis_name
                                    )
    if False: ### Shared sig TFs between CD8T and CD4T
        from hiara.src.feature_association.plots import plot_joint_scatter
        ttypes = ['CD8T', 'CD4T']
        mask = interaction_main_df[ttypes].sum(axis=1)==len(ttypes)
        features = mask[mask].index
        # Get significant TFs for CD4T and CD8T cells
        df = stats_sig[stats_sig['gene'].isin(features)].drop_duplicates(subset=['cell_type', 'gene'])
        df['neg_log10_adj_pval'] = -np.log10(df['meta_p_adj'])

        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        plot_joint_scatter(df, vars=['CD4T', 'CD8T'], annotate=True, ax=ax)
        ax.margins(x=0.1, y=0.1)

def plot_case_tf(args):
    from hiara.src.feature_association.plots import plot_feature_values_all_datasets
    selected_cell_types = ['CD8T', 'CD4T', 'NK'] # ['CD8T', 'CD4T', 'NK'] #Tcm_Naive_CD8
    datasets = DISCOVERY_COHORTS
    plot_genexpression = True
    show_cbar=False
    for case_tf in ['SATB1', 'GATA3']:  # 'TCF7' 'SATB1' 
        for i, cell_type in enumerate(selected_cell_types):
            if i == 0:
                show_ylabels=True
            else:
                show_ylabels=False
            
            if plot_genexpression:
                fig, axes = plt.subplots(2, 1, figsize=(2, 2.2), sharex=True)
            else:
                fig, ax = plt.subplots(1, 1, figsize=(2, .7))

            ax = axes[0] if plot_genexpression else ax
            plot_feature_values_all_datasets(cell_type, analysis_name=args.analysis_name, feature=case_tf,datasets=datasets, show_cbar=show_cbar, ax=ax,
                                            show_ylabels=show_ylabels)
            if plot_genexpression:
                ax.set_xlabel('')
                ax = axes[1]
                plot_feature_values_all_datasets(cell_type, analysis_name='ge_major_b', feature=case_tf, 
                                                datasets=datasets, show_cbar=show_cbar, ax=ax, show_ylabels=show_ylabels)
            
            plt.suptitle(f'{cell_type}', y=1.05)
            file_name = f"{PLOTS_DIR}/case_tf_{case_tf}_{cell_type}.png"
            print(f"Saving figure to {file_name}")
            plt.savefig(file_name, bbox_inches='tight', dpi=300)

def sig_features_counts(df, figsize=None, palette=None, ax=None):
    x_label_count = df['cell_type'].nunique() 
    df = df[['gene', 'cell_type', 'trend']]
    df['trend'] = df['trend'].astype(CategoricalDtype(categories=palette.keys(), ordered=True))
    df = df[~df.duplicated()].reset_index(drop=True)
    df_counts = df.groupby(['cell_type', 'trend']).size().reset_index(name='count')
    if ax is None:
        if figsize is None:
            figsize = (.3*x_label_count+1, 2.5)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.barplot(data=df_counts, x='cell_type', y='count', alpha=.8, hue='trend', palette=palette, ax=ax)
    # ax.set_ylabel('Gene count')
    ax.set_xlabel('')
    ax.margins(x=0.1 if x_label_count < 5 else 0.05, y=0.1)
    ax.legend(loc=(1, 0.5), title='Trend', frameon=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    # plt.suptitle('TFs significantly associated with age', fontsize=12)
    # plt.tight_layout()

def plot_feature_values_per_datasets(cell_type, features, data_type, datasets, feature_type='gene_expression', age_limit=[20, 75], cluster=False, figsize=None):
    import matplotlib.pyplot as plt
    from hiara.src.config import surrogate_names

    n_datasets = len(datasets)
    n_features = len(features)
    if figsize is None:
        figsize = (n_datasets*3, .2*n_features+1)
    fig, axes = plt.subplots(1, n_datasets, figsize=figsize, sharey=False)
    for i, (dataset) in enumerate(datasets):
        if feature_type == 'tf_activity':
            adata = retrieve_feature_data(dataset=dataset, cell_type=cell_type, data_type=data_type, feature_type=feature_type) 
        elif feature_type == 'gene_expression':
            adata = retrieve_adata(dataset=dataset, cell_type=cell_type, data_type=data_type)
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
        adata = adata[:, adata.var_names.isin(features)]
        
        if age_limit is not None:
            adata = adata[(adata.obs['age'] < age_limit[1]) & (adata.obs['age'] > age_limit[0])]
        # - plot target gene expression trend    
        ax = axes[i] if n_datasets > 1 else axes
        mean_expr = bin_feature_values(adata)
        if cluster:
            if i == 0:
                from sklearn.cluster import KMeans
                if len(mean_expr) > 2:
                    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(mean_expr)
                    mean_expr['cluster'] = clusters
                    mean_expr = mean_expr.sort_values('cluster').drop(columns=['cluster'])
                ordered_targets = mean_expr.index
            else:
                mean_expr = mean_expr.reindex(ordered_targets)
        else:
            mean_expr = mean_expr.reindex(features)
        if i == 0:
            show_cbar = True
        else:
            show_cbar = False
        
        heatplot_age_trend(mean_expr, cmap='viridis' if feature_type=='gene_expression' else 'magma', 
                            cbar_title="Gene expression" if feature_type=='gene_expression' else "TF activity", 
                            y_label="Genes" if feature_type=='gene_expression' else "TFs", 
                            ax=ax, 
                            show_cbar=show_cbar)
        if i != 0:
            ax.set_yticklabels([])
            ax.set_ylabel('')
        ax.set_title(surrogate_names[dataset], pad=10, fontsize=10, fontweight='bold')
    # plt.tight_layout()
    # plt.suptitle(cell_type, fontsize=12, fontweight='bold', y=1.05)
    return fig
def plot_feature_values_all_datasets(cell_type, feature, 
                                    analysis_name,
                                    datasets, ax=None, show_cbar=True,
                                    age_limit=[20, 80], show_ylabels=True):
    from hiara.src.feature_association.helper import retrieve_feature_data, bin_feature_values

    mean_expr_store = []
    for dataset in datasets:
        adata = retrieve_feature_data(dataset=dataset, cell_type=cell_type, analysis_name=analysis_name) 
        adata = adata[(adata.obs['age'] > age_limit[0]) & (adata.obs['age'] < age_limit[1])]
        adata = adata[:, adata.var_names==feature]
        assert adata.shape[1] == 1, f"Feature {feature} not found in dataset {dataset} for cell type {cell_type}"
        if adata.shape[1] == 0:
            continue
        expr = bin_feature_values(adata)
        expr.index = [dataset]
        mean_expr_store.append(expr)
    
    if len(mean_expr_store) == 0:
        return
    mean_expr = pd.concat(mean_expr_store)

    mean_expr.index = mean_expr.index.map(surrogate_names)
    ages = sorted(mean_expr.columns)
    # print(mean_expr)
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 2))

    feature_type = get_config_fa(analysis_name)['feature_type']
    heatplot_age_trend(mean_expr[ages], cmap='magma' if feature_type=='tf_activity' else 'viridis', 
                        cbar_title = "Gene \n expression" if feature_type == 'gene_expression' else (
                                    "TF \n activity" if feature_type == 'tf_activity' else "Gene score"
                                ),
                        ax=ax, 
                        show_cbar=show_cbar,
                        cbar_kws={
                            "shrink": 1,
                            "aspect": 5,       # Lower values = thicker colorbar (default is ~20)
                            "fraction": 0.1    # Controls the width space the cbar takes in the figure
                        })
    if not show_ylabels:
        ax.set_yticklabels([])
    ax.set_ylabel('')
    ax.set_xlabel('Age')
    # ax.set_title(f'{cell_type}: {feature}', pad=20)

def plot_trend_sle_case(adata, tf='LEF1', cell_type='CD8T'):
    from scipy.stats import linregress

    adata = retrieve_feature_data(dataset='SLE_European', cell_type=cell_type, feature_type='tf_activity', condition=None)
    adata = adata[:, adata.var_names == tf]
    # Extract feature values (flattened, assuming dense matrix)
    feature = adata.X.flatten()  # use .toarray().flatten() if sparse
    age = adata.obs['age'].values

    data = pd.DataFrame({
        'feature': feature,
        'age': age
    })

    # Define masks
    young_mask = data['age'] < 50
    old_mask = data['age'] >= 50

    fig, axes = plt.subplots(1, 2, figsize=(3, 2.2))

    i = 0
    for ax, (mask, color, title) in zip(
        axes,
        [(young_mask, 'royalblue', 'age < 50'), (old_mask, 'darkorange', 'age > 50')]
    ):
        subdata = data[mask]
        ax.scatter(
            subdata['age'],
            subdata['feature'],
            color=color,
            alpha=0.7,
            s=15
        )
        slope, intercept, r_value, p_value, std_err = linregress(subdata['age'], subdata['feature'])
        x_vals = np.linspace(subdata['age'].min(), subdata['age'].max(), 100)
        y_vals = slope * x_vals + intercept
        ax.plot(x_vals, y_vals, color='black', linestyle='--', linewidth=2)
        ax.set_yticks([])
        ax.margins(x=0.1, y=0.2)
        sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
        text = f'p = {p_value:.3g}{sig}' if p_value < 0.05 else f'p = {p_value:.3g}'
        # ax.text(0.5, 1.05, text, transform=ax.transAxes,
        #         ha='center', va='top', fontsize=10, color='black')
        if i == 0:
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_xlabel('Age')
        else:
            ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.set_title(f'{title}\n{text}', pad=15, fontsize=10)
        i+=1
    axes[0].set_ylabel('TF activity')
    plt.suptitle(f'{tf}', fontsize=10, fontweight='bold', y=.9)
    plt.tight_layout()

def plot_tf_act_central_tfs(df, all_groups, palette_all, feature_col='gene', figsize=(3.5, 5), plot_centrality=True,
                                ax2_margins={'y': 0.1, 'x': 0.1}, hide_ylabels=False, show_legend=True):
    
    # stats_d_sig = stats_d[stats_d['p_value_adj'] < 0.05]
    # --- Plot ---
    tfs = df[feature_col].unique()
    if plot_centrality:
        fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1.2, .8]})
    else:
        fig, axes = plt.subplots(1, 1, figsize=figsize)

    # - Scatter plot
    ax0 = axes[0] if plot_centrality else axes
    sns.scatterplot(data=df, x='analysis', y=feature_col, hue='trend', ax=ax0, palette=palette_all, s=100, alpha=.8)

    # Overlay black stars
    aging_df = df[df['analysis'] == 'Age-associated']
    assert aging_df.shape[0] > 0, "No Aging TFs found in the data"
    ax0.scatter(aging_df['analysis'], aging_df[feature_col], color='black', marker='*', s=10, zorder=10)
    sig_df = df[df.get('p_value_adj', 1.0) < 0.05]
    ax0.scatter(sig_df['analysis'], sig_df[feature_col], color='black', marker='*', s=10, zorder=10, alpha=.8)

    # Fill in missing x-axis categories
    missing = set(all_groups) - set(df['analysis'].unique())
    for cat in missing:
        ax0.scatter(cat, df[feature_col].iloc[0], color='white', alpha=0)

    ax0.set_xticklabels(ax0.get_xticklabels(), rotation=45, ha="right")
    ax0.set_xlabel('')
    ax0.set_ylabel('Top central TFs' if feature_col == 'gene' else 'Pathways')
    ax0.margins(x=.2, y=.05 if len(tfs) > 10 else 0.2)
    ax0.spines[['top', 'right']].set_visible(False)
    if hide_ylabels:
        ax0.set_yticklabels([])
        ax0.set_ylabel('')
    for spine in ax0.spines.values():
        spine.set_linewidth(0.5)
    ax0.get_legend().remove()
    # - Degree barplot
    if plot_centrality:
        ax1 = axes[1]
        bar_data = df.drop_duplicates(subset=feature_col)
        sns.barplot(data=bar_data, x='degree', y=feature_col, ax=ax1, color='#56B4E9', alpha=0.7, ci=None)
        ax1.set_xlabel('Centrality')
        ax1.set_ylabel('')
        ax1.set_yticks([])
        ax1.margins(**ax2_margins)
        ax1.spines[['top', 'right']].set_visible(False)
        
    if False:
        # - Place legend on the outer right of both subplots
        ordered_labels = ['Decrease in aging', 'Increase in aging', 'Decrease after treatment', 'Increase after treatment', 'Decrease in disease', 'Increase in disease']
        ordered_labels = [label for label in ordered_labels if label in palette_all.keys()]
        handles, labels = ax0.get_legend_handles_labels()

        # Create a dictionary from labels to handles
        label_handle_dict = dict(zip(labels, handles))

        # Reorder handles and labels
        ordered_handles = [label_handle_dict[label] for label in ordered_labels]
        ordered_labels = [label for label in ordered_labels]

        # Add legend
        fig.legend(ordered_handles, ordered_labels, loc='center left', bbox_to_anchor=(1.02, 0.8), frameon=False)

    
def plot_overlap(
        stats_drug_sig, 
        aging_stats_sig, 
        agreement,  # treatment effect should be 'opposite' to aging
        how='left',
        col='cell_type',
        ax=None,
        legend=True,
        figsize=(2.5, 2),
        legend_loc=(1.05, 0.5)
    ):
    

    if not pd.api.types.is_categorical_dtype(stats_drug_sig['cell_type']):
        stats_drug_sig['cell_type'] = stats_drug_sig['cell_type'].astype('category')

    included_celltypes = stats_drug_sig['cell_type'].cat.categories 

    merged = aging_stats_sig.merge(stats_drug_sig, on=['gene', col], how=how)
    merged['slope_sign'] = np.sign(merged['slope'])
    merged['slope_condition_sign'] = np.sign(merged['slope_condition'])
    merged = merged.drop_duplicates(subset=['gene', col, 'slope_sign', 'slope_condition_sign'])
    merged['trend'] = merged['slope_sign'].map({1: 'positive', -1: 'negative'})
    merged = merged[merged['cell_type'].isin(included_celltypes)]

    def compute_agreement(group):
        total = len(group)
        group = group[(~group['slope_sign'].isna()) & (~group['slope_condition_sign'].isna())]
        if agreement == 'opposite':
            agree = (group['slope_sign'] != group['slope_condition_sign']).sum()
        elif agreement == 'same':
            agree = (group['slope_sign'] == group['slope_condition_sign']).sum()
        else:
            raise ValueError("Agreement must be either 'opposite' or 'same'")
        return pd.Series({'n_total_tfs': total, 'n_agreeing_tfs': agree})

    summary = (
        merged.groupby([col, 'trend'])
        .apply(compute_agreement)
        .reset_index()
    )

    if col == 'cell_type':
        cell_types = [t for t in MAJOR_CTS if t in summary[col].unique()]
        summary[col] = pd.Categorical(summary[col], categories=cell_types, ordered=True)

    summary['trend'] = ['Increase in aging' if x == 'positive' else 'Decrease in aging' for x in summary['trend']]
    df = summary.copy()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Background bars
    offset = {'Decrease in aging': -0.22, 'Increase in aging': 0.22}
    width=0.35
    colors = {'Decrease in aging': 'tab:blue', 'Increase in aging': 'tab:red'}
    x_locs = {cat: i for i, cat in enumerate(summary[col].cat.categories)}
    for trend, offset_val in offset.items():
        df_trend = df[df['trend'] == trend]
        xpos = [x_locs[val] + offset_val for val in df_trend[col]]
        ax.bar(
            xpos,
            df_trend['n_total_tfs'],
            width=width,
            alpha=0.3,
            color=palette_trend_2[trend],
            edgecolor='black',
            linewidth=0.1,
            label=trend
        )

    for i, row in df.iterrows():
        base_x = x_locs[row[col]]
        xpos = base_x + offset[row['trend']]
        ax.bar(
            xpos,
            row['n_agreeing_tfs'],
            width=width,
            edgecolor=colors[row['trend']],
            facecolor='none',
            hatch='///',
            linewidth=1,
            zorder=1
        )

        y_pos = row['n_total_tfs']
        ax.text(
            xpos,
            y_pos + 2 + y_pos*.1*np.random.rand(),  # Random offset for better visibility,
            f"{int(100*(row['n_agreeing_tfs']/y_pos))}%",
            ha='center',
            va='bottom',
            fontsize=8
        )

    ax.set_xticks(list(x_locs.values()))
    ax.set_xticklabels(list(x_locs.keys()), rotation=45, ha='right')
    ax.set_ylabel("Number of TFs")
    ax.set_xlabel("")
    ax.margins(x=0.1, y=0.2)
    ax.spines[['right', 'top']].set_visible(False)

    # Custom legend
    handles, labels = ax.get_legend_handles_labels()
    if agreement == 'opposite':
        label = 'Opposing effect'
    elif agreement == 'same':
        label = 'Supporting effect'
    
    hatch_patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label=label)
    handles.append(hatch_patch)
    if legend:
        ax.legend(handles=handles, title='', loc=legend_loc, frameon=False)


def plot_features_vs_datasets(cell_type,
                              analysis_name,
                              datasets=DISCOVERY_COHORTS, 
                              features=None, 
                              sizes=(50, 100), 
                              top_features=15, 
                              filter_significant=True, 
                              features_dir=None,
                              show_size_legend=False,
                              plots_dir=PLOTS_DIR,
                              
                              ):

    feature_type = get_config_fa(analysis_name)['feature_type']
    n_datasets = len(datasets)
    estimated_n_features = top_features if features is None else len(features) if features is not None else top_features
    base_width = max(1.5, min(3.5, 1.0 + n_datasets * 0.2))  # Tighter width range: 1.5-3.5 instead of 1.8-4
    base_height = max(0.12, min(0.25, 0.2 - estimated_n_features * 0.002))  # Smaller height per row for many features

    # - format the data
    stats_t = retrieve_stats(analysis_name=analysis_name, cell_type=cell_type, features_dir=features_dir)
    stats_t = stats_t[stats_t['dataset'].isin(datasets)]

    if filter_significant:
        stats_sig = retrieve_sig_stats(analysis_name=analysis_name, cell_type=cell_type).drop_duplicates(subset=['gene', 'cell_type'])
        sig_genes = stats_sig['gene'].unique()
        stats_t = stats_t[stats_t['gene'].isin(sig_genes)]
    
    # - get centrality measure
    net = retrieve_net_consensus(cell_type=cell_type)
    group_col = 'target' if feature_type == 'gene_expression' else 'source'
    c = net.groupby([group_col]).size()
    c = c / c.max()
    c = c.reset_index(name='degree')
    c.rename(columns={group_col: 'gene'}, inplace=True)

    stats_t = stats_t.merge(c, on='gene', how='left')
    
    # - either find the central features and sort them or sort them based on the given features    
    if features is None:
        # - select the top features: top shared across datasets and top central
        c = c[c['gene'].isin(stats_t['gene'].unique())]
        features = c.sort_values('degree', ascending=False).head(top_features)['gene'].unique() # subset to top ones
        
        stats_t = stats_t[stats_t['gene'].isin(features)]
        stats_t = stats_t.sort_values('degree', ascending=False)
    else:
        if feature_type == 'tf_activity':
            pass
            # - check if the features are in the tf_all list
            # tf_all = np.loadtxt(f"{PRIOR_DIR}/tf_all.csv", dtype=str)
            # features = [tf for tf in features if tf in tf_all]

        # - check if the features are in the stats (remove those that are not present in at least one dataset)
        stats_t = stats_t[stats_t['gene'].isin(features)]
        features = [gene for gene in features if gene in stats_t['gene'].unique()]
        features = list(set(features))  # remove duplicates
        stats_t['gene'] = pd.Categorical(stats_t['gene'], categories=features, ordered=True)
        stats_t = stats_t.sort_values('gene')  
    stats_t['neg_log10_adj_pval'] = -np.log10(stats_t['p_value_adj'])
    stats_t['dataset'] = pd.Categorical(stats_t['dataset'], categories=datasets, ordered=True)
    stats_t['dataset'] = stats_t['dataset'].apply(lambda name: surrogate_names.get(name, name))

    if stats_t.shape[0]==0:
        raise ValueError(f'No data for {cell_type} {feature_type}')
        
    # Calculate automated layout parameters
    n_features = len(features)
    # Use logarithmic scaling for many features - more generous spacing (looser)
    if n_features <= 10:
        base_height = 0.1  # More generous height for very small number of features
    elif n_features <= 20:
        base_height = 0.16  # More generous height for small number of features
    elif n_features <= 50:
        base_height = 0.12  # More generous for medium number of features
    else:
        # For large feature sets, ensure minimum spacing between dots while keeping reasonable total height
        base_height = max(0.08, 0.20 / np.log10(n_features))  # Increased minimum to prevent dot overlap
    
    
    # Automated figure sizing with better scaling for many features - smaller width scaling
    width_factor = max(1, min(1.5, n_datasets / 8))  # Even smaller width scaling
    
    # Use square root scaling for height to prevent excessive stretching - more conservative
    if n_features <= 10:
        height_factor = 1
    elif n_features <= 50:
        height_factor = max(1, np.sqrt(n_features / 15))  # More conservative scaling
    elif n_features <= 100:
        height_factor = max(1, np.sqrt(n_features / 30))  # Even more conservative for medium-large sets
    else:
        # Very gentle scaling for large feature sets - much less aggressive
        height_factor = max(1, np.log10(n_features) * 1)  # Increased from 0.1 to 0.5 for better spacing
    fig_width = base_width * width_factor + 1.5  # Reduced legend space from 2 to 1.5
    fig_height = base_height * n_features * height_factor + 1.5  # Reduced title/label space from 2 to 1.5
        
    # Automated margin calculation - scalable based on features and datasets
    # X-axis margins: Scale with number of datasets (more datasets need tighter spacing)
    x_margin_base = 0.15  # Reduced base margin for x-axis 
    x_margin_scale = max(0.1, min(0.20, x_margin_base / np.sqrt(n_datasets)))
    
    # Y-axis margins: Scale with number of features - balanced for large sets
    y_margin_base = 0.10  # Reduced base margin for y-axis
    if n_features <= 10:
        y_margin_scale = 1
    elif n_features <= 20:
        y_margin_scale = .5
    elif n_features <= 100:
        y_margin_scale = max(0.02, min(0.10, y_margin_base / np.log10(n_features)))
    else:
        # For large feature sets, use moderate margins since we increased base_height
        y_margin_scale = .01
    
    # Additional scaling based on figure size - more balanced
    width_correction = min(1.3, fig_width / 4)  # Balanced width correction
    height_correction = min(1.3, fig_height / 10)  # Balanced height correction
    
    margins_ax1 = {
        'x': x_margin_scale * width_correction,
        'y': y_margin_scale * height_correction
    }
    margins_ax2 = {
        'x': x_margin_scale * width_correction * 0.8,  # Slightly tighter for centrality plot
        'y': y_margin_scale * height_correction
    }
    
    # Automated legend positioning based on number of features
    if n_features <= 5:
        size_legend_loc = None
        cbar_height = '15%'
        cbar_width = "60%"
        bbox_to_anchor_cbar = (1.25, -0.3, 1, 1)
        wspace = 0.15
    elif n_features <= 10:
        size_legend_loc = None
        cbar_height = '12%'
        cbar_width = "50%"
        bbox_to_anchor_cbar = (1.25, -0.4, 1, 1)
        wspace = 0.12
    elif n_features <= 20:
        size_legend_loc = None
        cbar_height = '8%'
        cbar_width = "45%"
        bbox_to_anchor_cbar = (1.2, -0.5, 1, 1)
        wspace = 0.1
    elif n_features <= 50:
        bbox_to_anchor_cbar = (1.2, -0.4, 1, 1)
        size_legend_loc = (0.95, -0.5, 1, 1)
        cbar_height = '6%'
        cbar_width = "40%"
        wspace = 0.09
    elif n_features <= 100:
        bbox_to_anchor_cbar = (1.2, -0.3, 1, 1)
        size_legend_loc = (0.95, -0.4, 1, 1)
        cbar_height = '5%'
        cbar_width = "40%"
        wspace = 0.08
    else:
        bbox_to_anchor_cbar = (1.2, -0.6, 1, 1)
        size_legend_loc = (0.97, -0.7, 1, 1)
        cbar_height = '1%'
        cbar_width = "40%"
        wspace = 0.1
        
    # Adjust grid width ratios based on data
    centrality_width = min(0.6, max(0.3, 0.4 + n_features * 0.01))
    legend_width = min(0.5, max(0.3, 0.3 + n_features * 0.005))
    width_ratios = [1, centrality_width, legend_width]
        
        
    # - main plot
    fig = plt.figure(figsize=(fig_width, fig_height))
    from matplotlib import gridspec
    gs = gridspec.GridSpec(1, 3, width_ratios=width_ratios)
    
    ax = fig.add_subplot(gs[0])
    ax_legend = fig.add_subplot(gs[-1])
    ax_legend.set_axis_off()
    
    unique_features = stats_t['gene'].unique()
    stats_t['gene'] = pd.Categorical(stats_t['gene'], categories=unique_features, ordered=True)
    ordered_features = stats_t['gene'].cat.categories  
    if stats_t['dataset'].nunique() != len(datasets):
        print( f"Only {stats_t['dataset'].nunique()} datasets are available in the stats.")   
    
    dotplot(stats_t, 
            x='dataset',
            y = 'gene',
            ax=ax, 
            ax_legend=ax_legend,
            color_col='slope', 
            size_col='neg_log10_adj_pval', 
            palette=cmap_trend, 
            show_color_legend=True, 
            show_size_legend=show_size_legend,
            alpha=1,
            size_legend_title='-Log10 p-value',
            color_legend_title='Correlation \nwith aging',
            size_legend_loc=size_legend_loc,
            bbox_to_anchor_cbar=bbox_to_anchor_cbar,
            cbar_height=cbar_height,
            cbar_width=cbar_width,
            linewidth=0.1,  
            sizes=sizes,
            size_legend_scale = 200/max(stats_t['neg_log10_adj_pval']),
            )
    ax.spines[['top', 'right']].set_visible(False)
    ax.margins(**margins_ax1)
    ax.set_ylabel('Genes')
    title = surrogate_names.get(feature_type, feature_type)
    ax.set_title(f'{title} - {cell_type}', pad=10, fontsize=10, fontweight='bold')
    
    # ------------ centrality
    c = c[c['gene'].isin(features)]
    c['gene'] = pd.Categorical(c['gene'], categories=ordered_features, ordered=True)
    ax = fig.add_subplot(gs[1])
    
    sns.barplot(
        data=stats_t,
        x='degree',
        y='gene',
        ax=ax,
        color='#56B4E9',
        alpha=0.7,
        ci=None,  # turn off seaborn's built-in error estimation
        # errorbar=('sd', stats_t['centrality_std']),  # pass your own std values
        errwidth=1.2,
        capsize=0.2
    )
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.margins(**margins_ax2)
    ax.set_xlabel('Centrality\n(out-degree)' if feature_type in ['tf_activity', 'tfa_peg'] else 'Centrality\n(in-degree)')
    ax.set_ylabel('')
    ax.set_yticks([])
    
    # Apply scalable figure-level margins and spacing
    # Calculate outer margins based on plot dimensions and content
    left_margin = max(0.08, min(0.2, 0.1 + 0.02 * np.log10(n_features)))  # More space for y-labels
    right_margin = max(0.85, min(0.95, 0.9 - 0.01 * n_datasets))  # Space for legends
    bottom_margin = max(0.1, min(0.25, 0.15 + 0.02 * np.log10(n_datasets)))  # Space for x-labels
    top_margin = max(0.9, min(0.98, 0.95 - 0.005 * n_features))  # Space for title
    
    plt.subplots_adjust(
        left=left_margin,
        right=right_margin, 
        bottom=bottom_margin,
        top=top_margin,
        wspace=wspace
    )
    if os.path.exists(plots_dir):
        file_name = f'{plots_dir}/feature_vs_datasets_{cell_type}_{feature_type}.png'
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        print(f'Saved figure to {file_name}')
    else:
        print(f'{plots_dir} is not found. Figure not saved.')

    return fig

def plot_tf_interactions_plus_target_stats(net, ax=None, show_legend=True, sizes=(20, 200), annotate_sig=True, annotate_targets=False):
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from pandas.api.types import CategoricalDtype
    cmap = plt.cm.RdYlGn  # Red = negative, Green = positive
    norm = TwoSlopeNorm(vmin=-.1, vcenter=0, vmax=.1)
    
    net['dataset'] = net['dataset'].apply(lambda name: surrogate_names.get(name, name))
    net['slope_direction'] = net['slope'].apply(lambda x: 'Increase in aging' if x > 0 else 'Decrease in aging')

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    # Sort by target alphabetically
    sns.scatterplot(
        data=net,
        x='target',
        y='dataset',
        hue='weight',
        palette=cmap,
        hue_norm=norm,
        style='slope_direction',
        markers={'Increase in aging': '^', 'Decrease in aging': 'v'},
        ax=ax,
        s = 100,
        alpha=0.7,
        legend=False  # Suppress default legend
        )
    ax.set_ylabel('Cohort', fontsize=12, labelpad=10)
    ax.set_xlabel('Targets', fontsize=12, labelpad=10)
    ax.margins(x=0.05, y=0.2)
    plt.xticks(rotation=90)
    if annotate_targets:
        tf_all = np.loadtxt(f"{PRIOR_DIR}/tf_all.csv", dtype=str)
        # Set tick labels with color
        plt.draw()  # ensures tick labels are populated

        # Loop through the tick labels and modify their color
        for label in ax.get_xticklabels():
            label_text = label.get_text()
            if label_text in tf_all:
                label.set_color('#56B4E9')
    # Custom legend handles
    if show_legend:
        style_legend = [
            Line2D([0], [0], marker='^', color='w', label='Increase in aging', markerfacecolor='gray', markersize=8),
            Line2D([0], [0], marker='v', color='w', label='Decrease in aging', markerfacecolor='gray', markersize=8)
        ]

        weight_values = [net['weight'].min(), 0, net['weight'].max()]
        color_legend = [
            Line2D([0], [0], marker='o', color='w', label=f'Regulation: {w:.2f}',
                markerfacecolor=cmap(norm(w)), markersize=10) for w in weight_values
        ]

        size_values = np.percentile(net['neg_log10_adj_pval'], [25, 50, 75])
        size_legend = [
            Line2D(
                [0], [0],
                marker='o',
                color='none',  # no line
                markeredgecolor='none',  # no border
                markerfacecolor='gray',
                label=f'-log10(p): {s:.1f}',
                markersize=np.interp(s, [min(size_values), max(size_values)], [6, 14])
            )
            for s in size_values
        ]

        # Combine and place legends
        spacer = Line2D([0], [0], linestyle="none", label="")

        # all_handles = style_legend + [spacer] + color_legend + [spacer] + size_legend
        all_handles = style_legend + [spacer] + color_legend

        ax.legend(
            handles=all_handles,
            loc='center left',
            bbox_to_anchor=(1.01, 0.5),
            borderaxespad=0,
            title='',
            frameon=False
        )
    
    # - annotate significant targets
    if annotate_sig:
        pval_threshold = 1.4
        x_vals = net['target'].astype('category').cat.codes.values
        y_vals = net['dataset'].astype('category').cat.codes.values

        for (x, y), (_, row) in zip(zip(x_vals, y_vals), net.iterrows()):
            if row['neg_log10_adj_pval'] >= pval_threshold:
                ax.text(x, y + 0.1, '*', ha='center', va='center', alpha=0.9, 
                        fontsize=8, weight='bold', color='black', zorder=10)


def wrapper_flesh_out_tf_interactions(datasets, cell_type, tf, data_type='bulk', n_top=10, keep_sig_only=False, sizes=(20, 100), ax=None, show_legend=True):
    from hiara.src.feature_association.helper import retrieve_stats
    from hiara.src.feature_association.plots import plot_tf_interactions_plus_target_stats
    # - get the net for different datasets
    top_targets = []
    net_store = []
    for dataset in datasets:
        net = retrieve_net(dataset, cell_type)[['source', 'target', 'weight', 'cell_type']]
        net = net[net['source'] == tf].copy()
        top_targets.append(net.sort_values(by='weight', ascending=False, key=abs).head(n_top)['target'].tolist())
        net['dataset'] = dataset
        net_store.append(net)
    net = pd.concat(net_store)
    top_targets = np.unique(np.concatenate(top_targets))
        
    # - get the stats of targets per dataset 
    stats_targets = retrieve_stats(data_type, feature_type='gene_expression', cell_type=cell_type, condition='healthy')
    stats_targets = stats_targets[stats_targets['dataset'].isin(datasets)]
    net_stats = net.merge(stats_targets[['dataset', 'target', 'p_value_adj', 'slope']], on=['dataset', 'target'], how='left')
    net_stats = net_stats[~net_stats['p_value_adj'].isna()]
    net_stats['neg_log10_adj_pval'] = -np.log10(net_stats['p_value_adj'])

    net_stats = net_stats[net_stats['target'].isin(top_targets)]
    if keep_sig_only:
        net_stats = net_stats[net_stats['neg_log10_adj_pval'] > 1.4]
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(len(top_targets)*.2 + 1, len(datasets)*.2 + 1))

    plot_tf_interactions_plus_target_stats(net_stats.copy(), ax=ax, show_legend=show_legend, sizes=sizes)


def plot_overall_heatmap(stats_all, 
                        first_col='cell_type', first_col_palette=None,
                        second_col='dataset', second_col_palette=None,
                        figsize=(6, 8), 
                        sig_dots_y_offset=.8,
                        map_names={},
                        bbox_to_anchor=(1.1, 1),
                        bbox_to_anchor_col2=(1.1, .75),
                        bbox_to_anchor_col1=(1.1, 0.4),
                        trend_colors = ['#B0BF1A', '#E52B50'],
                        trend_names = ['Decrease in aging', 'Increase in aging'],
                        dendrogram_visible=True,
                        draw_legend=True

                        ):
    # format the data
    first_col_unique_values = stats_all[first_col].cat.categories
    second_col_unique_values = stats_all[second_col].cat.categories
    
    stats_all = stats_all[stats_all[second_col].isin(second_col_unique_values)]
    stats_all['trend_int'] = stats_all['slope'].map(lambda value: 1 if value > 0 else (-1 if value < 0 else 0))
    multi_index = pd.MultiIndex.from_product([first_col_unique_values, second_col_unique_values], names=[first_col, second_col])
    pivot_df = stats_all.pivot(index='gene', columns=[first_col, second_col], values="trend_int")
    df_plot = pivot_df.reindex(columns=multi_index).fillna(0)

    # - format the is_sig if needed
    if 'is_significant' in stats_all.columns:
        sig_df = stats_all.pivot(index='gene', columns=[first_col, second_col], values="is_significant")
        sig_df = sig_df.reindex(columns=multi_index).fillna(0)
    else:
        sig_df = None

    # prepare for plot
    col_colors = pd.DataFrame({
        map_names.get(second_col, second_col): [second_col_palette.get(analysis, "gray") for _, analysis in df_plot.columns],
        map_names.get(first_col, first_col): [first_col_palette.get(ct, "lightgray") for ct, _ in df_plot.columns]
    }, index=df_plot.columns)

    row_linkage = linkage(df_plot, method='ward')
    # print(palette_trend)
    cmap = ListedColormap([trend_colors[0], 'white', trend_colors[1]])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = BoundaryNorm(bounds, cmap.N)

    g = sns.clustermap(
        df_plot,
        row_linkage=row_linkage,
        col_cluster=False,
        row_cluster=True,
        cmap=cmap,
        norm=norm,
        col_colors=col_colors,
        linewidths=1,
        alpha=.8,
        linecolor=None,
        figsize=figsize,
    )
    g.ax_row_dendrogram.set_visible(dendrogram_visible)
    # if True: # tick labels
    #     g.ax_heatmap.set_xticklabels([])


    g.cax.set_visible(False)
    g.ax_heatmap.set_yticks([])
    g.ax_heatmap.set_ylabel('', fontsize=10, labelpad=5)
    
    g.ax_heatmap.set_xticks([])
    g.ax_heatmap.set_xlabel('', fontsize=12, labelpad=15)

    # - add sig if given
    if sig_df is not None:
        row_order = g.dendrogram_row.reordered_ind
        col_order = list(df_plot.columns)  # Column order stays the same since col_cluster=False

        cell_height = g.ax_heatmap.get_position().height / len(row_order)
        cell_width = g.ax_heatmap.get_position().width / len(col_order)
        # Loop through and add asterisks for significant values
        for i, row_idx in enumerate(row_order):
            for j, col in enumerate(col_order):
                # Extract cell_type and gender from col (tuple format)
                col1, col2 = col
                is_significant = sig_df.loc[sig_df.index[row_idx], (col1, col2)]
                
                # Check if the value is significant
                if is_significant:
                    y_coord = i - sig_dots_y_offset
                    x_coord = (j + 0.5) 
                    
                    g.ax_heatmap.text(
                        x_coord, y_coord,
                        '.',
                        color='black', ha='center', va='center', fontsize=8, fontweight='bold'
                    )
    # Legends
    if draw_legend:
        datasets_legend = [Patch(color=second_col_palette[label], label=map_names.get(label, label)) for label in second_col_unique_values]
        celltype_legend = [Patch(color=first_col_palette[label], label=map_names.get(label, label)) for label in first_col_unique_values]
        trend_legend = [Patch(color=color, label=label, alpha=.8) for label, color in zip(trend_names, trend_colors)]
        legend_datasets = g.ax_heatmap.legend(
            handles=datasets_legend,
            title=map_names.get(second_col, second_col),
            bbox_to_anchor=bbox_to_anchor_col2, 
            loc='upper left',
            fontsize=9,
            title_fontsize=9,
            frameon=False
        )
        legend_datasets.get_title().set_fontweight('bold')  

        legend_celltypes = g.ax_heatmap.legend(
            handles=celltype_legend,
            title=map_names.get(first_col, first_col),
            bbox_to_anchor=bbox_to_anchor_col1,
            loc='upper left',
            fontsize=9,
            title_fontsize=9,
            frameon=False
        )
        legend_celltypes.get_title().set_fontweight('bold') 

        legend_trend = g.ax_heatmap.legend(
            handles=trend_legend,
            title="Trend",
            bbox_to_anchor=bbox_to_anchor,
            loc='upper left',
            fontsize=9,
            title_fontsize=9,
            frameon=False
        )
        legend_trend.get_title().set_fontweight('bold')  


        g.ax_heatmap.add_artist(legend_celltypes)
        g.ax_heatmap.add_artist(legend_datasets) 
    # Tighten layout to reduce whitespace
    # plt.subplots_adjust(top=1.1)
    # plt.show()


def plot_joint_scatter(stats_all, col='cell_type', vars=['CD4T', 'CD8T'], annotate=True, figsize=(2.5, 3), ax=None):
    # - plot
    xy_vars = [f'{v}_pval' for v in vars]
    trend_vars = [f'{v}_trend' for v in vars]

    stats_all_table = stats_all.pivot(index='gene', columns=col, values='neg_log10_adj_pval').reset_index().fillna(0)
    stats_all_trend = stats_all.pivot(index='gene', columns=col, values='trend').reset_index()
    stats_all_table = stats_all_table.merge(stats_all_trend, on='gene', suffixes=('_pval', '_trend'))
    stats_all_table[trend_vars] = stats_all_table[trend_vars].fillna('Inconsistent')

    stats_all_table['trend'] = stats_all_table[trend_vars].apply(
                            lambda x: 'Increase in aging' if (x[trend_vars[0]]=='Increase in aging' and x[trend_vars[1]]=='Increase in aging') else ('Decrease in aging' if (x[trend_vars[0]]=='Decrease in aging' and x[trend_vars[1]]=='Decrease in aging') else 'Inconsistent') , axis=1)
    stats_all_table['trend'] = stats_all_table['trend'].astype(CategoricalDtype(categories=['Increase in aging', 'Decrease in aging', 'Inconsistent'], ordered=True))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    scatter = sns.scatterplot(
        data=stats_all_table, 
        x=xy_vars[0], y=xy_vars[1], 
        hue='trend',  # Use trend as color
        palette= palette_trend,
        s=50,  # Adjust point size
        edgecolor='black', 
        linewidth=0.2,
        alpha=0.5,
        ax=ax
    )

    # Improve labels and colorbar
    ax.set_xlabel(vars[0] + '\n' + r'($- \log_{10} p$)')
    ax.set_ylabel(vars[1] + '\n' + r'($- \log_{10} p$)')
    ax.axvline(x=1.4, color=colors_blind[0], linestyle='--')
    ax.axhline(y=1.4, color=colors_blind[0], linestyle='--')

    ax.margins(x=0.1, y=0.1)
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend(loc=(1.1, 0.2), title='Trend', frameon=False)
    
    if annotate:
        quantile = .8
        high_tf_points = stats_all_table[(stats_all_table[xy_vars[0]] > stats_all_table[xy_vars[0]].quantile(quantile))&
                                                (stats_all_table[xy_vars[1]] > stats_all_table[xy_vars[1]].quantile(quantile))
                                            ]
        
        inconsistent_tfs = stats_all_table[stats_all_table['trend'] == 'Inconsistent']

        df_to_annotate = pd.concat([inconsistent_tfs, high_tf_points], axis=0)
        
        x_offset = 10*np.asarray([-1, -1, 0, -1, 1, .5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0])
        y_offset = 10*np.asarray([ -1,  1, 1, -1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0])

        ii = 0
        for idx, row in df_to_annotate.iterrows():
            
            # Adjust the annotation position slightly away from the point
            ax.annotate(
                row['gene'], 
                xy=(row[xy_vars[0]], row[xy_vars[1]]), 
                xytext=(row[xy_vars[0]]  + x_offset[ii], row[xy_vars[1]] + y_offset[ii]),  # Adjust this value for distance
                textcoords='data',
                color='black', 
                fontsize=8, 
                ha='left', va='top',
                arrowprops=dict(arrowstyle="->", color='black', lw=0.5)  # Arrow pointing to the point
            )
            ii += 1
    # plt.show()

def heatplot_age_trend(mean_expr, cmap="viridis", cbar_title="Gene expression", y_label="Genes", figsize=(2.5, 3), 
                ax=None, show_cbar=True, cbar_kws={
                                        "shrink": 1,
                                        "aspect": 10,       # Lower values = thicker colorbar (default is ~20)
                                        "fraction": 0.1    # Controls the width space the cbar takes in the figure
                                    }):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    # Plot heatmap
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(mean_expr, cmap=cmap, cbar=show_cbar, 
                cbar_kws=cbar_kws,
                 ax=ax)
    ax.set_yticks(np.arange(mean_expr.shape[0]) + 0.5)
    ax.set_yticklabels(mean_expr.index, rotation=0)

    # Modify the colorbar
    if show_cbar:
        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel(cbar_title, rotation=90, labelpad=5)

        # Set ticks at the min and max values of the colorbar
        vmin, vmax = cbar.vmin, cbar.vmax
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels(['0', '1'], rotation=0, fontsize=7)

    # Labels and formatting
    ax.set_xlabel("Age")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

def heamap_overview_cell_types(stats_all, 
                               palette, 
                                map_names={}, 
                                slope_col='slope',
                                main_col='cell_type', 
                                figsize=(6, 8), 
                                sig_dots_y_offset = 0.5, 
                                annotate_x_ticks=True, 
                                dendrogram_visible=True,
                                show_legend=True,
                                show_dots=False
                                ):

    from hiara.src.config import palette_major_cts, surrogate_names
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from scipy.cluster.hierarchy import linkage
    from matplotlib.patches import Patch

    cell_types = stats_all[main_col].cat.categories

    
    stats_all['trend_int'] = stats_all[slope_col].map(lambda value: 1 if value > 0 else (-1 if value < 0 else 0))
    pivot_df = stats_all.pivot(index='gene', columns=main_col, values='trend_int').fillna(0)
    
    pivot_df = pivot_df.reindex(columns=cell_types).fillna(0)


    # - color map
    cols_names = pivot_df.columns.map(lambda name: mapping_minor_2_major.get(name, name))
    col_colors = [palette_major_cts[name] for name in cols_names]

    if 'trend' not in stats_all.columns:
        raise ValueError("The 'trend' column is missing in stats_all DataFrame.")
    trends = stats_all['trend'].unique()
    palette = {key: palette[key] for key in trends}
    palette_values = list(palette.values())
    
    if len(palette_values) < 2:
        raise ValueError(f"Expected at least 2 trends for colormap, but got {len(palette_values)}: {trends}")
    
    cmap = ListedColormap([palette_values[0], 'white', palette_values[1]])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = BoundaryNorm(bounds, cmap.N)
    assert not pivot_df.isnull().values.any(), "NaNs found in pivot_df"
    g = sns.clustermap(
        pivot_df,
        row_linkage=linkage(pivot_df, method='ward'),
        col_cluster=False,
        row_cluster=True,
        cmap=cmap,
        # norm=norm,
        col_colors=col_colors,
        linewidths=1,
        alpha=.8,
        linecolor=None,
        figsize=figsize,
    )
    g.ax_row_dendrogram.set_visible(dendrogram_visible)

    g.cax.set_visible(False)
    g.ax_heatmap.set_yticks([])
    g.ax_heatmap.set_ylabel('', fontsize=10, labelpad=5)
    new_labels = [surrogate_names.get(label.get_text(), label.get_text()) for label in g.ax_heatmap.get_xticklabels()]
    if annotate_x_ticks:
        g.ax_heatmap.set_xticklabels(new_labels, rotation=90)  # or any angle you prefer
    else:
        g.ax_heatmap.set_xticks([])

    g.ax_heatmap.set_xlabel('', fontsize=12, labelpad=15)


    # - format the is_sig if needed
    if 'is_significant' in stats_all.columns:
        sig_df = stats_all.pivot(index='gene', columns='cell_type', values="is_significant").fillna(0)
        sig_df = sig_df.reindex(columns=cell_types)
    else:
        sig_df = None
    if show_dots:
        row_order = g.dendrogram_row.reordered_ind
        col_order = list(cell_types)  # Column order stays the same since col_cluster=False
        for i, row_idx in enumerate(row_order):
            for j, col in enumerate(col_order):
                is_significant = sig_df.loc[sig_df.index[row_idx], col]
                if is_significant:
                    y_coord = i - sig_dots_y_offset
                    x_coord = (j + 0.5) 
                    g.ax_heatmap.text(
                        x_coord, y_coord,
                        '.',
                        color='black', ha='center', va='center', fontsize=8, fontweight='bold'
                    )
    if show_legend:
        celltype_legend = [Patch(color=palette_major_cts[label], label=map_names.get(label, label)) for label in cell_types]
        trend_legend = [Patch(color=color, label=label, alpha=.8) for label, color in palette.items()]

        legend_celltypes = g.ax_heatmap.legend(
            handles=celltype_legend,
            title=map_names.get(main_col, main_col),
            bbox_to_anchor=(1.1, 0.6),
            loc='upper left',
            fontsize=9,
            title_fontsize=9,
            frameon=False
        )
        legend_celltypes.get_title().set_fontweight('bold') 

        legend_trend = g.ax_heatmap.legend(
            handles=trend_legend,
            title="Trend",
            bbox_to_anchor=(1.1, 1),
            loc='upper left',
            fontsize=9,
            title_fontsize=9,
            frameon=False
        )
        legend_trend.get_title().set_fontweight('bold')  


        g.ax_heatmap.add_artist(legend_celltypes)

def plot_trends(top_tfs, datasets, data_dict, palette, cell_type, surrogate_names={}, stats_df=None, is_expression=False, y_label='TF Activity score'):
    """
    Plots transcription factor (TF) activity/expression trends across datasets.

    Parameters:
    - top_tfs: list of top transcription factors to plot
    - datasets: list of dataset names
    - data_dict: dictionary containing either tf_acts or adata objects
    - palette: dictionary mapping datasets to colors
    - cell_type: cell type to include in the title
    - is_adata: if True, expects `data_dict` to contain AnnData objects instead of DataFrames
    """
    
    for tf in top_tfs:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        legend_handles = []  # Store handles for the legend

        for dataset in datasets:
            
            adata = data_dict[dataset].to_memory()
            mask_tf = adata.var_names == tf
            adata_sub = adata[:, mask_tf]
            assert adata_sub.shape[1]>0, f"TF {tf} not found in dataset {dataset}"

            ages = adata_sub.obs['age'].values
            X = adata_sub.X.todense().A if hasattr(adata_sub.X, 'todense') else adata_sub.X
            expression = X.flatten() 
            cell_count = adata_sub.obs['cell_count'].values
            
            # expression = expression / expression[0]

            cell_count_n = cell_count / max(cell_count)

            print(ages.shape, expression.shape, cell_count_n.shape)
            # Scatter plot
            ax.scatter(
                ages, expression, 
                color=palette[dataset], 
                alpha=0.4,  
                linewidth=1,
                s=cell_count_n * 50
            )

            # Fit linear regression
            if len(ages) > 1:
                age_range = np.linspace(min(ages), max(ages), 100)
                spearman_corr, spearman_p = spearmanr(ages, expression)
                slope, intercept, r_value, p_value, _ = linregress(ages, expression)
                r2 = r_value**2
                # - correct for multiple testing
                if stats_df is not None:
                    stats_df_sub = stats_df[(stats_df['gene'] == tf) & (stats_df['dataset'] == dataset)]
                    p_value_adj = stats_df_sub.loc[:, 'meta_p_adj'].values[0]
                    p_value_adj = min([p_value_adj, 1])  # Ensure p-value is not greater than 1
                else:
                    n_tests = len(top_tfs)*len(datasets)
                    p_value_adj = p_value * n_tests
                # Plot fitted line
                fitted_line = slope * age_range + intercept
                # ax.plot(age_range, fitted_line, color=palette[dataset], linestyle='-', linewidth=2)

                # Create legend handle with both R² and Spearman ρ
                dataset_name = surrogate_names.get(dataset, dataset)
                legend_label = '    ' + dataset_name + f' ({slope.round(2)})'+'\n' + r' ($p-value$=' + "{:.2e}".format(p_value) + ")"

                handle = mpatches.Patch(color=palette[dataset], label=legend_label)
                legend_handles.append(handle)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xlabel('Age')
        ax.set_ylabel(y_label)
        ax.set_title(f'{cell_type}: {tf}', pad=20)

        # Add properly formatted legend
        ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)

def plot_activation_vs_expression(df_combined, 
                                  col_x = 'signed_-log10_pval_exp', 
                                  col_y = 'signed_-log10_pval', 
                                  y_label = "Activation\nsigned -log10(p adj)",
                                  x_label = "Expression\nsigned -log10(p adj)",
                                  figsize=(4, 2.7)):
    import matplotlib.patches as mpatches
    cell_types_local = df_combined["cell_type"].unique()
    n_cell_types = len(cell_types_local)
    for i, cell_type in enumerate(cell_types_local):
        df_cell_type = df_combined[df_combined["cell_type"] == cell_type]
        fig, ax = plt.subplots(1, 1, figsize=figsize, sharey=False, sharex=False)
        df_dataset = df_cell_type
        df_dataset['dataset'] = df_dataset['dataset'].map(surrogate_names)
        assert df_dataset.shape[0]>0, f"No data for {cell_type} in {datasets[j]}"
        sns.scatterplot(
            data=df_dataset,
            x=col_x,
            y=col_y,
            palette=palette_datasets_pretty,  # Use the consistent color mapping
            # size="centrality",
            s=10,
            hue="dataset",
            # sizes=(20, 100),
            edgecolor=None,
            alpha=0.7,
            ax=ax,
        )
        top_tfs = df_dataset.sort_values(by='p_value_adj_target', ascending=False).head(2)
        # compute small offsets relative to axis ranges
        x_range = df_dataset[col_x].max() - df_dataset[col_x].min()
        y_range = df_dataset[col_y].max() - df_dataset[col_y].min()
        x_offset = 0.2 * x_range

        for _, row in top_tfs.iterrows():
            ax.text(
                row[col_x] + x_offset,
                row[col_y] + np.random.rand() * .1 *  y_range,
                row['gene'],  # assumes TF names are in column 'gene'
                fontsize=7,
                ha='left',    # anchor text to the left since we shift right
                va='bottom',  # anchor text above since we shift up
                color='black'
            )
        # Annotate top genes with the highest activation significance
        xmin, xmax = df_dataset[col_x].min(), df_dataset[col_x].max()
        ymin, ymax = df_dataset[col_y].min(), df_dataset[col_y].max()
        global_min = min(xmin, ymin)
        global_max = max(xmax, ymax)
        placed_positions = []
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label, labelpad=15)        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_aspect("equal", adjustable="datalim")
        padding = 0.15 * (global_max - global_min)
        sig_threshold = 1.4
        ax.margins(x=0.01, y=0.05)
        linewidth = .5
        alpha = .4
        ax.axvline(sig_threshold, linestyle="--", color="red", alpha=alpha, linewidth=linewidth)  # Vertical
        ax.axvline(-sig_threshold, linestyle="--", color="red", alpha=alpha, linewidth=linewidth)  # Vertical
        ax.axhline(sig_threshold, linestyle="--", color="red", alpha=alpha, linewidth=linewidth)  # Horizontal
        ax.axhline(-sig_threshold, linestyle="--", color="red", alpha=alpha, linewidth=linewidth)  # Horizontal
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, fontsize=10, title='Dataset', title_fontsize=10)


