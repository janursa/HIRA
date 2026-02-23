#!/usr/bin/env python
"""
Plotting functions for condition analysis.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.api.types import CategoricalDtype
from hiara.src.config import surrogate_names, MAJOR_CTS, SUB_CTS, palette_major_cts, palette_sub_cts
from hiara import retrieve_sig_stats, retrieve_stats, retrieve_feature_data
from hiara.src.feature_association.plots import heatplot_age_trend, heamap_overview_cell_types, plot_tf_act_central_tfs
from hiara.src.utils.util import retrieve_net_consensus
from hiara.src.feature_association.helper_condition import format_tf_activity_for_disease_trend_plot, get_condition_palette


def plot_healthy_disease_trend(dataset, analysis_name, cell_type, case_tf, condition_col, ax=None, normalize=False):
    """
    Plot healthy vs disease trend for a TF across age.
    
    Parameters
    ----------
    dataset : str
        Dataset name
    analysis_name : str
        Analysis configuration name
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
        dataset=dataset, analysis_name=analysis_name, cell_type=cell_type, tf=case_tf, condition_col=condition_col
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


def plot_overview_heatmap(stats, args):
    """Generate overview heatmap of minor cell types."""
    print("Generating overview heatmap...")
    
    dataset = args.dataset
    analysis_type = args.analysis_type
    granularity = args.granularity
    output_dir = args.output_dir

    if len(stats) == 0:
        print(f"Warning: No stats to plot. Skipping overview heatmap.")
        return

    if 'comparison' in stats.columns:
        if len(stats['comparison'].unique()) > 1:
            raise ValueError("Overview heatmap only supports single comparison at a time.")
    if 'age_group' in stats.columns:
        if len(stats['age_group'].unique()) > 1:
            raise ValueError("Overview heatmap only supports single age group at a time.")

    slope_col = 'slope'  
    if 'Major' in granularity:
        categories = MAJOR_CTS
        palette_cols = palette_major_cts
    elif 'Sub' in granularity:
        categories = SUB_CTS
        palette_cols = palette_sub_cts
    else:
        raise ValueError(f"Unexpected granularity: {granularity}. Expected 'Major' or 'Sub'.")
    if not set(stats['cell_type'].unique()).issubset(set(categories)):
        missing = set(stats['cell_type'].unique()) - set(categories)
        print(f"Warning: The following cell types in stats are not in the expected categories and will be ignored: {missing}")
        raise ValueError("Unexpected cell types found in stats.")
    stats['cell_type'] = pd.Categorical(stats['cell_type'], categories=categories, ordered=True)
    palette = get_condition_palette(analysis_type)
    heamap_overview_cell_types(
            stats, 
            slope_col=slope_col, 
            palette=palette, 
            figsize=(2, 3), 
            sig_dots_y_offset=3, 
            annotate_x_ticks=False, 
            map_names={'cell_type': 'Cell type'},
            dendrogram_visible=False, 
            show_legend=True,
            palette_cols=palette_cols
        )    
    output_path = os.path.join(output_dir, f'overview_{dataset}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
    plt.close()
    print(f"  Saved: {output_path}")


def wrapper_plot_central_tfs_condition(stats, cell_types, group_col, args):
    """Plot aging vs perturbation comparison."""
    aging_stats_sig = retrieve_sig_stats(analysis_name=args.analysis_name).drop_duplicates(subset=['cell_type', 'gene'])
    aging_stats_sig['analysis'] = 'Age-associated'
    stats['analysis'] = stats[group_col]    
    groups = stats[group_col].unique()
    palette_all = get_condition_palette(args.analysis_type)
    for cell_type in cell_types:
        stats_aging_t = aging_stats_sig[aging_stats_sig['cell_type'] == cell_type].copy()
        aging_tfs = stats_aging_t['gene'].unique()
        stats_t = stats[stats['cell_type'] == cell_type].copy()
        stats_t = stats_t[stats_t['gene'].isin(aging_tfs)]
        # Merge
        stats_all = pd.concat([stats_aging_t, stats_t])        
        stats_all['analysis'] = pd.Categorical(
            stats_all['analysis'], 
            categories=['Age-associated'] + list(groups), 
            ordered=True
        )
        # Add centrality information
        net = retrieve_net_consensus(cell_type=cell_type)
        c_df = net.groupby('source').size().reset_index(name='degree')
        stats_all = stats_all.merge(c_df, left_on='gene', right_on='source', how='left')
        
        top_tfs = stats_all.drop_duplicates(subset=['cell_type', 'gene']).sort_values(
            'degree', ascending=False
        ).head(args.top_aging_tfs)['gene'].unique()

        stats_all = stats_all[stats_all['gene'].isin(top_tfs)].sort_values('degree', ascending=False)
        
        stats_all['degree'] = stats_all['degree'].div(stats_all['degree'].max())  # Normalize degree
        
        plot_tf_act_central_tfs(
            stats_all, 
            all_groups=groups, 
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


def plot_disease_case_tfs(args, cell_type, case_tfs):
    """Plot healthy vs disease trends for specific genes."""
    # Determine condition column based on dataset
    if not cell_type:
        print(" This function only supports single cell type at a time.")
        raise ValueError("Please specify a single cell type using --cell_type.")
    condition_col = 'condition'
   
    for i, case_tf in enumerate(case_tfs):
        fig, ax = plt.subplots(1, 1, figsize=(2, .6), sharey=False, sharex=False)
        
        plot_healthy_disease_trend(
            dataset=args.dataset, 
            analysis_name=args.analysis_name, 
            cell_type=cell_type, 
            case_tf=case_tf, 
            condition_col=condition_col, 
            ax=ax
        )
        
        ax.set_title(f'{case_tf}', fontsize=10, pad=10, weight='bold')
        if i == 0:
            ax.set_xlabel('')
        
        output_path = os.path.join(args.output_dir, f'healthy_disease_trend_{case_tf}_{cell_type}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
        plt.close()
        print(f"  Saved: {output_path}")


def plot_ctr_condition_donor_level(args, cell_types):
    """Plot donor-level perturbation effects for case genes."""    
    stats = retrieve_stats(
        dataset=args.dataset, analysis_name=args.analysis_name)
    comparisons = stats['comparison'].unique()
    aggregate_per_donor = args.aggregate_per_donor if hasattr(args, 'aggregate_per_donor') else False
    case_tfs = args.case_tfs
    
    for cell_type in cell_types:
        adata = retrieve_feature_data(dataset=args.dataset, analysis_name=args.analysis_name, cell_type=cell_type)
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
                
                donors = sorted(plot_data['donor_id'].unique())
                donor_map = {donor: f'Donor {i+1}' for i, donor in enumerate(donors)}
                plot_data['donor_id'] = plot_data['donor_id'].map(donor_map)
                
                # Set categorical order for donor_id to ensure proper legend sorting
                donor_order = [f'Donor {i+1}' for i in range(len(donors))]
                plot_data['donor_id'] = pd.Categorical(plot_data['donor_id'], categories=donor_order, ordered=True)
                
                if aggregate_per_donor:
                    plot_data = plot_data.groupby(['donor_id', 'condition'])[case_tf].mean().reset_index()
                    # Reapply categorical after groupby
                    plot_data['donor_id'] = pd.Categorical(plot_data['donor_id'], categories=donor_order, ordered=True)

                # Plot points
                sns.stripplot(
                    data=plot_data,
                    x='condition',
                    y=case_tf,
                    hue='donor_id',
                    ax=ax,
                    dodge=False,
                    jitter=False if aggregate_per_donor else True,
                    alpha=0.7,
                    size=4,
                    palette='tab10'
                )
                
                # Add lines connecting donors if aggregated
                if aggregate_per_donor:
                    # Extract colors from the legend handles (seaborn's actual color mapping)
                    handles, labels = ax.get_legend_handles_labels()
                    donor_colors = {}
                    for label, handle in zip(labels, handles):
                        if hasattr(handle, 'get_facecolor'):
                            donor_colors[label] = handle.get_facecolor()[0]
                        elif hasattr(handle, 'get_color'):
                            donor_colors[label] = handle.get_color()
                        else:
                            donor_colors[label] = handle.get_markerfacecolor()
                    
                    conditions_list = plot_data['condition'].cat.categories.tolist()
                    if len(conditions_list) == 2:
                        ctr_name, treatment_name = conditions_list[0], conditions_list[1]
                        for donor in plot_data['donor_id'].unique():
                            donor_data = plot_data[plot_data['donor_id'] == donor]
                            if len(donor_data) == 2:
                                ctr_val = donor_data[donor_data['condition'] == ctr_name][case_tf].values
                                treat_val = donor_data[donor_data['condition'] == treatment_name][case_tf].values
                                if len(ctr_val) > 0 and len(treat_val) > 0:
                                    ax.plot([0, 1], [ctr_val[0], treat_val[0]], 
                                           color=donor_colors[donor], alpha=0.5, linewidth=1, zorder=0)

                # rotate x-axis labels
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.margins(x=0.3, y=0.3)
                # Add p-value annotation with bracket
                y_max = plot_data[case_tf].max()
                y_min = plot_data[case_tf].min()
                y_range = y_max - y_min
                y_pos = y_max + 0.4 * y_range
                bracket_height = 0.02 * y_range
                
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
                
                # Draw bracket: left vertical line, horizontal line, right vertical line
                bracket_y = y_pos - bracket_height
                ax.plot([0, 0], [bracket_y - bracket_height, bracket_y], 'k-', linewidth=1)  # Left bracket
                ax.plot([0, 1], [bracket_y, bracket_y], 'k-', linewidth=1)  # Horizontal line
                ax.plot([1, 1], [bracket_y - bracket_height, bracket_y], 'k-', linewidth=1)  # Right bracket
                ax.spines[['top', 'right']] .set_visible(False)
                
                ax.set_xlabel('')
                ax.set_ylabel('TF activity' if i == 0 else '', fontsize=10)
                ax.set_title(case_tf, fontsize=10, pad=5)
                if i == len(case_tfs) - 1:
                    bbox_to_anchor = [1.05, 1 if len(donors) <= 8 else 1.2]
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
                ax.set_yticks([])
               

            output_path = os.path.join(
                args.output_dir, 
                f'case_donors_{comparison}_{cell_type}.png'
            )
            output_path = output_path.replace(' ', '_').replace('(', '_').replace(')', '_').replace(':', '_')
            plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
            plt.close()
            print(f"    Saved: {output_path}")


def plot_pathway_analysis(stats_sig, args):
    """Run and plot pathway analysis."""
    dataset = args.dataset
    output_dir = args.output_dir
    # Choose pathway analysis method based on analysis type
    from hiara.src.pathway_analysis.util import gsea_func
    from hiara.src.pathway_analysis.plots import plot_pathway_gsea
    
    print("\n  Running GSEA enrichment analysis...")
    trends = stats_sig['trend'].unique()
    palette = get_condition_palette(args.analysis_type)
    palette = {k: v for k, v in palette.items() if k in trends}
    # Run GSEA
    pathway_scores = gsea_func(
        stats_sig,
        pvalue_col='p_value_adj',
        gene_sets=['MSigDB_Hallmark_2020'],
        feature_col='gene'
    )
    
    if pathway_scores is not None and len(pathway_scores) > 0:
        print(f"  Found {len(pathway_scores)} significant pathways")
        
        # Plot combined GSEA results
        plot_pathway_gsea(pathway_scores, palette=palette)
        
        output_path = os.path.join(output_dir, f'{dataset}_pathway_gsea.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
        plt.close()
        print(f"  Saved GSEA plot: {output_path}")
    else:
        print("  No significant pathways found")
