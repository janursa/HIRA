
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
from hiara import retrieve_feature_data, retrieve_sig_stats, retrieve_stats
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
    heamap_overview_cell_types,
    plot_tf_act_central_tfs,
    plot_aging_overlap,
    plot_directional_consistency_scatter
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
        palette = palette_disease_effect
    elif analysis_type == 'perturbation':
        palette = palette_treatment
    elif analysis_type == 'aging':
        palette = palette_trend
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
    palette_all = {**palette_trend, **palette}
    return palette_all
    

def plot_overview_heatmap(stats, args):
    """Generate overview heatmap of minor cell types."""
    print("Generating overview heatmap...")
    
    dataset = args.dataset
    analysis_type = args.analysis_type
    output_dir = args.output_dir

    if 'comparision' in stats.columns:
        if len(stats['comparison'].unique()) > 1:
            raise ValueError("Overview heatmap only supports single comparison at a time.")
    if 'age_group' in stats.columns:
        if len(stats['age_group'].unique()) > 1:
            raise ValueError("Overview heatmap only supports single age group at a time.")

    slope_col = 'slope'  
    stats['cell_type'] = pd.Categorical(stats['cell_type'], categories=CELL_TYPES, ordered=True)
    palette = get_condition_palette(analysis_type)

    heamap_overview_cell_types(
            stats, 
            slope_col=slope_col, 
            palette=palette, 
            figsize=(2, 3), 
            sig_dots_y_offset=3, 
            annotate_x_ticks=False, 
            # map_names={'cell_type': 'Sub type', 'major_cell_type': 'Cell type'},
            dendrogram_visible=False, 
            show_legend=False
        )    
    output_path = os.path.join(output_dir, f'overview_{dataset}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
    plt.close()
    print(f"  Saved: {output_path}")



def wrapper_plot_central_tfs_condition(stats, cell_types, group_col, args):
    """Plot aging vs perturbation comparison."""
    aging_stats_sig = retrieve_sig_stats(data_type='bulk', feature_type=args.feature_type).drop_duplicates(subset=['cell_type', 'gene'])
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

def plot_disease_case_tfs(args):
    """Plot healthy vs disease trends for specific genes."""
    # Determine condition column based on dataset
    if not args.cell_type:
        print(" This function only supports single cell type at a time.")
        raise ValueError("Please specify a single cell type using --cell_type.")
    condition_col = 'condition'
   
    for i, case_tf in enumerate(args.case_tfs):
        fig, ax = plt.subplots(1, 1, figsize=(2, .6), sharey=False, sharex=False)
        
        plot_healthy_disease_trend(
            dataset=args.dataset, 
            data_type=args.data_type, 
            cell_type=args.cell_type, 
            case_tf=case_tf, 
            condition_col=condition_col, 
            ax=ax
        )
        
        ax.set_title(f'{case_tf}', fontsize=10, pad=10, weight='bold')
        if i == 0:
            ax.set_xlabel('')
        
        output_path = os.path.join(args.output_dir, f'healthy_disease_trend_{case_tf}_{args.cell_type}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
        plt.close()
        print(f"  Saved: {output_path}")

def plot_ctr_condition_donor_level(args, cell_types):
    """Plot donor-level perturbation effects for case genes."""    
    stats = retrieve_stats(
        dataset=args.dataset, data_type=args.data_type, feature_type=args.feature_type)
    comparisons = stats['comparison'].unique()
    aggregate_per_donor = args.aggregate_per_donor if hasattr(args, 'aggregate_per_donor') else False
    case_tfs = args.case_tfs
    
    for cell_type in cell_types:
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
    
    if False:
        print("="*60)
        print(f"Dataset: {args.dataset}")

        print(f"Analysis type: {args.analysis_type}")
        print(f"Data type: {args.data_type}")
        print(f"Feature type: {args.feature_type}")
        print("="*60)

    args.case_tfs = ['KLF6', 'PRDM1' ,'LEF1', 'ZEB2']
    stats = retrieve_stats(
        dataset=args.dataset,
        data_type=args.data_type,
        feature_type=args.feature_type,
        multi_cohort=False
    )

    stats_sig = stats[stats['is_significant']]

    args.cell_types = CELL_TYPES if args.cell_types == ['all'] else args.cell_types
    
    config = get_config(dataset=args.dataset)

    if args.dataset == 'soundlife':
        plot_overview_heatmap(stats_sig, args)
        plot_aging_overlap(stats_sig, cell_types=['CD4T', 'CD8T', 'NK', 'MONO'] ,args=args)
        plot_directional_consistency_scatter(stats, cell_types=['CD4T', 'CD8T', 'NK', 'MONO'], 
                                             args=args,
                                             x_label = 'Validation analysis \n(significance)',
                                             y_label = 'Discovery analysis \n(significance)',
                                             agreement='same',
                                             label_consistent = 'Consistent',
                                             label_opposing = 'Opposing'
                                            )

    if args.dataset == 'perez_sle':
        groups = stats_sig['age_group'].unique()
        stats_sub = stats_sig[stats_sig['age_group'].isin(groups[0:1])]
        if not args.skip_overview:
            plot_overview_heatmap(stats_sub, args)
        # plot_aging_overlap(
        #     stats_sig, 
        #     cell_types=['CD4T', 'CD8T'],
        #     args=args
        # )
        plot_directional_consistency_scatter(stats_sub, cell_types=['CD4T', 'CD8T'], args=args,
                                             x_label = f'SLE \n(significance)',
                                             y_label = 'Natural aging \n(significance)',
                                             agreement='same',
                                             label_consistent = 'Acceleration',
                                             label_opposing = 'Rejuvenation'
                                             )
        wrapper_plot_central_tfs_condition(stats, group_col='age_group' ,cell_types=['CD4T', 'CD8T'], args=args)
        args.cell_type = 'CD8T'
        args.case_tfs = ['LEF1']
        plot_disease_case_tfs(args)
        plot_pathway_analysis(stats_sig, args)

    if args.dataset == 'parsebioscience':
        if not args.skip_overview:
            plot_overview_heatmap(stats_sig, args)
        args.case_tfs = ['LEF1', 'TCF7']
        selected_cell_types = ['CD4T', 'CD8T']
        args.cell_type = 'CD8T'
        args.aggregate_per_donor = True
        plot_ctr_condition_donor_level(args, cell_types=selected_cell_types)
        plot_overview_heatmap(stats_sig, args)
        
        args.cell_types = [ct for ct in args.cell_types if ct in selected_cell_types]
        plot_directional_consistency_scatter(stats, cell_types=selected_cell_types, args=args,
                                             x_label = f'{config.treatment_groups[1]} \n(significance)',
                                             y_label = 'Natural aging \n(significance)',
                                             agreement='opposite',
                                             label_consistent = 'Acceleration',
                                             label_opposing = 'Rejuvenation'
                                             )
        wrapper_plot_central_tfs_condition(stats, group_col='comparison', cell_types=selected_cell_types, args=args)
        
        plot_pathway_analysis(stats_sig, args)

    if args.dataset == 'op':
        if not args.skip_overview:
            plot_overview_heatmap(stats_sig, args)
        # args.case_tfs = ['KLF6', 'GATA3']
        args.case_tfs = ['STAT1', 'BATF']
        args.cell_type = 'CD4T'
        args.aggregate_per_donor = True   

        wrapper_plot_central_tfs_condition(stats, group_col='comparison', cell_types=['CD4T'], args=args)
        plot_ctr_condition_donor_level(args, cell_types=['CD4T']) 
        plot_directional_consistency_scatter(stats, cell_types=['CD4T', 'CD8T'], args=args,
                                             x_label = f'{config.treatment_groups[1]} \n(significance)',
                                             y_label = 'Natural aging \n(significance)',
                                             agreement='opposite',
                                             label_consistent = 'Acceleration',
                                             label_opposing = 'Rejuvenation'
                                             )

        # plot_pathway_analysis(stats_sig, args)

    if args.dataset == 'CXCL9':
        # if not args.skip_overview:
        #         plot_overview_heatmap(stats_sig, args)
        # plot_aging_overlap(
        #     stats_sig, 
        #     args
        # )    
        args.case_tfs = ['STAT1', 'BATF'] 
        for comparison in stats['comparison'].unique():
            stats_sub = stats[stats['comparison']==comparison]
            plot_directional_consistency_scatter(stats_sub, cell_types=['CD4T', 'CD8T'], args=args, 
                                                 x_label = f'{comparison} \n(significance)',
                                                 y_label = 'Natural aging \n(significance)',
                                                 save_tag = f"_{comparison.replace(' ', '_').replace('(', '_').replace(')', '_').replace(':', '_')}",
                                                 agreement='opposite',
                                                 label_consistent = 'Acceleration' if comparison != 'LPS \n (ctr: RPMI)' else 'Age deceleration',
                                                 label_opposing = 'Rejuvenation' if comparison != 'LPS \n (ctr: RPMI)' else 'Age deceleration',
                                                 )
            plot_ctr_condition_donor_level(args, cell_types=['CD4T']) 

        wrapper_plot_central_tfs_condition(stats, group_col='comparison', cell_types=['CD4T'], args=args)

        # plot_aging_overlap(stats_sig, cell_types=['CD4T', 'CD8T', 'NK', 'MONO'] ,args=args)

        # plot_pathway_analysis(stats_sig, args)
        # plot_ctr_condition_donor_level(args) 
  
    
if __name__ == '__main__':
    main()
