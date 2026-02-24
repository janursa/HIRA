#!/usr/bin/env python
"""
Plot group wrappers for different aging and condition analyses.
Each wrapper contains the plotting logic for a specific analysis type.
"""

from hiara.src.feature_association.plots import (
    wrapper_sig_features_counts, 
    plot_heatmap_overal, 
    plot_central_features,
    plot_interaction_of_features_between_cell_types, 
    plot_case_tf, 
    gsea_analysis, 
    plot_scatter_feature_vs_age,
    plot_features_vs_datasets, 
    plot_directional_consistency_scatter,
    plot_young_vs_aging,
    plot_aging_overlap
    
)
from hiara.src.feature_association.cc.plots import plot_ccc_lr_pairs_vs_datasets, plot_ccc_directionality, plot_ccc_hub_analysis, plot_ccc_ligand_receptor_families, plot_ccc_sender_receiver_matrix, plot_ccc_top_pairs
from hiara import retrieve_sig_stats, retrieve_stats, mapping_minor_2_major
from hiara.src.config import get_config_fa, get_config, MAJOR_CTS


def wrapper_plots_tfa_major_b_aging(args, stats_features, stats_features_sig, skip_pathway):
    """Plot group for tfa_major_b analysis."""
    analysis_name = args.analysis_name
    
    plot_scatter_feature_vs_age(analysis_name, cell_types=['CD8T', 'CD4T'])
    plot_heatmap_overal(stats_features, analysis_name=args.analysis_name)
    wrapper_sig_features_counts(args)
    plot_central_features(stats_features_sig, cell_types=['CD4T', 'CD8T', 'NK', 'MONO'])
    plot_interaction_of_features_between_cell_types(args)
    plot_case_tf(args)
    if not skip_pathway:
        gsea_analysis(stats_sig=stats_features_sig)


def wrapper_plots_tfa_sub_b_aging(args, stats_features, stats_features_sig, skip_pathway):
    """Plot group for tfa_sub_b analysis."""
    analysis_name = args.analysis_name
    
    wrapper_sig_features_counts(args)
    plot_heatmap_overal(stats_features, analysis_name=args.analysis_name)

    stats_ref = retrieve_stats(analysis_name='tfa_major_b', cell_type='CD8T')
    stats_features_c = stats_features[stats_features['cell_type']=='Tcm_Naive_CD8'].copy()
    stats_features_c['cell_type'] = stats_features_c['cell_type'].map(mapping_minor_2_major)
    
    plot_directional_consistency_scatter(
        stats_features_c, 
        stats_ref,
        x_label='Tcm/Naive CD8 \n(significance)',
        y_label='CD8T \n(significance)',
        association_col='-log10_p_adj',
        agreement='same',
        label_consistent='Consistent',
        label_opposing='Opposing',
        save_suffix='sub_vs_major_aging',
        pvalue_col='meta_p_adj'
    )


def wrapper_plots_gene_expression_aging(args, stats_features, stats_features_sig, skip_pathway):
    """Plot group for gene_expression analysis."""
    wrapper_sig_features_counts(args)
    plot_heatmap_overal(stats_features, analysis_name=args.analysis_name)


def wrapper_plots_ct_tf_markers_aging(args, stats_features, stats_features_sig, skip_pathway):
    """Plot group for ct_tf_markers analysis."""
    analysis_name = args.analysis_name
    
    plot_heatmap_overal(stats_features, analysis_name=args.analysis_name)
    stats_features_ref = retrieve_sig_stats(analysis_name='tfa_major_b', cell_type='CD8T')
    stats_features_s = stats_features[stats_features['cell_type']=='Tcm_Naive_CD8']
    stats_features_s['cell_type'] = stats_features_s['cell_type'].map(mapping_minor_2_major)

    plot_directional_consistency_scatter(
        stats_features_s, 
        stats_features_ref,
        x_label='TCM/Naive CD8 markers \n(significance)',
        y_label='TF act vs age in CD8T \n(significance)',
        association_col='-log10_p_adj',
        agreement='same',
        label_consistent='Consistent',
        label_opposing='Opposing',
        save_suffix='',
        pvalue_col='meta_p_adj'
    )


def wrapper_plots_tfa_peg_aging(args, stats_features, stats_features_sig, skip_pathway):
    """Plot group for tfa_peg analysis."""
    analysis_name = args.analysis_name
    
    plot_young_vs_aging(analysis_name, cell_type='CD8T', young_age_threshold=30, annotate_top_n=10)
    wrapper_sig_features_counts(args)
    plot_scatter_feature_vs_age(analysis_name, cell_types=['CD8T'], features=['TCF7', 'LEF1', 'GATA3', 'KLF6'])
    plot_scatter_feature_vs_age(analysis_name, cell_types=['CD8T'], feature_selection_mode='top_central')
    plot_scatter_feature_vs_age(analysis_name, cell_types=['CD8T'], feature_selection_mode='top_sig')
    plot_features_vs_datasets(cell_type='CD8T', analysis_name=analysis_name, top_features=20)

    stats_features_ref = retrieve_sig_stats(analysis_name='tfa_major_b', cell_type='CD8T')
    
    plot_directional_consistency_scatter(
        stats_features[stats_features['cell_type']=='CD8T'], 
        stats_features_ref,
        x_label='TF act vs traj. \n(significance)',
        y_label='TF act. \n(significance)',
        association_col='-log10_p_adj',
        agreement='same',
        label_consistent='Consistent',
        label_opposing='Opposing',
        save_suffix='',
        pvalue_col='meta_p_adj'
    )


def wrapper_plots_ct_freq_aging(args, stats_features, stats_features_sig, skip_pathway):
    """Plot group for ct_freq analysis."""
    analysis_name = args.analysis_name
    
    cell_types = get_config_fa(analysis_name)['cell_types']
    plot_scatter_feature_vs_age(analysis_name, cell_types=cell_types, feature_selection_mode='top_sig', top_features=10)


def wrapper_plots_ct_pol_dist_aging(args, stats_features, stats_features_sig, skip_pathway):
    """Plot group for ct_pol_dist analysis."""
    analysis_name = args.analysis_name
    
    plot_scatter_feature_vs_age(analysis_name, cell_types=['CD8T', 'CD4T'], features=['pol_dist'], filter_for_sig=False)


def wrapper_plots_ccc_aging(args, stats_features, stats_features_sig, skip_pathway):
    """Plot group for ccc_sub_b analysis."""
    analysis_name = args.analysis_name    
    plot_scatter_feature_vs_age(analysis_name, cell_types=['all'], feature_selection_mode='top_sig', top_features=10)
    
    # Hub L-R pairs across datasets
    plot_ccc_lr_pairs_vs_datasets(analysis_name, top_n=15, filter_significant=True)
    
    # CCC-specific plots
    plot_ccc_hub_analysis(stats_features_sig.copy(), analysis_name)
    plot_ccc_sender_receiver_matrix(stats_features_sig.copy(), analysis_name, trend='both')
    plot_ccc_sender_receiver_matrix(stats_features_sig.copy(), analysis_name, trend='positive')
    plot_ccc_sender_receiver_matrix(stats_features_sig.copy(), analysis_name, trend='negative')
    plot_ccc_directionality(stats_features_sig.copy(), analysis_name)
    plot_ccc_ligand_receptor_families(stats_features_sig.copy(), analysis_name)
    plot_ccc_top_pairs(stats_features_sig.copy(), analysis_name, top_n=20)

    

# =============================================================================
# CONDITION ANALYSIS WRAPPERS
# =============================================================================

def wrapper_plots_tfa_major_b_condition(args, stats, stats_sig):
    """Plot group for tfa_major_b condition analysis."""
    from hiara.src.feature_association.plots_condition import (
        plot_overview_heatmap,
        wrapper_plot_central_tfs_condition,
        plot_disease_case_tfs,
        plot_ctr_condition_donor_level,
        plot_pathway_analysis
    )
    
    analysis_name = args.analysis_name
    dataset = args.dataset
    
    if dataset == 'soundlife':
        plot_overview_heatmap(stats_sig, args)
        plot_aging_overlap(analysis_name, stats_sig, cell_types=['CD4T', 'CD8T', 'NK', 'MONO'], args=args)
        plot_directional_consistency_scatter(
            stats, 
            analysis_name=analysis_name,
            save_suffix=args.dataset,
            x_label='Validation analysis \n(significance)',
            y_label='Discovery analysis \n(significance)',
            agreement='same',
            label_consistent='Consistent',
            label_opposing='Opposing'
        )
    
    elif dataset == 'perez_sle':
        age_group = 'Both age groups'
        stats_sub = stats_sig[stats_sig['age_group']==age_group]
        assert len(stats_sub['age_group'].unique()) == 1, "Expected only one age group in stats_sub"
        if not args.skip_overview:
            plot_overview_heatmap(stats_sub, args)
        if 'major' in analysis_name:
            cell_types = ['CD4T', 'CD8T']
        elif 'sub' in analysis_name:
            cell_types = ['Tcm_Naive_CD4', 'Tcm_Naive_CD8', 'MAIT']
        else:
            raise ValueError(f"Unexpected analysis name: {analysis_name}")

        plot_directional_consistency_scatter(
            stats_sub[stats_sub['cell_type'].isin(cell_types)], 
            stats_ref=retrieve_sig_stats(analysis_name=analysis_name),
            save_suffix=args.dataset,
            x_label=f'SLE \n(significance)',
            y_label='Natural aging \n(significance)',
            agreement='same',
            label_consistent='Acceleration',
            label_opposing='Rejuvenation'
        )
        
        wrapper_plot_central_tfs_condition(stats, group_col='age_group', cell_types=cell_types, args=args)
        if analysis_name == 'tfa_major_b':
            cell_type = 'CD8T'
            case_tfs = ['LEF1']
            plot_disease_case_tfs(args, cell_type=cell_type, case_tfs=case_tfs)
            
        else:
            pass
        if not args.skip_pathway:
            plot_pathway_analysis(stats_sig, args)
    
    elif dataset == 'parsebioscience':
        if not args.skip_overview:
            plot_overview_heatmap(stats_sig, args)
        args.case_tfs = ['LEF1', 'TCF7']
        selected_cell_types = ['CD4T', 'CD8T']
        args.cell_type = 'CD8T'
        args.aggregate_per_donor = True
        plot_ctr_condition_donor_level(args, cell_types=selected_cell_types)
        plot_overview_heatmap(stats_sig, args)
        
        args.cell_types = [ct for ct in args.cell_types if ct in selected_cell_types]
        config = get_config(dataset=args.dataset)
        plot_directional_consistency_scatter(
            stats_sig[stats_sig['cell_type'].isin(['CD4T', 'CD8T'])], 
            stats_ref=retrieve_sig_stats(analysis_name='tfa_major_b'),
            save_suffix=args.dataset,
            x_label=f'{config.treatment_groups[1]} \n(significance)',
            y_label='Natural aging \n(significance)',
            agreement='opposite',
            label_consistent='Acceleration',
            label_opposing='Rejuvenation'
        )
        wrapper_plot_central_tfs_condition(stats, group_col='comparison', cell_types=selected_cell_types, args=args)
        if not args.skip_pathway:
            plot_pathway_analysis(stats_sig, args)
    
    elif dataset == 'op':
        if not args.skip_overview:
            plot_overview_heatmap(stats_sig, args)
        args.case_tfs = ['STAT1', 'BATF']
        args.cell_type = 'CD4T'
        args.aggregate_per_donor = True   

        wrapper_plot_central_tfs_condition(stats, group_col='comparison', cell_types=['CD4T'], args=args)
        plot_ctr_condition_donor_level(args, cell_types=['CD4T']) 
        stats_aging = retrieve_sig_stats(analysis_name='tfa_major_b').drop_duplicates(subset=['cell_type', 'gene'])
        config = get_config(dataset=args.dataset)
        plot_directional_consistency_scatter(
            stats_sig[stats_sig['cell_type'].isin(['CD4T', 'CD8T'])], 
            stats_ref=stats_aging,
            save_suffix=args.dataset,
            x_label=f'{config.treatment_groups[1]} \n(significance)',
            y_label='Natural aging \n(significance)',
            agreement='opposite',
            label_consistent='Acceleration',
            label_opposing='Rejuvenation'
        )
    
    elif dataset == 'CXCL9':
        args.case_tfs = ['STAT1', 'BATF'] 
        for comparison in stats['comparison'].unique():
            stats_sub = stats[stats['comparison']==comparison]
            plot_directional_consistency_scatter(
                stats_sub[stats_sub['cell_type'].isin(['CD4T', 'CD8T'])], 
                stats_ref=retrieve_sig_stats(analysis_name='tfa_major_b'),
                save_suffix=f"{args.dataset}_f_{comparison.replace(' ', '_').replace('(', '_').replace(')', '_').replace(':', '_')}",
                x_label=f'{comparison} \n(significance)',
                y_label='Natural aging \n(significance)',
                agreement='opposite',
                label_consistent='Acceleration' if comparison != 'LPS \n (ctr: RPMI)' else 'Age deceleration',
                label_opposing='Rejuvenation' if comparison != 'LPS \n (ctr: RPMI)' else 'Age deceleration',
            )
            plot_ctr_condition_donor_level(args, cell_types=['CD4T']) 

        wrapper_plot_central_tfs_condition(stats, group_col='comparison', cell_types=['CD4T'], args=args)
    
    else:
        raise ValueError(f'Undefined dataset: {dataset}')


def wrapper_plots_tfa_sub_b_condition(args, stats, stats_sig):
    """Plot group for tfa_sub_b condition analysis."""
    wrapper_plots_tfa_major_b_condition(args, stats, stats_sig)


def wrapper_plots_gene_expression_condition(args, stats, stats_sig):
    """Plot group for gene_expression condition analysis."""
    raise ValueError(f'gene_expression condition analysis not yet implemented')


def wrapper_plots_ct_tf_markers_condition(args, stats, stats_sig):
    """Plot group for ct_tf_markers condition analysis."""
    raise ValueError(f'ct_tf_markers condition analysis not yet implemented')


def wrapper_plots_tfa_peg_condition(args, stats, stats_sig):
    """Plot group for tfa_peg condition analysis."""
    raise ValueError(f'tfa_peg condition analysis not yet implemented')


def wrapper_plots_ct_freq_condition(args, stats, stats_sig):
    """Plot group for ct_freq condition analysis."""
    raise ValueError(f'ct_freq condition analysis not yet implemented')


def wrapper_plots_ct_pol_dist_condition(args, stats, stats_sig):
    """Plot group for ct_pol_dist condition analysis."""
    raise ValueError(f'ct_pol_dist condition analysis not yet implemented')


def wrapper_plots_ccc_sub_b_condition(args, stats, stats_sig):
    """Plot group for ccc_sub_b condition analysis."""
    raise ValueError(f'ccc_sub_b condition analysis not yet implemented')
