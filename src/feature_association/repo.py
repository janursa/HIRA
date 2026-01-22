
def _plot_directional_consistency_scatter(stats, args):
    """
    Generate directional consistency scatter plots comparing Sound Life vs Reference aging genes.
    Shows signed -log10(p-values) with direction concordance.
    """
    print("Generating directional consistency scatter plots...")
    
    feature_type = args.feature_type
    dataset = args.dataset
    included_cell_types = args.cell_types
    output_dir = args.output_dir
    
    # Load reference aging genes
    ref_stats_sig = retrieve_sig_stats(data_type='bulk', feature_type=feature_type, filter_inconsistent=True)
    ref_stats_sig = ref_stats_sig.groupby(['cell_type', 'gene']).agg({'slope': 'mean', 'meta_p_adj': 'min'}).reset_index()
    
    for cell_type in included_cell_types:
        if cell_type not in stats['cell_type'].unique():
            print(f"  Warning: No significant genes for {cell_type}")
            continue        
        sl_ct = stats[stats['cell_type'] == cell_type].copy()
        ref_ct = ref_stats_sig[ref_stats_sig['cell_type'] == cell_type].copy()
        sl_ct_renamed = sl_ct[['gene', 'slope', 'p_value_adj']].rename(columns={'slope': 'slope_condition'})
        merged = ref_ct.merge(
            sl_ct_renamed[['gene', 'slope_condition', 'p_value_adj']],
            on='gene',
            how='outer'
        ) 
        merged = merged.dropna(subset=['slope'])
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
        fig = plt.figure(figsize=(2.5, 2.5))
        ax_main = plt.subplot(1, 1, 1)
        s = 20
        if len(opp_dir) > 0:
            ax_main.scatter(
                opp_dir['slope_condition'],
                opp_dir['slope'],
                c='red',
                s=s,
                alpha=0.6,
                label=f'Inconsistent (n={len(opp_dir)})',
                edgecolors='darkred',
                linewidths=.1
            )
        if len(same_dir) > 0:
            ax_main.scatter(
                same_dir['slope_condition'],
                same_dir['slope'],
                c='green',
                s=s,
                alpha=0.6,
                label=f'Consistent (n={len(same_dir)})',
                edgecolors='darkred',
                linewidths=.1
            )
        ax_main.grid(False)
        ax_main.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax_main.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax_main.set_xlabel('Validation \n-log10(p)', fontsize=10)
        ax_main.set_ylabel('Reference aging \n-log10(p)', fontsize=10)
        ax_main.set_title(f'{cell_type}', fontsize=12, weight='bold', pad=10)
        ax_main.legend(loc=(1.01, 0.5), framealpha=0.9, fontsize=10, frameon=False)
        
        output_path = os.path.join(output_dir, f'consistent_{dataset}_{cell_type}.png')
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {output_path}")