
## Identify sig networks
def plot_sig_networks(data_type = 'bulk'):
    from hiara.src.feature_association.helper import determine_sig_network
    if False:
        if True:
            determine_sig_network(data_type, min_degree=3)
        sig_net = retrieve_sig_net()
        sig_net_size = sig_net.groupby('cell_type').size()
        sig_net_size = sig_net_size.reindex(MAJOR_CTS, fill_value=0).reset_index()
        sig_net_size.columns = ['cell_type', 'edge_count']

        # Ensure categorical order for plotting
        sig_net_size['cell_type'] = pd.Categorical(sig_net_size['cell_type'], categories=MAJOR_CTS, ordered=True)
        sig_net_size = sig_net_size.sort_values('cell_type')

        # Plot
        fig, ax = plt.subplots(figsize=(1.5, 2))
        ax.barh(sig_net_size['cell_type'], sig_net_size['edge_count'], color=colors_blind[5], alpha=.5)

        # Aesthetics
        ax.invert_yaxis()
        ax.set_xlabel('TF-target pair')
        ax.set_ylabel('')
        ax.spines[['top', 'right']].set_visible(False)
        ax.margins(y=0.1, x=0.1)
        # ax.set_xscale('log')
        ax.set_title('Aging GRNs size', weight='bold', fontsize=10, pad=15)


    if False:
        sig_net = retrieve_sig_net()
        n_top = 20
        cell_type = 'CD8T'
        df = sig_net[sig_net['cell_type'] == cell_type]

        # Compute top source and target degrees
        source_counts = df['source'].value_counts().head(n_top).reset_index()
        source_counts.columns = ['source', 'degree']
        source_info = df[['source', 'trend_source']].drop_duplicates(subset='source')
        source_df = source_counts.merge(source_info, on='source', how='left')

        target_counts = df['target'].value_counts().head(n_top).reset_index()
        target_counts.columns = ['target', 'degree']
        target_info = df[['target', 'trend_target']].drop_duplicates(subset='target')
        target_df = target_counts.merge(target_info, on='target', how='left')

        # Setup plot
        fig, axes = plt.subplots(1, 2, figsize=(3, 4), sharey=False)

        # Plot sources
        source_colors = source_df['trend_source'].map(palette_trend).values
        target_colors = target_df['trend_target'].map(palette_trend).values[::-1]
        axes[0].barh(source_df['source'], source_df['degree'], color=source_colors)
        axes[0].set_title('TFs', fontsize=10)
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Out-degree')
        axes[0].spines[['top', 'right']].set_visible(False)

        # Plot targets
        axes[1].barh(target_df['target'][::-1], target_df['degree'][::-1], color=target_colors)
        axes[1].set_title('Targets', fontsize=10)
        axes[1].set_xlabel('In-degree')
        axes[1].spines[['top', 'right']].set_visible(False)

        # Title and layout
        fig.suptitle(f'Aging TFs and targets: {cell_type}', fontsize=10, weight='bold', y=.95)
        from matplotlib.patches import Patch

        legend_elements = [Patch(facecolor=color, label=label) for label, color in palette_trend_2.items()]
        # fig.legend(handles=legend_elements, bbox_to_anchor=(1.5, .8), fontsize=10, frameon=False, title='Trend', title_fontsize=10)
        fig.tight_layout()

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