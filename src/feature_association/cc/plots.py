import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from hira.src.config import PLOTS_DIR, palette_trend_2, colors_blind
from hira.src.feature_association.helper import retrieve_sig_stats


def parse_ccc_feature_name(feature_name):
    """
    Parse CCC feature name: source__ligand__target__receptor
    Returns: (source, ligand, target, receptor)
    """
    parts = feature_name.split('__')
    if len(parts) == 4:
        return parts[0], parts[1], parts[2], parts[3]
    return None, None, None, None


def plot_ccc_sender_receiver_matrix(stats_sig, analysis_name, trend='both'):
    """
    1. Sender-Receiver Cell Type Matrix
    Heatmap showing number of significant L-R pairs between cell types
    """
    print("\n[1/5] Creating Sender-Receiver Matrix...")
    
    # Parse feature names
    stats_sig['source'] = stats_sig['gene'].apply(lambda x: parse_ccc_feature_name(x)[0])
    stats_sig['target'] = stats_sig['gene'].apply(lambda x: parse_ccc_feature_name(x)[2])
    
    # Filter by trend if needed
    if trend == 'positive':
        stats_sig = stats_sig[stats_sig['slope'] > 0]
    elif trend == 'negative':
        stats_sig = stats_sig[stats_sig['slope'] < 0]
    
    # Count L-R pairs per sender-receiver pair
    matrix_data = stats_sig.groupby(['source', 'target']).size().unstack(fill_value=0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Significant L-R pairs'})
    ax.set_xlabel('Receiver Subtype', fontsize=12)
    ax.set_ylabel('Sender Subtype', fontsize=12)
    title_suffix = f' ({trend} with age)' if trend != 'both' else ''
    ax.set_title(f'Age-Associated Cell-Cell Communications{title_suffix}', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    file_name = f"{PLOTS_DIR}/ccc_sender_receiver_matrix_{analysis_name}_{trend}.png"
    plt.savefig(file_name, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: {file_name}")
    
    return matrix_data


def plot_ccc_directionality(stats_sig, analysis_name):
    """
    2. Directionality Analysis
    Bar plots showing increasing vs decreasing communications
    """
    print("\n[2/5] Creating Directionality Analysis...")
    
    # Parse feature names
    stats_sig['source'] = stats_sig['gene'].apply(lambda x: parse_ccc_feature_name(x)[0])
    stats_sig['target'] = stats_sig['gene'].apply(lambda x: parse_ccc_feature_name(x)[2])
    stats_sig['trend'] = stats_sig['slope'].apply(lambda x: 'Increase in aging' if x > 0 else 'Decrease in aging')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # By sender
    sender_counts = stats_sig.groupby(['source', 'trend']).size().unstack(fill_value=0)
    sender_counts.plot(kind='barh', stacked=False, ax=axes[0], color=[palette_trend_2['Increase in aging'], palette_trend_2['Decrease in aging']])
    axes[0].set_xlabel('Number of L-R pairs', fontsize=11)
    axes[0].set_ylabel('Sender Subtype', fontsize=11)
    axes[0].set_title('By Sender', fontsize=12)
    axes[0].legend(title='', frameon=False)
    
    # By receiver
    receiver_counts = stats_sig.groupby(['target', 'trend']).size().unstack(fill_value=0)
    receiver_counts.plot(kind='barh', stacked=False, ax=axes[1], color=[palette_trend_2['Increase in aging'], palette_trend_2['Decrease in aging']])
    axes[1].set_xlabel('Number of L-R pairs', fontsize=11)
    axes[1].set_ylabel('Receiver Subtype', fontsize=11)
    axes[1].set_title('By Receiver', fontsize=12)
    axes[1].legend(title='', frameon=False)
    
    # Overall
    overall_counts = stats_sig['trend'].value_counts()
    axes[2].bar(range(len(overall_counts)), overall_counts.values, 
                color=[palette_trend_2[t] for t in overall_counts.index])
    axes[2].set_xticks(range(len(overall_counts)))
    axes[2].set_xticklabels(overall_counts.index, rotation=45, ha='right')
    axes[2].set_ylabel('Number of L-R pairs', fontsize=11)
    axes[2].set_title('Overall', fontsize=12)
    
    plt.suptitle('Communication Directionality with Aging', fontsize=14, y=1.02)
    
    file_name = f"{PLOTS_DIR}/ccc_directionality_{analysis_name}.png"
    plt.savefig(file_name, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: {file_name}")


def plot_ccc_ligand_receptor_families(stats_sig, analysis_name):
    """
    3. Ligand/Receptor Family Enrichment
    """
    print("\n[3/5] Creating Ligand/Receptor Family Enrichment...")
    
    # Parse feature names
    stats_sig['ligand'] = stats_sig['gene'].apply(lambda x: parse_ccc_feature_name(x)[1])
    stats_sig['receptor'] = stats_sig['gene'].apply(lambda x: parse_ccc_feature_name(x)[3])
    stats_sig['trend'] = stats_sig['slope'].apply(lambda x: 'Increase in aging' if x > 0 else 'Decrease in aging')
    
    # Define families
    families = {
        'Interleukins': lambda x: x.startswith('IL') and x[2:].split('_')[0].replace('R', '').isdigit(),
        'TNF family': lambda x: x.startswith('TNF'),
        'Interferons': lambda x: x.startswith('IFN'),
        'Chemokines': lambda x: x.startswith(('CCL', 'CXCL', 'CX3CL', 'XCL')),
        'Growth factors': lambda x: any(x.startswith(gf) for gf in ['VEGF', 'FGF', 'EGF', 'PDGF', 'TGF', 'HGF', 'IGF']),
        'Adhesion': lambda x: any(x.startswith(a) for a in ['ICAM', 'VCAM', 'ITGA', 'ITGB', 'CD', 'SELP', 'SELL']),
        'Checkpoint': lambda x: any(cp in x for cp in ['PD1', 'PDL', 'CTLA', 'LAG3', 'TIM3', 'TIGIT'])
    }
    
    # Categorize ligands and receptors
    def categorize(gene):
        for family, check_func in families.items():
            if check_func(gene):
                return family
        return 'Other'
    
    stats_sig['ligand_family'] = stats_sig['ligand'].apply(categorize)
    stats_sig['receptor_family'] = stats_sig['receptor'].apply(categorize)
    
    # Count by family
    ligand_family_counts = stats_sig.groupby(['ligand_family', 'trend']).size().unstack(fill_value=0)
    ligand_family_counts = ligand_family_counts[ligand_family_counts.sum(axis=1) > 0].sort_values(by='Increase in aging', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ligand_family_counts.plot(kind='barh', stacked=False, ax=ax, 
                               color=[palette_trend_2['Increase in aging'], palette_trend_2['Decrease in aging']])
    ax.set_xlabel('Number of L-R pairs', fontsize=11)
    ax.set_ylabel('Ligand/Receptor Family', fontsize=11)
    ax.set_title('Age-Associated Communications by Signaling Family', fontsize=13)
    ax.legend(title='', frameon=False, loc='lower right')
    
    file_name = f"{PLOTS_DIR}/ccc_family_enrichment_{analysis_name}.png"
    plt.savefig(file_name, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: {file_name}")


def plot_ccc_hub_analysis(stats_sig, analysis_name):
    """
    4. Hub Analysis - Identify sender/receiver hubs
    """
    print("\n[4/5] Creating Hub Analysis...")
    
    # Parse feature names
    stats_sig['source'] = stats_sig['gene'].apply(lambda x: parse_ccc_feature_name(x)[0])
    stats_sig['target'] = stats_sig['gene'].apply(lambda x: parse_ccc_feature_name(x)[2])
    
    # Count outgoing (sender) and incoming (receiver) communications
    sender_counts = stats_sig['source'].value_counts().sort_values(ascending=True)
    receiver_counts = stats_sig['target'].value_counts().sort_values(ascending=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sender hubs
    axes[0].barh(range(len(sender_counts)), sender_counts.values, color=colors_blind[0])
    axes[0].set_yticks(range(len(sender_counts)))
    axes[0].set_yticklabels(sender_counts.index)
    axes[0].set_xlabel('Number of significant L-R pairs sent', fontsize=11)
    axes[0].set_title('Sender Hubs', fontsize=12)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    
    # Receiver hubs
    axes[1].barh(range(len(receiver_counts)), receiver_counts.values, color=colors_blind[1])
    axes[1].set_yticks(range(len(receiver_counts)))
    axes[1].set_yticklabels(receiver_counts.index)
    axes[1].set_xlabel('Number of significant L-R pairs received', fontsize=11)
    axes[1].set_title('Receiver Hubs', fontsize=12)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    
    plt.suptitle('Communication Hub Cell Types', fontsize=14, y=1.02)
    
    file_name = f"{PLOTS_DIR}/ccc_hub_analysis_{analysis_name}.png"
    plt.savefig(file_name, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: {file_name}")


def plot_ccc_top_pairs(stats_sig, analysis_name, top_n=20):
    """
    5. Top Specific Communications
    """
    print("\n[5/5] Creating Top L-R Pairs Plot...")
    
    # Get top by absolute slope
    stats_sig_sorted = stats_sig.sort_values('slope', key=abs, ascending=False).head(top_n)
    
    # Create readable labels
    stats_sig_sorted['source'] = stats_sig_sorted['gene'].apply(lambda x: parse_ccc_feature_name(x)[0])
    stats_sig_sorted['ligand'] = stats_sig_sorted['gene'].apply(lambda x: parse_ccc_feature_name(x)[1])
    stats_sig_sorted['target'] = stats_sig_sorted['gene'].apply(lambda x: parse_ccc_feature_name(x)[2])
    stats_sig_sorted['receptor'] = stats_sig_sorted['gene'].apply(lambda x: parse_ccc_feature_name(x)[3])
    stats_sig_sorted['label'] = (stats_sig_sorted['source'] + ' → ' + stats_sig_sorted['target'] + 
                                   '\n' + stats_sig_sorted['ligand'] + '-' + stats_sig_sorted['receptor'])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))
    colors = [palette_trend_2['Increase in aging'] if s > 0 else palette_trend_2['Decrease in aging'] 
              for s in stats_sig_sorted['slope']]
    
    ax.barh(range(len(stats_sig_sorted)), stats_sig_sorted['slope'].values, color=colors)
    ax.set_yticks(range(len(stats_sig_sorted)))
    ax.set_yticklabels(stats_sig_sorted['label'].values, fontsize=9)
    ax.set_xlabel('Association with age (slope)', fontsize=11)
    ax.set_title(f'Top {top_n} Age-Associated Cell-Cell Communications', fontsize=13, pad=15)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='-')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Add legend
    legend_elements = [
        Patch(facecolor=palette_trend_2['Increase in aging'], label='Increase in aging'),
        Patch(facecolor=palette_trend_2['Decrease in aging'], label='Decrease in aging')
    ]
    ax.legend(handles=legend_elements, frameon=False, loc='lower right')
    
    file_name = f"{PLOTS_DIR}/ccc_top_pairs_{analysis_name}.png"
    plt.savefig(file_name, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: {file_name}")


def plot_ccc_lr_pairs_vs_datasets(analysis_name, datasets=None, top_n=15, filter_significant=True):
    """
    Plot hub L-R pairs across datasets - shows which ligand-receptor interactions
    are consistently significant across multiple datasets and cell type pairs.
    Similar to plot_features_vs_datasets but for CCC data.
    """
    from hira.src.feature_association.helper import retrieve_stats
    from hira.src.config import DISCOVERY_COHORTS, surrogate_names, cmap_trend
    from matplotlib.colors import Normalize
    from matplotlib import gridspec
    
    if datasets is None:
        datasets = DISCOVERY_COHORTS
    
    print(f"\nCreating L-R pairs vs datasets plot...")
    
    # Load all stats
    stats_all = retrieve_stats(analysis_name=analysis_name)
    stats_all = stats_all[stats_all['dataset'].isin(datasets)]
    
    if filter_significant:
        stats_sig = retrieve_sig_stats(analysis_name=analysis_name)
        sig_genes = stats_sig['gene'].unique()
        stats_all = stats_all[stats_all['gene'].isin(sig_genes)]
    
    # Parse L-R pair from feature name (extract ligand-receptor: indices 1 and 3)
    def get_lr_pair(gene_name):
        parts = parse_ccc_feature_name(gene_name)
        if parts[1] and parts[3]:
            return f"{parts[1]}-{parts[3]}"  # ligand-receptor
        return None
    
    stats_all['lr_pair'] = stats_all['gene'].apply(get_lr_pair)
    stats_all = stats_all[stats_all['lr_pair'].notna()]
    
    # Extract source and target cell types
    stats_all['source'] = stats_all['gene'].apply(lambda x: parse_ccc_feature_name(x)[0])
    stats_all['target'] = stats_all['gene'].apply(lambda x: parse_ccc_feature_name(x)[2])
    stats_all['cell_pair'] = stats_all['source'] + '_' + stats_all['target']
    
    # Count how many unique cell type pairs each L-R pair appears in (centrality)
    lr_counts = stats_all.groupby('lr_pair')['cell_pair'].nunique().reset_index(name='centrality')
    lr_counts = lr_counts.sort_values('centrality', ascending=False)
    
    # Get top hub L-R pairs
    top_lr_pairs = lr_counts.head(top_n)['lr_pair'].values
    stats_subset = stats_all[stats_all['lr_pair'].isin(top_lr_pairs)]
    
    # Prepare data for plotting
    stats_subset['neg_log10_adj_pval'] = -np.log10(stats_subset['p_value_adj'])
    stats_subset['dataset'] = stats_subset['dataset'].apply(lambda name: surrogate_names.get(name, name))
    stats_subset['dataset'] = pd.Categorical(stats_subset['dataset'], categories=[surrogate_names.get(d, d) for d in datasets], ordered=True)
    stats_subset['lr_pair'] = pd.Categorical(stats_subset['lr_pair'], categories=top_lr_pairs, ordered=True)
    
    # Group by L-R pair and dataset, aggregate across cell type pairs
    plot_data = stats_subset.groupby(['lr_pair', 'dataset']).agg({
        'slope': 'mean',
        'neg_log10_adj_pval': 'max',
        'gene': 'size'
    }).reset_index()
    plot_data.rename(columns={'gene': 'n_cell_pairs'}, inplace=True)
    
    # Prepare centrality data
    centrality_data = lr_counts[lr_counts['lr_pair'].isin(top_lr_pairs)].copy()
    centrality_data['lr_pair'] = pd.Categorical(centrality_data['lr_pair'], categories=top_lr_pairs, ordered=True)
    centrality_data = centrality_data.sort_values('lr_pair')
    
    # Normalize centrality for plotting
    centrality_data['centrality_norm'] = centrality_data['centrality'] / centrality_data['centrality'].max()
    
    # Plotting
    n_datasets = len(datasets)
    base_width = max(1.5, min(3.5, 1.0 + n_datasets * 0.2))
    base_height = max(0.12, min(0.25, 0.2 - top_n * 0.002))
    fig_width = base_width * n_datasets + 2  # Extra space for centrality
    fig_height = base_height * top_n + 1
    
    # Create figure with gridspec for multiple panels
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)
    
    # Main scatter plot
    ax_main = fig.add_subplot(gs[0])
    
    # Create color normalization for slope (symmetric around 0)
    max_abs_slope = abs(plot_data['slope']).max()
    norm = Normalize(vmin=-max_abs_slope, vmax=max_abs_slope)
    
    # Create scatter plot
    for dataset in plot_data['dataset'].unique():
        data_subset = plot_data[plot_data['dataset'] == dataset]
        x = [list(plot_data['dataset'].cat.categories).index(dataset)] * len(data_subset)
        y = [list(plot_data['lr_pair'].cat.categories).index(lr) for lr in data_subset['lr_pair']]
        
        scatter = ax_main.scatter(
            x, y,
            s=data_subset['neg_log10_adj_pval'] * 10,
            c=data_subset['slope'],
            cmap=cmap_trend,
            norm=norm,
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5
        )
    
    # Format main plot axes
    ax_main.set_xticks(range(len(plot_data['dataset'].cat.categories)))
    ax_main.set_xticklabels(plot_data['dataset'].cat.categories, rotation=45, ha='right')
    ax_main.set_yticks(range(len(top_lr_pairs)))
    ax_main.set_yticklabels(top_lr_pairs)
    ax_main.set_xlabel('Dataset', fontsize=11)
    ax_main.set_ylabel('Ligand-Receptor Pair', fontsize=11)
    ax_main.set_title(f'Top {top_n} Hub L-R Pairs Across Datasets', fontsize=12, pad=10)
    ax_main.spines['right'].set_visible(False)
    ax_main.spines['top'].set_visible(False)
    ax_main.grid(True, alpha=0.2, axis='y')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax_main, label='Association with age (slope)', pad=0.02)
    
    # Centrality barplot
    ax_centrality = fig.add_subplot(gs[1])
    ax_centrality.barh(
        range(len(centrality_data)),
        centrality_data['centrality'].values,
        color='#56B4E9',
        alpha=0.7
    )
    ax_centrality.set_yticks(range(len(top_lr_pairs)))
    ax_centrality.set_yticklabels([])  # No labels, shared with main plot
    ax_centrality.set_xlabel('Centrality\n(# cell type pairs)', fontsize=10)
    ax_centrality.spines['right'].set_visible(False)
    ax_centrality.spines['top'].set_visible(False)
    ax_centrality.spines['left'].set_visible(False)
    ax_centrality.set_ylim(ax_main.get_ylim())
    
    file_name = f"{PLOTS_DIR}/ccc_lr_pairs_vs_datasets_{analysis_name}.png"
    plt.savefig(file_name, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: {file_name}")
