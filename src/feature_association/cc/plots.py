import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from hiara.src.config import PLOTS_DIR, palette_trend_2, colors_blind
from hiara.src.feature_association.helper import retrieve_sig_stats


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


def wrapper_ccc_post_analysis(analysis_name):
    """
    Main wrapper for CCC post-analysis
    """
    print("\n" + "="*80)
    print("CCC POST-ANALYSIS")
    print("="*80)
    
    # Load significant results
    stats_sig = retrieve_sig_stats(analysis_name=analysis_name)
    
    if len(stats_sig) == 0:
        print("No significant results found!")
        return
    
    print(f"\nTotal significant L-R pairs: {len(stats_sig)}")
    print(f"  Increasing with age: {(stats_sig['slope'] > 0).sum()}")
    print(f"  Decreasing with age: {(stats_sig['slope'] < 0).sum()}")
    
    # Run all analyses
    plot_ccc_sender_receiver_matrix(stats_sig.copy(), analysis_name, trend='both')
    plot_ccc_sender_receiver_matrix(stats_sig.copy(), analysis_name, trend='positive')
    plot_ccc_sender_receiver_matrix(stats_sig.copy(), analysis_name, trend='negative')
    plot_ccc_directionality(stats_sig.copy(), analysis_name)
    plot_ccc_ligand_receptor_families(stats_sig.copy(), analysis_name)
    plot_ccc_hub_analysis(stats_sig.copy(), analysis_name)
    plot_ccc_top_pairs(stats_sig.copy(), analysis_name, top_n=20)
    
    print("\n" + "="*80)
    print("CCC POST-ANALYSIS COMPLETE")
    print("="*80)
