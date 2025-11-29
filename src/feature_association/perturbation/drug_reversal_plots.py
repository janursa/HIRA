"""
Plotting functions for drug reversal analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from ciim.src.common import palette_treatment, palette_trend_2


def plot_reversal_heatmap(
    results_df: pd.DataFrame,
    dataset: str,
    cell_types: list,
    figsize: tuple = None
) -> plt.Figure:
    """
    Plot heatmap of reversal scores for all drug-cell type combinations.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from analyze_drug_reversal
    dataset : str
        Dataset name
    cell_types : list
        List of cell types
    figsize : tuple, optional
        Figure size
    
    Returns
    -------
    plt.Figure
        Figure object
    """
    # Pivot data for heatmap
    pivot_score = results_df.pivot(
        index='drug',
        columns='cell_type',
        values='reversal_score'
    )
    
    pivot_pval = results_df.pivot(
        index='drug',
        columns='cell_type',
        values='fisher_pvalue_adj'
    )
    
    # Reorder columns to match cell_types order
    pivot_score = pivot_score.reindex(columns=cell_types, fill_value=0)
    pivot_pval = pivot_pval.reindex(columns=cell_types, fill_value=1)
    
    # Determine figure size
    if figsize is None:
        n_drugs = len(pivot_score)
        n_celltypes = len(cell_types)
        figsize = (n_celltypes * 0.8 + 2, n_drugs * 0.4 + 1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(
        pivot_score.values,
        cmap='RdBu_r',
        aspect='auto',
        vmin=-1,
        vmax=1,
        interpolation='nearest'
    )
    
    # Add significance markers
    for i in range(len(pivot_score)):
        for j in range(len(cell_types)):
            pval = pivot_pval.iloc[i, j]
            score = pivot_score.iloc[i, j]
            
            if pval < 0.05 and abs(score) > 0.2:
                marker = '***' if pval < 0.001 else '**' if pval < 0.01 else '*'
                color = 'white' if abs(score) > 0.5 else 'black'
                ax.text(j, i, marker, ha='center', va='center',
                       fontsize=12, color=color, weight='bold')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(cell_types)))
    ax.set_yticks(np.arange(len(pivot_score)))
    ax.set_xticklabels(cell_types, fontsize=10)
    ax.set_yticklabels(pivot_score.index, fontsize=9)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Reversal Score', rotation=270, labelpad=20, fontsize=10)
    
    # Title
    ax.set_title(f'Drug Reversal Analysis - {dataset.upper()}', 
                fontsize=12, weight='bold', pad=15)
    
    ax.set_xlabel('Cell Type', fontsize=10)
    ax.set_ylabel('Drug', fontsize=10)
    
    plt.tight_layout()
    
    return fig


def plot_drug_ranking(
    results_df: pd.DataFrame,
    cell_type: str,
    dataset: str,
    n_top: int = 10,
    figsize: tuple = (4, 5)
) -> plt.Figure:
    """
    Plot ranking of drugs by reversal score for a specific cell type.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from analyze_drug_reversal
    cell_type : str
        Cell type to plot
    dataset : str
        Dataset name
    n_top : int
        Number of top drugs to show
    figsize : tuple
        Figure size
    
    Returns
    -------
    plt.Figure
        Figure object
    """
    # Filter for cell type
    df_ct = results_df[results_df['cell_type'] == cell_type].copy()
    
    if len(df_ct) == 0:
        print(f"No data for {cell_type}")
        return None
    
    # Sort by reversal score
    df_ct = df_ct.sort_values('reversal_score', ascending=True)
    
    # Take top N
    if len(df_ct) > n_top:
        df_ct = df_ct.tail(n_top)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color by classification
    colors = []
    for classification in df_ct['classification']:
        if classification == 'rejuvenating':
            colors.append('#2E7D32')  # Green
        elif classification == 'accelerating':
            colors.append('#C62828')  # Red
        else:
            colors.append('#757575')  # Gray
    
    # Plot horizontal bar
    y_pos = np.arange(len(df_ct))
    bars = ax.barh(y_pos, df_ct['reversal_score'], color=colors, alpha=0.7)
    
    # Add significance markers
    for i, (idx, row) in enumerate(df_ct.iterrows()):
        if row['fisher_pvalue_adj'] < 0.05:
            marker = '***' if row['fisher_pvalue_adj'] < 0.001 else \
                     '**' if row['fisher_pvalue_adj'] < 0.01 else '*'
            x_pos = row['reversal_score'] + (0.05 if row['reversal_score'] > 0 else -0.05)
            ax.text(x_pos, i, marker, va='center', ha='left' if row['reversal_score'] > 0 else 'right',
                   fontsize=10, weight='bold')
    
    # Set labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_ct['drug'], fontsize=9)
    ax.set_xlabel('Reversal Score', fontsize=10)
    ax.set_title(f'{cell_type} - Drug Ranking by Reversal\n{dataset.upper()}',
                fontsize=11, weight='bold', pad=15)
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    
    # Add reference lines
    ax.axvline(x=0.2, color='green', linestyle='--', linewidth=0.8, alpha=0.3)
    ax.axvline(x=-0.2, color='red', linestyle='--', linewidth=0.8, alpha=0.3)
    
    # Set x limits
    max_abs = max(abs(df_ct['reversal_score'].min()), abs(df_ct['reversal_score'].max()))
    ax.set_xlim(-max_abs * 1.2, max_abs * 1.2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    return fig


def plot_contingency_tables(
    results_df: pd.DataFrame,
    dataset: str,
    n_top: int = 6,
    figsize: tuple = None
) -> plt.Figure:
    """
    Plot contingency tables for top rejuvenating drugs.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from analyze_drug_reversal
    dataset : str
        Dataset name
    n_top : int
        Number of top drugs to show
    figsize : tuple, optional
        Figure size
    
    Returns
    -------
    plt.Figure
        Figure object
    """
    # Filter rejuvenating drugs
    rejuv = results_df[results_df['classification'] == 'rejuvenating'].copy()
    
    if len(rejuv) == 0:
        print("No rejuvenating drugs to plot")
        return None
    
    # Sort by reversal score and take top N
    rejuv = rejuv.sort_values('reversal_score', ascending=False).head(n_top)
    
    # Determine grid layout
    n_drugs = len(rejuv)
    n_cols = min(3, n_drugs)
    n_rows = int(np.ceil(n_drugs / n_cols))
    
    if figsize is None:
        figsize = (n_cols * 3, n_rows * 3)
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_drugs == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each drug
    for i, (idx, row) in enumerate(rejuv.iterrows()):
        ax = axes[i]
        
        # Create contingency table
        table = np.array([
            [row['a'], row['c']],  # Drug increases
            [row['b'], row['d']]   # Drug decreases
        ])
        
        # Plot heatmap
        im = ax.imshow(table, cmap='Blues', aspect='auto', vmin=0)
        
        # Add text annotations
        for i_row in range(2):
            for i_col in range(2):
                value = table[i_row, i_col]
                color = 'white' if value > table.max() * 0.5 else 'black'
                ax.text(i_col, i_row, str(int(value)), 
                       ha='center', va='center',
                       fontsize=14, color=color, weight='bold')
        
        # Labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Age ↑', 'Age ↓'], fontsize=9)
        ax.set_yticklabels(['Drug ↑', 'Drug ↓'], fontsize=9)
        
        # Title with drug info
        title = f"{row['drug']}\n{row['cell_type']}"
        title += f"\nScore: {row['reversal_score']:.2f}"
        title += f"\np={row['fisher_pvalue_adj']:.2e}"
        ax.set_title(title, fontsize=9, weight='bold')
        
        # Highlight reversal cells (b and c)
        rect_b = Rectangle((-.5, 1-.5), 1, 1, fill=False, 
                          edgecolor='green', linewidth=3)
        rect_c = Rectangle((1-.5, 0-.5), 1, 1, fill=False,
                          edgecolor='green', linewidth=3)
        ax.add_patch(rect_b)
        ax.add_patch(rect_c)
    
    # Hide empty subplots
    for i in range(n_drugs, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Contingency Tables - Top Rejuvenating Drugs\n{dataset.upper()}',
                fontsize=12, weight='bold', y=0.98)
    
    plt.tight_layout()
    
    return fig


def plot_reversal_overview(
    results_df: pd.DataFrame,
    dataset: str,
    figsize: tuple = (10, 4)
) -> plt.Figure:
    """
    Plot overview of reversal analysis results.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from analyze_drug_reversal
    dataset : str
        Dataset name
    figsize : tuple
        Figure size
    
    Returns
    -------
    plt.Figure
        Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Panel 1: Classification distribution
    ax = axes[0]
    class_counts = results_df['classification'].value_counts()
    colors_class = {
        'rejuvenating': '#2E7D32',
        'accelerating': '#C62828',
        'neutral': '#757575'
    }
    colors = [colors_class.get(c, '#757575') for c in class_counts.index]
    
    ax.bar(range(len(class_counts)), class_counts.values, color=colors, alpha=0.7)
    ax.set_xticks(range(len(class_counts)))
    ax.set_xticklabels(class_counts.index, rotation=45, ha='right')
    ax.set_ylabel('Number of drug-cell type pairs', fontsize=9)
    ax.set_title('Classification Distribution', fontsize=10, weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Panel 2: Reversal score distribution
    ax = axes[1]
    ax.hist(results_df['reversal_score'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.axvline(x=0.2, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Threshold')
    ax.axvline(x=-0.2, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Reversal Score', fontsize=9)
    ax.set_ylabel('Frequency', fontsize=9)
    ax.set_title('Reversal Score Distribution', fontsize=10, weight='bold')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Panel 3: Scatter plot - reversal count vs p-value
    ax = axes[2]
    
    # Color by classification
    color_map = {
        'rejuvenating': '#2E7D32',
        'accelerating': '#C62828',
        'neutral': '#757575'
    }
    
    for classification in results_df['classification'].unique():
        subset = results_df[results_df['classification'] == classification]
        ax.scatter(
            subset['n_reversal'],
            -np.log10(subset['fisher_pvalue_adj']),
            c=color_map.get(classification, '#757575'),
            label=classification.capitalize(),
            alpha=0.6,
            s=50
        )
    
    ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Number of Reversal TFs', fontsize=9)
    ax.set_ylabel('-log10(adjusted p-value)', fontsize=9)
    ax.set_title('Reversal TFs vs Significance', fontsize=10, weight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.suptitle(f'Drug Reversal Analysis Overview - {dataset.upper()}',
                fontsize=12, weight='bold', y=1.02)
    
    plt.tight_layout()
    
    return fig
