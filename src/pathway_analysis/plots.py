
import pandas as pd
import matplotlib.pyplot as plt

def plot_pathway_gsea(res_pathways, palette):
    from hira.src.feature_association.plots import dotplot_category_color
    

    n_terms = res_pathways['Term'].nunique()
    cell_types = res_pathways['cell_type'].unique()

    
    fig, ax = plt.subplots(1, 1, figsize=(len(cell_types)*.12+1, 1+.15*n_terms), sharey=True, sharex=True)

    res_pathways['cell_type'] = pd.Categorical(res_pathways['cell_type'], categories=cell_types, ordered=True)
    show_color_legend = True
    show_size_legend = True

    dotplot_category_color(res_pathways, 
                ax, 
                color_col='trend', 
                size_col='neg_log10_adj_pval', 
                x='cell_type', 
                y='Term', 
                palette=palette, sizes=(50, 250),
                show_color_legend=show_color_legend, 
                show_size_legend=show_size_legend,
                size_legend_title='Significance',
                color_legend_title='Pathway activity',
                size_legend_loc=(1.2, 0.35),
                color_legend_loc=(1.02, 0.75),
                y_label='',
                alpha=0.5)
    return fig
def plot_pathway_kde(
    df_aging,
    res_aging,
    df_condition=None,
    res_cond=None,
    sets=None,
    cell_types=None,
    max_height=0.2,
    row_spacing=0.4,
    cell_spacing=1,
    min_genes=10,
    feature_col='gene'):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from hira.src.pathway_analysis.util import get_genesets
    genesets=get_genesets(pathway='hallmark')
    # Helper for significance stars
    def p_to_star(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return "ns"

    # Define cell types
    if cell_types is None:
        cell_types = []
        if df_aging is not None:
            cell_types += list(df_aging['cell_type'].unique())
        if df_condition is not None:
            cell_types += list(df_condition['cell_type'].unique())
        cell_types = sorted(set(cell_types))

    # Define pathway sets to plot
    all_sets = []
    if res_aging is not None:
        all_sets += list(res_aging['gene_set'].unique())
    if res_cond is not None:
        all_sets += list(res_cond['gene_set'].unique())
    common_sets = sorted(set(all_sets))
    print(f"Available pathways from statistical results: {len(common_sets)}")
    if len(common_sets) == 0:
        raise ValueError("No overlapping pathways found.")

    # Filter pathways by minimum gene count per cell type and track skipped ones
    skipped_pathways = {cell: [] for cell in cell_types}
    valid_sets = []
    
    if sets is not None and len(sets) > 0:
        print(f"User selected sets: {sets}")
        print(f"Available pathways: {common_sets[:10]}")  # Show first 10
        common_sets = [gs for gs in common_sets if gs in sets]
        print(f"After filtering by user selection: {len(common_sets)} pathways")
    elif sets is not None and len(sets) == 0:
        print("Empty sets provided, using all available pathways")
    else:
        print("No sets filter applied, using all available pathways")
    
    
    
    for i, gs_name in enumerate(common_sets):
        
            
        pathway_valid = False
        for cell in cell_types:
            # Check aging data
            if df_aging is not None:
                df_a = df_aging[(df_aging['cell_type'] == cell) & (df_aging[feature_col].isin(genesets[gs_name]))]
                genes_a = df_a[feature_col].unique()
                if len(genes_a) >= min_genes:
                    pathway_valid = True
                    break
            
            # Check condition data
            if df_condition is not None:
                df_c = df_condition[(df_condition['cell_type'] == cell) & (df_condition[feature_col].isin(genesets[gs_name]))]
                genes_c = df_c[feature_col].unique()
                if len(genes_c) >= min_genes:
                    pathway_valid = True
                    break
        
        if pathway_valid:
            valid_sets.append(gs_name)
            if i < 3:
                print(f"  ✓ Added {gs_name} to valid sets")
        else:
            if i < 3:
                print(f"  ✗ Skipped {gs_name} - insufficient genes in all cell types")
            # Add to skipped for all cell types
            for cell in cell_types:
                skipped_pathways[cell].append(gs_name)
    
    common_sets = valid_sets
    print(f"Valid pathways after filtering: {len(common_sets)} out of {len(all_sets if 'all_sets' in locals() else 'unknown')}")
    print(f"Valid pathways: {common_sets[:5]}...")  # Show first 5

    # Scale condition slopes globally between -1 and 1
    if df_condition is not None:
        all_vals = df_condition["slope"].values
        max_abs_val = np.nanmax(np.abs(all_vals))
        df_condition["slope_scaled"] = df_condition["slope"] / max_abs_val

    # Plot setup
    n_sets = len(common_sets)
    if n_sets == 0:
        # Create an empty plot with informative message
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f'No pathways found with >= {min_genes} genes.\nTry reducing the min_genes parameter.',
                ha='center', va='center', transform=ax.transAxes, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig, skipped_pathways
    
    fig, ax = plt.subplots(figsize=(len(cell_types) * 1 + 2.5, max(n_sets * row_spacing * 0.9, 2)))
    y_pos = np.arange(n_sets) * row_spacing

    # Main loop over pathways
    for j, gs_name in enumerate(common_sets):
        for i, cell in enumerate(cell_types):
            x_offset = i * cell_spacing 

            ax.axvline(x_offset, color='gray', linestyle='--', lw=0.7, alpha=0.6)

            # Aging
            if (df_aging is not None):
                df_a = df_aging[(df_aging['cell_type'] == cell) & (df_aging[feature_col].isin(genesets[gs_name]))]
                slopes_a = df_a['slope'].dropna().values
                if len(slopes_a) > 1:
                    mean_slope = slopes_a.mean()
                    sign_slope = np.sign(mean_slope)
                    kde = gaussian_kde(slopes_a)
                    x = np.linspace(min(slopes_a), max(slopes_a), 200)
                    y = kde(x)
                    y = y / y.max() * max_height
                    ax.fill_between(x + x_offset, y_pos[j], y_pos[j] + y, color='tomato', alpha=0.6)

                    # significance above top of KDE
                    if res_aging is not None:
                        pvals = res_aging[(res_aging['cell_type'] == cell) &
                                        (res_aging['gene_set'] == gs_name)]['p_adj']
                        if len(pvals):
                            ax.text(
                                x_offset + sign_slope*.2, y_pos[j] + max_height * .5,
                                p_to_star(pvals.iloc[0]),
                                ha='center', va='bottom',
                                color='tomato', fontsize=9
                            )

            # Condition
            if (df_condition is not None):
                df_c = df_condition[(df_condition['cell_type'] == cell) & (df_condition[feature_col].isin(genesets[gs_name]))]
                slopes_c = df_c['slope_scaled'].dropna().values
                if len(slopes_c) > 1:
                    sign_slope = np.sign(slopes_c.mean())
                    kde = gaussian_kde(slopes_c)
                    x = np.linspace(-1, 1, 200)
                    y = kde(x)
                    y = y / y.max() * max_height
                    ax.fill_between(x + x_offset, y_pos[j], y_pos[j] - y, color='seagreen', alpha=0.6)

                    # significance below bottom of KDE
                    pvals = res_cond[(res_cond['cell_type'] == cell) &
                                    (res_cond['gene_set'] == gs_name)]['p_adj']
                    if len(pvals):
                        ax.text(
                            x_offset + sign_slope * .2, y_pos[j] - max_height * .5,
                            p_to_star(pvals.iloc[0]),
                            ha='center', va='top',
                            color='seagreen', fontsize=9
                        )
    x_centers = [i * cell_spacing for i in range(len(cell_types))]
    ax.set_xticks(x_centers)
    ax.set_xticklabels(cell_types)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(common_sets)
    ax.set_ylabel("Pathway")
    ax.set_xlabel("Cell type")
    # ax.set_xlim(-cell_spacing * 0.7, cell_spacing * (len(cell_types) - 0.7))
    ax.margins(x=.2, y=0.1)
    
    # Safe tight_layout with error handling
    try:
        plt.tight_layout()
    except Exception:
        # If tight_layout fails, just adjust margins manually
        plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1)
    
    return fig, skipped_pathways
def plot_term_genes(pathway_scores, cell_type, term):
    pathway_scores_t = pathway_scores[
        (pathway_scores['Term'] == term) &
        (pathway_scores['cell_type'] == cell_type)
    ]
    if pathway_scores_t.empty:
        print(f"No data found for cell type '{cell_type}' and term '{term}'")
        return
    df = (
        pathway_scores_t
        .groupby('trend')['Genes']
        .apply(lambda x: ', '.join(x))
        .reset_index(name='Genes')
        .set_index('trend')
    )
    pp_dict = {}
    every_n_words = 5
    for trend in df.index:
        genes = df.loc[trend, 'Genes'].split(';')
        if trend == 'Increase in aging':
            wrapped = ', \n'.join(
                [', '.join(genes[i:i + every_n_words]) for i in range(0, len(genes), every_n_words)]
            )
        else:
            wrapped = ', '.join(genes)
        pp_dict[trend] = wrapped

    # Actual plot
    from matplotlib.lines import Line2D
    alpha = 0.5
    plt.figure(figsize=(0, 0))
    color_legend = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=color,
               markersize=10, label=pp_dict[trend], alpha=alpha)
        for trend, color in palette_trend_2.items()
        if trend in pp_dict
    ]

    legend = plt.legend(
        title=f'{cell_type}: {term}',
        title_fontsize=10,
        fontsize=10,
        handles=color_legend,
        loc=(1, .1),
        frameon=False
    )
    legend.get_title().set_fontweight('bold')