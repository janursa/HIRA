import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

from hiara.src.config import OUTPUT_DIR, surrogate_names, palette_datasets_pretty
from hiara.src.feature_association.helper import retrieve_sig_stats
from hiara.src.feature_association.helper import retrieve_net_consensus
from hiara.src.config import  colors_blind
from hiara.src.utils.util import get_genesets
import warnings
from matplotlib.patches import Patch
warnings.filterwarnings('ignore')


def barplot_yvalue_tfs(pivot_df, ax=None, color=colors_blind[1], x='tf', y='value', figsize=(4, 2), temp_dir=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    barplot = sns.barplot(
            data=pivot_df,
            y=y,
            x=x,
            ax=ax,
            width=0.7,
            # palette='viridis'
            color=color,
        )
    ax.spines[['top', 'right']].set_visible(False)  # Hide top and right spines
    ax.set_ylabel('Gene score shift \n (pseudo-log2FC)')
    ax.set_xlabel('TFs')
    ax.margins(x=0.05, y=0.05)
    bb = ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')


def heatplot_perturbation_effect(pivot_df, ax, gene_score_shift_col):
    from matplotlib.colors import LinearSegmentedColormap
    from hiara.src.config import palette_trend_2, surrogate_names

    custom_cmap = LinearSegmentedColormap.from_list(
        'aging_effect_cmap',
        [palette_trend_2['Decrease in aging'], 'white', palette_trend_2['Increase in aging']]
    )
    cbar_label = 'Gene score shift \n (pseudo-log2FC)' if 'log2fc' in gene_score_shift_col else 'Gene score shift'
    sns.heatmap(pivot_df, 
                cmap=custom_cmap, 
                center=0, 
                linewidths=0.1, 
                linecolor=None,
                ax=ax,
                cbar_kws={'label': cbar_label, 'shrink': 0.6})
    ax.set_ylabel('')
    ax.set_xlabel('Number of top TFs perturbed')
    ax.set_title('')
    bb = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center')

def disease_tfs():
    pass

def barplot_overlap_n(df_risk_s):
    n_cols = df_risk_s.shape[1]
    fig, axes = plt.subplots(n_cols, 1, figsize=(4, n_cols*1), sharex=False)
    legend_elements = []
    for i, (col, ax) in enumerate(zip(df_risk_s.columns, axes)):
        df = df_risk_s[col].reset_index(name='value')
        color=colors_blind[i]
        barplot_yvalue_tfs(df, ax, color=color)
        ax.set_ylabel('Number of targets')
        if i != len(axes) - 1:
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_xticklabels([])

        legend_elements.append(Patch(facecolor=color, edgecolor='none', label=col))
    fig.legend(handles=legend_elements, bbox_to_anchor=(1.4, .8), frameon=False)


def summarize_pathway_scores(raw_rr, cols=['cell_type', 'dataset', 'tf', 'n']):
    terms = list(get_genesets().keys())
    df_store = []
    for term in terms:
        if term not in raw_rr.columns:
            continue
        df_sub_store = []
        for i, col in enumerate(['baseline_gene_score', 'perturbed_gene_score' ,'gene_score_shift']):
            df = raw_rr.groupby(cols)[term].mean()[term][col].reset_index()
            if i == 0:
                df_sub_store.append(df)
            else:
                df_sub_store.append(df[[col]])
        df = pd.concat(df_sub_store, axis=1)
        df['pathway'] = term
        df_store.append(df)
    score_shift_summary = pd.concat(df_store).reset_index(drop=True)
    baseline_adj = score_shift_summary['baseline_gene_score'].abs().quantile(.25)
    score_shift_summary['gene_score_shift_n'] = score_shift_summary['gene_score_shift'].abs() / (score_shift_summary['baseline_gene_score'] )

    score_shift_summary['gene_score_shift_log2fc'] = np.log2((score_shift_summary['gene_score_shift'].abs() + baseline_adj) / (score_shift_summary['baseline_gene_score'].abs() + baseline_adj))
    
    return score_shift_summary
def determine_overlap_with_risk_genes(cell_type):
    from scipy.stats import fisher_exact, entropy
    from statsmodels.stats.multitest import multipletests
    import pandas as pd

    opengenes_pathways = get_genesets('opengenes')
    essential_genes = get_genesets('essential')
    essential_hallmark = get_genesets('essential_hallmark')
    net = retrieve_net_consensus(cell_type=cell_type)

    # -- Flatten gene sets --
    genes_opengenes = set().union(*opengenes_pathways.values())
    genes_hallmark = set().union(*essential_hallmark.values()) if isinstance(essential_hallmark, dict) else set(essential_hallmark)

    # -- Setup main tracking variables --
    results = []
    per_pathway_data = {}  # (category, term): list of values per TF

    all_targets = set(net['target'])
    all_essentials = set(*essential_genes.values())
    all_opengenes = genes_opengenes
    all_hallmark = genes_hallmark

    # -- Create space for detailed pathway overlap counts --
    all_term_sets = {
        **{('OpenGenes', k): set(v) for k, v in opengenes_pathways.items()},
        **{('Essential hallmark', k): set(v) for k, v in essential_hallmark.items()}
    }
    for (category, term), geneset in all_term_sets.items():
        per_pathway_data[(category, term, 'n_overlap')] = []
        per_pathway_data[(category, term, 'overlap_ratio')] = []

    # -- Main loop per TF --
    for tf, group in net.groupby('source'):
        tf_targets = set(group['target'])
        n_targets = len(tf_targets)

        # Global overlaps
        n_essential = len(tf_targets & all_essentials)
        n_opengenes = len(tf_targets & all_opengenes)
        n_hallmark = len(tf_targets & all_hallmark)

        # Fisher test
        a, b = n_essential, n_targets - n_essential
        c, d = len(all_essentials - tf_targets), len(all_targets - all_essentials - tf_targets)
        odds, pval = fisher_exact([[a, b], [c, d]], alternative='greater')

        # Entropy of TF's target distribution
        tf_entropy = entropy(pd.Series(group['target']).value_counts(normalize=True))

        results.append({
            'tf': tf,
            'n_targets': n_targets,
            'n_essential': n_essential,
            'n_opengenes': n_opengenes,
            'n_hallmark': n_hallmark,
            'entropy': tf_entropy,
            'fisher_pval_essential': pval
        })

        # -- Per pathway overlaps --
        for (category, term), geneset in all_term_sets.items():
            overlap = tf_targets & geneset
            per_pathway_data[(category, term, 'n_overlap')].append(len(overlap))
            per_pathway_data[(category, term, 'overlap_ratio')].append(len(overlap) / len(geneset) if len(geneset) > 0 else 0)

    # -- Compile main TF-level df --
    df = pd.DataFrame(results)
    df['fisher_fdr'] = multipletests(df['fisher_pval_essential'], method='fdr_bh')[1]

    # Normalize global metrics
    for col in ['n_essential', 'n_opengenes', 'n_hallmark', 'entropy']:
        df[f'{col}_norm'] = df[col] / df[col].max()

    # Composite risk score
    df['risk_score'] = df[['n_essential_norm', 'n_opengenes_norm', 'n_hallmark_norm', 'entropy_norm']].mean(axis=1)

    # -- Per pathway detailed df with MultiIndex columns --
    multi_columns = pd.MultiIndex.from_tuples(per_pathway_data.keys(), names=['Category', 'Term', 'Metric'])
    df_per_pathway = pd.DataFrame(per_pathway_data, index=df['tf'])
    df_per_pathway.columns = multi_columns

    # -- Final full dataframe with everything --
    df_risk_full = pd.concat([df.set_index('tf'), df_per_pathway], axis=1).reset_index()
    return df_risk_full

def plot_pathway_set(score_shift_summary, geneset, gene_score_shift_col, cell_type, col_name='tf', xlabel='TFs', temp_dir=None):
    if temp_dir is None:
        temp_dir = OUTPUT_DIR / 'insilico_perturbation' / 'biological_analysis' / 'figures'
        os.makedirs(temp_dir, exist_ok=True)
    genesets = get_genesets(geneset)
    df_summary_opengenes_s = score_shift_summary[score_shift_summary['pathway'].isin(genesets)]
    # df_summary_opengenes_s[gene_score_shift_col] = df_summary_opengenes_s[gene_score_shift_col].abs()
    pivot_df = df_summary_opengenes_s.pivot_table(index='pathway', columns=col_name, values=gene_score_shift_col)
    fig, ax1 = plt.subplots(1, 1, figsize=(pivot_df.shape[1]*.16+1, len(pivot_df) * 0.16+1))
    heatplot_perturbation_effect(pivot_df, ax1, gene_score_shift_col)
    ax1.set_xlabel(xlabel)
    if geneset=='opengenes':
        title = 'OpenGenes'
    elif geneset=='essential_hallmark':
        title = 'Essential hallmark'
    else:
        raise ValueError(f'Unknown geneset: {geneset}')
    plt.title(f'{title}', pad=15, weight='bold')
    plt.savefig(f'{temp_dir}/{geneset}_gene_score_shift.png', bbox_inches='tight', dpi=300, transparent=True)
    # - mean score shift
    mean_pivot_df = pivot_df.abs().mean(axis=0).reset_index(name='value')
    barplot_yvalue_tfs(mean_pivot_df, color=colors_blind[1], x=col_name)
    plt.title(f'{title} (absolute mean)', pad=15, weight='bold')

    # - Baseline gene score + n overlap with targets
    df_baseline_score = df_summary_opengenes_s.groupby(['pathway'])['baseline_gene_score'].mean().reset_index(name='baseline_gene_score')
    targets = retrieve_net_consensus(cell_type=cell_type)['target'].unique().tolist()
    n_overlap = {p: len(np.intersect1d(targets, gs)) for p, gs in genesets.items()}
    n_overlap  = pd.DataFrame(list(n_overlap.items()), columns=['pathway', 'n_overlap'])
    df_merged = df_baseline_score.merge(n_overlap, on='pathway', how='left')
    n_pathways = df_baseline_score['pathway'].nunique()
    fig, axes = plt.subplots(1, 2, figsize=(3, n_pathways * 0.18+1), sharey=True)
    ax = axes[0]
    barplot_yvalue_tfs(df_merged, x='baseline_gene_score', y='pathway', ax=ax, color=colors_blind[0])
    ax.set_xlabel('Baseline gene score')
    ax.set_ylabel('')

    ax = axes[1]
    barplot_yvalue_tfs(df_merged, x='n_overlap', y='pathway', ax=ax, color=colors_blind[3])
    ax.set_xlabel('Overlap with targets')
    ax.set_ylabel('')
    plt.title(title, pad=15, fontsize=10, weight='bold')

def summarize_essential_genes_shift(score_shift_summary, gene_score_shift_col='gene_score_shift_log2fc'):
    genesets = get_genesets('essential')
    df_summary_opengenes = score_shift_summary[score_shift_summary['pathway'].isin(genesets)]
    df_summary_opengenes[gene_score_shift_col] = df_summary_opengenes[gene_score_shift_col].abs()
    df_mean = df_summary_opengenes.groupby('tf')[gene_score_shift_col].median().reset_index(name='value')
    return df_mean

def plot_pathway_score_shift(mean_scores_s, loc=[1.1, 0.1]):
    terms = mean_scores_s['pathway'].unique()

    # Generate a color palette with distinct colors
    terms_palette = dict(zip(
        terms,
        sns.color_palette('tab20', n_colors=len(terms))
    ))
    fig, ax = plt.subplots(figsize=(3, 2.5))
    # Plot each pathway separately to maintain color consistency
    for pathway, df in mean_scores_s.groupby('pathway'):
        color = terms_palette[pathway]
        
        ordered_tfs = df['added_tf'].cat.categories.tolist()
        df = df.set_index('added_tf').reindex(ordered_tfs).reset_index()
        sns.scatterplot(
            data=df,
            x='added_tf',
            y='gene_score_shift_log2fc',
            label=pathway,
            color=color,
            s=30,
            ax=ax,
            alpha=0.8,
        )
        ax.plot(
            df['added_tf'],
            df['gene_score_shift_log2fc'],
            marker='s',
            linestyle='--',
            color=color,
            alpha=0.8,
            linewidth=1.5,
        )

    ax.legend(loc=loc, title='Pathway', frameon=False, markerscale=1.2)
    ax.set_xlabel('TFs added')
    ax.set_ylabel('Gene score shift \n (log2FC)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.margins(x=0.1, y=0.15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
def plot_all(score_shift_summary, gene_score_shift_col, cell_type, ordered_tfs, 
            overvap_risky_genes=True, 
            geneshift_opengenes=True,
            geneshift_hallmark=True,
            ageshift_depmap=True,
            temp_dir=None
            ):
    from hiara.src.utils.util import get_genesets

    score_shift_summary = score_shift_summary[score_shift_summary['cell_type'] == cell_type]
    score_shift_summary['tf'] = pd.Categorical(score_shift_summary['tf'], categories=ordered_tfs, ordered=True)
    
    # --- opengenes
    if geneshift_opengenes:
        plot_pathway_set(score_shift_summary, 'opengenes', gene_score_shift_col, cell_type, temp_dir=temp_dir)
    
    # --- hallmark essential genes
    if geneshift_hallmark:
        plot_pathway_set(score_shift_summary, 'essential_hallmark', gene_score_shift_col, cell_type, temp_dir=temp_dir)
    
    # -- depmap essential genes
    if ageshift_depmap:
        df_mean = summarize_essential_genes_shift(score_shift_summary, gene_score_shift_col)
        df_mean = df_mean[df_mean['tf'].isin(ordered_tfs)]  # filter to top tfs
        print(df_mean['value'].abs().median())
        df_mean['tf'] = pd.Categorical(df_mean['tf'], categories=ordered_tfs, ordered=True)
        barplot_yvalue_tfs(df_mean, color=colors_blind[2])

        plt.title('DepMap essential genes', pad=15)

    # ---- overlap with risk genes 
    if overvap_risky_genes:
        df_risk = determine_overlap_with_risk_genes(cell_type)
        df_risk_s = df_risk[df_risk['tf'].isin(ordered_tfs)].copy()
        df_risk_s['tf'] = pd.Categorical(df_risk_s['tf'], categories=ordered_tfs, ordered=True)
        barplot_overlap_n(df_risk_s[['tf', 'n_targets', 'n_essential', 'n_opengenes', 'n_hallmark']].set_index('tf'))


