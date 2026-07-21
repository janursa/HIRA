
import os
import numpy as np
import pandas as pd
import json
from hira.src.config import  PRIOR_DIR
def get_opengenes_sets():
    df = pd.read_csv(f'{PRIOR_DIR}/gene-aging-mechanisms.tsv', sep='\t')
    reported_genes = df.index.unique().to_list()
    # Flattened reverse map
    reverse_map = defaultdict(list)
    for idx, row in df.iterrows():
        for cell in row:
            if cell is None or pd.isna(cell) or cell == '':
                continue
            try:
                cell.strip()
            except:
                print(cell)
            for item in cell.split(','):
                k = item.strip('\'"')
                k = k[0].upper() + k[1:]
                reverse_map[k].append(idx)

    # Convert to regular dict if needed
    reverse_map = dict(reverse_map)
    del reverse_map['Transcriptional alterations']

    return reverse_map
def calculate_genes_scores(adata, genes, key='gene_score', min_genes=5):
    genes = [g for g in genes if g in adata.var_names]
    if len(genes) < min_genes:
        raise ValueError("Privided genes list is empty.")
    sc.tl.score_genes(adata, gene_list=genes, score_name=key, use_raw=False)
    return adata
def get_hallmark():
    if False:
        geneset_file = f'{PRIOR_DIR}/h.all.v2024.1.Hs.symbols.gmt'
        genesets_all = read_gmt(geneset_file) 
        genesets_all = {' '.join(key.split('_')[1:]):gs['genes'] for key, gs in genesets_all.items()}
    else:
        geneset_file = f'{PRIOR_DIR}/MSigDB_Hallmark_2020.json'
        if os.path.exists(geneset_file):
            with open(geneset_file, 'r') as f:
                genesets_all = json.load(f)
        else:
            from gseapy import get_library_name, get_library
            genesets_all = get_library(name='MSigDB_Hallmark_2020')
            with open(geneset_file, 'w') as f:
                json.dump(genesets_all, f, indent=4)
    return genesets_all
def get_essential_hallmark():
    hallmark_sets = get_hallmark()
    essential_keywords = ['DNA repair', 'Apoptosis', 'MTORC1', 'G2M', 'E2F', 
                        'Oxidative phosphorylation', 'MYC', 'P53']
    essential_pathways = {k: v for k, v in hallmark_sets.items() if any(keyword.lower() in k.lower() for keyword in essential_keywords)}
    return essential_pathways
def get_essential_genes():
    df = pd.read_csv(f'{PRIOR_DIR}/CRISPRInferredCommonEssentials.csv')
    essential_genes = df['Essentials'].str.extract(r'^(\S+)')[0].tolist()
    essential_gene_set = set(essential_genes)
    return  {'DepMap': list(essential_gene_set)}
def get_genesets(pathway=None):
    if pathway == 'hallmark':
        genesets_all = get_hallmark()
    elif pathway == 'essential_hallmark':
        genesets_all = get_essential_hallmark()
    elif pathway == 'opengenes':
        genesets_all = get_opengenes_sets()
    elif pathway == 'essential':
        genesets_all = get_essential_genes()
    elif pathway is None:
        halmark_sets = get_hallmark()
        opengenes = get_opengenes_sets()
        essential = get_essential_genes()

        genesets_all = {**halmark_sets,  **opengenes, **essential}
    else:
        raise ValueError(f"Unsupported pathway type: {pathway}. Choose 'hallmark' or 'opengenes' or 'essential' or 'essential_hallmark'.")
    return genesets_all
def get_gene2pathway():
    genesets_dict = get_genesets()
    target_genes = np.unique(np.concatenate(list(genesets_dict.values())))

    # Create a DataFrame for pathway annotations
    pathway_assignments = []
    for pathway, genes in genesets_dict.items():
        for gene in genes:
            pathway_assignments.append((gene, pathway))

    pathway_df = pd.DataFrame(pathway_assignments, columns=['gene', 'pathway']).set_index('gene')
    return pathway_df


def pathway_kde_func(stats_df, pathway='hallmark', sets=None, test='wilcoxon', min_genes=10, fdr_method='fdr_bh', feature_col='gene'):
    from statsmodels.stats.multitest import multipletests
    from scipy import stats
    from hira.src.pathway_analysis.util import get_genesets
    genesets = get_genesets(pathway=pathway)
    if sets is not None:
        genesets = {key:value for key, value in genesets.items() if key in sets}

    df_store = []
    for cell in stats_df['cell_type'].unique():
        stats_cell = stats_df[stats_df['cell_type'] == cell]
        results = []
        for gs_name, gs_genes in genesets.items():
            slopes = stats_cell[stats_cell[feature_col].isin(gs_genes)]['slope'].dropna()
            if len(slopes) < min_genes:
                continue

            mean_slope = slopes.mean()

            # One-sided test depending on mean slope direction
            if test == 'wilcoxon':
                alt = 'greater' if mean_slope > 0 else 'less'
                _, p_val = stats.wilcoxon(slopes, alternative=alt)
            elif test == 'ttest':
                _, p_val = stats.ttest_1samp(slopes, 0, nan_policy='omit')
                # For one-sided test, halve the p-value appropriately
                if mean_slope > 0:
                    p_val = p_val / 2 if mean_slope > 0 else 1 - p_val / 2
            else:
                raise ValueError("test must be 'wilcoxon' or 'ttest'")

            results.append({
                'cell_type': cell,
                'gene_set': gs_name,
                'n_genes': len(slopes),
                'mean_slope': mean_slope,
                'p_val': p_val
            })

        df_rr = pd.DataFrame(results)
        if len(df_rr)<=0:
            continue
        df_rr['p_adj'] = multipletests(df_rr['p_val'], method=fdr_method)[1]
        df_store.append(df_rr)
    assert len(df_store) > 0, "No cell types had sufficient genes in any gene set."
    results_df = pd.concat(df_store, ignore_index=True)
    assert len(results_df) > 0, "No results found. Check if gene sets match the targets in the dataframe."

    return results_df.sort_values(['cell_type', 'p_adj'])


def get_canonical_pathways():
    geneset_file = '/home/jnourisa/projs/ongoing/hira/input/prior/h.all.v2024.1.Hs.symbols.gmt'
    genesets_all = read_gmt(geneset_file) 
    genesets_all = {key: gs['genes'] for key, gs in genesets_all.items()}

    # Create a list of gene-to-pathway mappings (one-to-one mapping)
    gene_to_pathway_list = [
        (gene, pathway)
        for pathway, genes in genesets_all.items()
        for gene in genes
    ]

    # Convert the list to a DataFrame
    df_pathway = pd.DataFrame(gene_to_pathway_list, columns=["gene", "pathway"])
    df_pathway = df_pathway.set_index("gene")
    df_pathway['pathway'] = df_pathway['pathway'].str.replace('HALLMARK_','')
    df_pathway['pathway'] = df_pathway['pathway'].str.replace('_',' ')
    df_pathway['pathway'] = df_pathway['pathway'].str.title()

    return df_pathway


def run_ora_local(gene_list, background_genes, gene_sets, min_size=5, max_size=500):
    """
    Perform Over-Representation Analysis (ORA).

    Parameters:
    - gene_list: set or list of input genes
    - background_genes: set or list of background genes (universe)
    - gene_sets: dict of pathway_name -> set/list of genes
    - min_size: minimum gene set size to consider
    - max_size: maximum gene set size to consider

    Returns:
    - pd.DataFrame with columns: Term, Overlap, P-value, Adjusted P-value (FDR), Gene Ratio, Genes
    """

    gene_list = set(gene_list)
    background_genes = set(background_genes)

    results = []

    M = len(background_genes)  # total genes in background
    n = len(gene_list & background_genes)  # overlap between input genes and background

    for term, term_genes in gene_sets.items():
        term_genes = set(term_genes)
        term_genes = term_genes & background_genes  # restrict to universe

        N = len(term_genes)
        if N < min_size or N > max_size:
            continue

        k = len(gene_list & term_genes)  # hits in gene list
        if k == 0:
            continue

        # Hypergeometric test: P(X ≥ k)
        pval = hypergeom.sf(k - 1, M, N, n)

        # Collect data
        results.append({
            "Term": term,
            "Gene Set Size": N,
            "Hits": k,
            "P-value": pval,
            "Gene Ratio": k / n,
            "Genes": list(gene_list & term_genes),
        })

    # Compile and adjust p-values
    df = pd.DataFrame(results)
    if not df.empty:
        df['FDR'] = np.minimum(1.0, df['P-value'] * len(df))  # Benjamini-Hochberg correction (simplified)
        df = df.sort_values("P-value")
    return df

def gsea_func(df, pvalue_col='meta_p_adj', gene_sets=['MSigDB_Hallmark_2020'], feature_col='gene'):
    import gseapy as gp
    # from hira.src.utils.util import get_genesets
    from gseapy import barplot, dotplot
    all_genes = np.loadtxt(f'{PRIOR_DIR}/tf_all.csv', dtype=str)
    # all_genes = np.loadtxt(f'{PRIOR_DIR}/gene_names.txt', dtype=str)
    # gene_sets =  get_genesets()
    res2d_store = []
    for cell_type in df['cell_type'].unique():
        for trend in df['trend'].unique():
            # print(f'{cell_type}-{trend}', flush=True)
            mask = (df['cell_type'] == cell_type) & (df['trend'] == trend)
            if mask.sum() == 0:
                continue
            stats_df = df[mask]
            stats_df = stats_df[[feature_col, pvalue_col]]
            
            genes = stats_df[stats_df[pvalue_col]<0.001][feature_col].unique().tolist()
            if len(genes) == 0:
                continue
            if True:
                rr = gp.enrichr(gene_list=list(genes),
                                gene_sets=gene_sets, #, 'KEGG_2021_Human'
                                organism='human', 
                                outdir=None, 
                                cutoff=1,
                                background=all_genes,
                                )
                res2d = rr.res2d
                res2d.rename(columns={'Adjusted P-value': 'FDR'}, inplace=True)
            else:
                res2d = run_ora_local(genes, background_genes=all_genes, gene_sets=gene_sets, min_size=1, max_size=500)
                res2d['Term'] = (
                res2d['Term']
                    .str.replace('HALLMARK_', '', regex=False)
                    .str.replace('_', ' ', regex=False)
                    # .str.title()
                )

            filter_col = 'FDR' #'FDR q-val'
            res2d = res2d[res2d[filter_col]<0.05]
            res2d['n_genes'] = res2d['Genes'].apply(lambda x: len(x.split(';')))
            res2d = res2d[res2d['n_genes']>=3]
            
            if res2d.shape[0] == 0:
                continue
            print(res2d.shape)
            res2d['cell_type'] = cell_type
            res2d['trend'] = trend
            res2d_store.append(res2d)
    if len(res2d_store) == 0:
        return None
    
    res2d_all = pd.concat(res2d_store)
    res2d_all["neg_log10_adj_pval"] = -np.log10(res2d_all["FDR"])
    return res2d_all


def wrapper_gsea(stats, palette=None, **kwargs):
    """
    Wrapper function for GSEA analysis with visualization.
    
    Parameters
    ----------
    stats : pd.DataFrame
        Statistics dataframe with columns including 'cell_type', 'trend'
    palette : dict, optional
        Color palette for trends
    **kwargs : dict
        Additional arguments passed to gsea_func
    
    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt
    from hira.src.feature_association.plots import dotplot_category_color
    
    if palette is None:
        from hira.src.config import palette_trend_2
        palette = palette_trend_2
    
    # Handle feature_type parameter (convert to feature_col for gsea_func)
    pathway_scores = gsea_func(stats, **kwargs)
    n_terms = pathway_scores['Term'].nunique()
    cell_types = pathway_scores['cell_type'].unique()
    fig, ax = plt.subplots(1, 1, figsize=(len(cell_types)*.12+1, 1+.15*n_terms), sharey=True, sharex=True)

    pathway_scores['cell_type'] = pd.Categorical(pathway_scores['cell_type'], categories=cell_types, ordered=True)
    show_color_legend = True
    show_size_legend = True

    dotplot_category_color(pathway_scores, 
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
    
    return fig, ax
