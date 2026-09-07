"""Interpret the trained clocks: feature weights, TF overlap across cell types,
and how clock features relate to the consensus GRNs.

Usage: python src/clock/run_exp_analysis.py   (needs run_train.py first)
Writes: PLOTS_DIR/
"""
# from hira.src.clock.train import wrapper_build_model_cell_type
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hira import MAJOR_CTS, CLOCK_PLOTS_DIR as PLOTS_DIR, PRIOR_DIR, CLOCK_TRAINING_COHORTS, CLOCKS_DIR, CLOCK_V, CLOCK_CV_SCORING, TUNE_CLOCK, palette_major_cts, USE_LOCAL_CLOCK, DISCOVERY_COHORTS
from hira import retrieve_net_consensus
from hira.src.network_analysis.plots import dotplot_category_color
from hira.src.config import surrogate_names, REF_TFA_ANALYSIS, REF_GE_ANALYSIS

def features_stats():
    from grnimmuneclock import retrieve_function
    from hira import get_clock_cell_types

    features_dict = {}
    for cell_type in get_clock_cell_types():
        model, gene_names = retrieve_function(cell_type=cell_type, model_dir=CLOCKS_DIR if USE_LOCAL_CLOCK else None, version=CLOCK_V)
        print(cell_type, len(gene_names))
        features_dict[cell_type] = list(gene_names)
    from hira.src.utils.plots import plot_interactions, create_interaction_df

    interaction_main_df = create_interaction_df(features_dict)
    aa = plot_interactions(interaction_main_df, min_subset_size=200, min_degree=2, color_map=palette_major_cts)
    file_name = f'{PLOTS_DIR}/interactions_targets.png'
    print(f"Saving figure to {file_name}")
    plt.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')
    # plt.title(surrogate_names[race], pad=40, fontsize=10, fontweight='bold')
    return features_dict
def gsea(features_dict):
    import gseapy as gp
    from gseapy import barplot, dotplot
    all_genes = np.loadtxt(f'{PRIOR_DIR}/gene_names.txt', dtype=str)
    gene_sets = 'MSigDB_Hallmark_2020'
    res2d_store = []
    for cell_type, genes in features_dict.items():
        rr = gp.enrichr(gene_list=list(genes),
                    gene_sets=gene_sets, #, 'KEGG_2021_Human'
                    organism='human', 
                    outdir=None, 
                    cutoff=1,
                    # background=list(all_genes),
                    )
        res2d = rr.res2d
        res2d.rename(columns={'Adjusted P-value': 'FDR'}, inplace=True)
        filter_col = 'FDR' #'FDR q-val'
        res2d = res2d[res2d[filter_col]<0.05]
        if res2d.shape[0] == 0:
            continue
        print(res2d.shape)
        res2d['cell_type'] = cell_type
        res2d_store.append(res2d)

    pathway_scores = pd.concat(res2d_store)

    pathway_scores['n_genes'] = pathway_scores['Genes'].str.split(';').apply(len)
    pathway_scores["neg_log10_adj_pval"] = -np.log10(pathway_scores["FDR"])
    
        
    fig, ax = plt.subplots(1, 1, figsize=(2, 8), sharey=True, sharex=True)

    pathway_scores['cell_type'] = pd.Categorical(pathway_scores['cell_type'], categories=MAJOR_CTS, ordered=True)
    show_color_legend = True
    show_size_legend = True

    dotplot_category_color(pathway_scores, 
                ax, 
                color_col='neg_log10_adj_pval', 
                size_col='n_genes', 
                x='cell_type', 
                y='Term', 
                palette='viridis', 
                sizes=(50, 250),
                show_color_legend=show_color_legend, 
                show_size_legend=show_size_legend,
                size_legend_title='Genes',
                color_legend_title='-log10(FDR)',
                color_legend_loc=(1.1, 0.75),
                size_legend_loc=(1.05, 0.35),
                size_legend_scale=1,
                y_label='',
                alpha=0.7)
    file_name = f'{PLOTS_DIR}/enrichment_targets.png'
    print(f"Saving figure to {file_name}")
    plt.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')

def plot_feature_values(cell_type, features, feature_type, dataset='data1', ax=None, show_cbar=True, data_type='bulk', show_ylabels=True):
    from hira.src.feature_association.helper import retrieve_feature_data, bin_feature_values
    from hira.src.feature_association.plots import heatplot_age_trend

    # ponytail: only bulk+major callers exist here, so feature_type maps 1:1 to these analysis names
    analysis_name = {'gene_expression': REF_GE_ANALYSIS, 'tf_activity': REF_TFA_ANALYSIS}[feature_type]
    adata = retrieve_feature_data(dataset=dataset, cell_type=cell_type, analysis_name=analysis_name)
    # print(features)
    # aaa
    adata = adata[:, adata.var_names.isin(features)]
    assert adata.shape[1]>0, f"Features {features} not found in dataset {dataset} for cell type {cell_type}"

    mean_expr = bin_feature_values(adata)
    mean_expr = mean_expr.loc[features]
    ages = sorted(mean_expr.columns)
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 2))

    heatplot_age_trend(mean_expr[ages], cmap='magma' if feature_type=='tf_activity' else 'viridis', 
                        cbar_title = "Gene \n expression" if feature_type == 'gene_expression' else (
                                    "TF \n activity" if feature_type == 'tf_activity' else "Gene score"
                                ),
                        ax=ax, 
                        show_cbar=show_cbar,
                        cbar_kws={
                            "shrink": 1,
                            "aspect": 5,       # Lower values = thicker colorbar (default is ~20)
                            "fraction": 0.1    # Controls the width space the cbar takes in the figure
                        })
    if not show_ylabels:
        ax.set_yticklabels([])
    ax.set_ylabel('')
    ax.set_xlabel('Age')
    
    
from matplotlib.colors import ListedColormap

def wrapper_trend(top_features_dict, top_feature_values_dict, feature_type, dataset=DISCOVERY_COHORTS[0]):
    from matplotlib import patches as mpatches
    for cell_type, features in top_features_dict.items():
        print(cell_type)
        n_features = len(features)
        fig, axes = plt.subplots(1, 2, figsize=(2, .11*n_features + 1), width_ratios=[.1, 1], gridspec_kw={'wspace':0.05})
        # --- Left: weight sign heatmap ---
        ax = axes[0]
        top_weights_values = top_feature_values_dict[cell_type]
        top_weights = np.sign(top_weights_values)  # ±1
        df = pd.DataFrame({'feature': features, 'weight': top_weights, 'abs_weight': np.abs(top_weights_values)})
        df = df.set_index('feature').loc[features]  # enforce order
        features = df.index.tolist()
        cmap = ListedColormap(['#E52B50', '#B0BF1A'])
        # print(df[['weight']])
        sns.heatmap(
            df[['weight']],
            cmap=cmap,
            cbar=False,
            ax=ax,
            annot=False )
        ax.set_yticks(np.arange(len(features)) + 0.5)
        ax.set_yticklabels(features, rotation=0)
        ax.set_xlabel("")
        red_patch = mpatches.Patch(color='#E52B50', label='Negative')
        green_patch = mpatches.Patch(color='#B0BF1A', label='Positive')
        ax.legend(handles=[green_patch, red_patch], bbox_to_anchor=(1.7, 1.6), loc='upper right', borderaxespad=0., frameon=False)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_xticks([])

        ax = axes[1]

        plot_feature_values(cell_type, features, feature_type=feature_type, dataset=dataset, ax=ax, show_cbar=True, data_type='bulk', show_ylabels=True)
        ax.set_yticks([])
        file_name = f'{PLOTS_DIR}/feature_values_{cell_type}_{feature_type}_{dataset}.png'
        print(f"Saving figure to {file_name}")
        plt.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')

        # plot these tfs in natural aging
        if feature_type == 'tf_activity':
            from hira.src.feature_association.plots import plot_features_vs_datasets
            aa = plot_features_vs_datasets(cell_type=cell_type, features=features, analysis_name=REF_TFA_ANALYSIS, sizes=(90, 100), plots_dir=PLOTS_DIR)
def plot_coeff():
    from grnimmuneclock import retrieve_function
    from hira import get_clock_cell_types
    for cell_type in get_clock_cell_types():
        model, gene_names = retrieve_function(cell_type=cell_type, model_dir=CLOCKS_DIR if USE_LOCAL_CLOCK else None, version=CLOCK_V)
        coefs = model.named_steps["ridge"].coef_
        abs_coefs = np.abs(coefs)
        sorted_abs_coefs = np.sort(abs_coefs)[::-1]
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(sorted_abs_coefs)), sorted_abs_coefs)
        ax.set_ylabel('Weights')
        ax.set_xlabel('Genes')
        ax.set_xticks([])
        ax.set_title(f'{cell_type}')
        plt.tight_layout()
        
        file_name = f'{PLOTS_DIR}/clock_coeff_{cell_type}.png'
        print(f"Saving figure to {file_name}")
        plt.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')
        plt.close()

def tf_act_analysis(sig_threshold=0.05, n_repeats=50, min_targets=5, n_display=5):
    from grnimmuneclock import retrieve_function, permutation_gene_importance, tf_activity_from_coefs
    from hira import retrieve_stats, retrieve_sig_stats, retrieve_adata, CLOCK_TEST_COHORTS
    from scipy import sparse
    from scipy.stats import fisher_exact

    dataset = DISCOVERY_COHORTS[0]
    top_genes_dict = {}
    top_weights_dict = {}
    top_tfs_dict = {}
    top_acts_dict = {}
    concordance_dfs = []
    for cell_type in ['CD8T', 'CD4T']:
        model, gene_names = retrieve_function(cell_type=cell_type, model_dir=CLOCKS_DIR if USE_LOCAL_CLOCK else None, version=CLOCK_V)
        coefs = model.named_steps["ridge"].coef_
        clock_df = pd.DataFrame({'gene': gene_names, 'clock_coef': coefs})

        # --- gene selection: out-of-sample permutation-importance p-value per gene, held out
        # on CLOCK_TEST_COHORTS (the same cohorts run_cv.py validates the clock on), scored by
        # Spearman rho. The clock's own gene set is already restricted to age-significant
        # genes at training time (retrieve_adata(only_sig_genes=True)), so there's no need to
        # re-filter for significance here -- just cross-check the two sets still agree below.
        X_parts, y_parts = [], []
        for ds in CLOCK_TEST_COHORTS:
            adata_ds = retrieve_adata(dataset=ds, data_type='bulk', cell_type=cell_type, condition='healthy')
            Xd = pd.DataFrame(adata_ds.X.toarray() if sparse.issparse(adata_ds.X) else np.asarray(adata_ds.X),
                               columns=adata_ds.var_names, index=adata_ds.obs_names).reindex(columns=gene_names, fill_value=0)
            X_parts.append(Xd.values)
            y_parts.append(adata_ds.obs['age'].astype(float).values)
        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)

        padj, spearman_full = permutation_gene_importance(model, X, y, n_repeats=n_repeats)
        clock_df['importance_padj'] = padj
        print(f"[{cell_type}] held-out Spearman rho={spearman_full:.3f} (n={len(y)})")

        # cross-check: every clock feature should be a sig gene from training (a few sig
        # genes can be missing from the clock -- e.g. dropped by the inner-join across
        # CLOCK_TRAINING_COHORTS -- so this is a subset check, not equality). Skip entirely
        # where retrieve_adata fell back to consensus-net targets for too few sig genes
        # (currently B cells only), and skip if sig stats aren't available at all.
        try:
            sig_genes = set(retrieve_sig_stats(analysis_name=REF_GE_ANALYSIS, cell_type=cell_type, multi_cohort=True)['gene'])
        except Exception:
            sig_genes = None
        if sig_genes is not None and cell_type != 'B':
            assert set(gene_names) <= sig_genes, f"{cell_type}: clock has features that aren't significant genes"

        # keep genes with real out-of-sample attribution AND a concordant empirical trend
        # (not re-checking significance here -- see cross-check above)
        emp_ge = retrieve_stats(analysis_name=REF_GE_ANALYSIS, cell_type=cell_type, multi_cohort=True).drop_duplicates(subset='gene').set_index('gene')
        gdf = clock_df.join(emp_ge, on='gene', how='inner')
        gdf['agree'] = np.sign(gdf['clock_coef']) == np.sign(gdf['pooled_rho'])
        final_genes = gdf[(gdf['importance_padj'] < sig_threshold) & gdf['agree']].copy()
        print(f"[{cell_type}] {(gdf['importance_padj'] < sig_threshold).sum()}/{len(gdf)} genes have significant "
              f"out-of-sample attribution; {len(final_genes)} are also concordant with the empirical trend")

        top_genes = final_genes.reindex(final_genes['clock_coef'].abs().sort_values(ascending=False).index).head(n_display)
        top_genes_dict[cell_type] = top_genes['gene'].tolist()
        top_weights_dict[cell_type] = top_genes['clock_coef'].values

        # TF activity from the final gene set's coefficients, against a freshly rebuilt
        # consensus GRN (force=True avoids a stale cached file), restricted to TFs already
        # significant with age when that's available -- else tested against all TFs in the net.
        try:
            age_sig_tfs = set(retrieve_sig_stats(analysis_name=REF_TFA_ANALYSIS, cell_type=cell_type, multi_cohort=True)['gene'].unique())
        except Exception:
            age_sig_tfs = None
        net = retrieve_net_consensus(cell_type=cell_type, force=True)
        net = net[net['target'].isin(set(final_genes['gene']))]
        if age_sig_tfs is not None:
            net = net[net['source'].isin(age_sig_tfs)]

        tf_df = tf_activity_from_coefs(final_genes['gene'].values, final_genes['clock_coef'].values, net, min_targets=min_targets)
        ulm_score = tf_df.set_index('tf')['score']
        ulm_padj = tf_df.set_index('tf')['padj']
        tfs = tf_df['tf']
        print(f"[{cell_type}] {len(age_sig_tfs) if age_sig_tfs is not None else 'all'} candidate TFs -> {len(tfs)} testable on the final gene set "
              f"-> {(ulm_padj < sig_threshold).sum()} ULM-significant")

        # restrict top-n selection to TFs whose ULM score is itself significant
        sig_tfs = ulm_padj[ulm_padj < sig_threshold].index
        if len(sig_tfs) < n_display:
            print(f"[{cell_type}] only {len(sig_tfs)} TFs pass ULM padj<{sig_threshold}; "
                  f"falling back to all {len(tfs)} testable TFs for top-{n_display} selection")
            sig_tfs = tfs
        top_tfs = ulm_score.loc[sig_tfs].abs().sort_values(ascending=False).head(n_display).index.tolist()

        top_tfs_dict[cell_type] = top_tfs
        top_acts_dict[cell_type] = ulm_score.loc[top_tfs].values

        # --- sign-concordance of ULM-inferred regulation vs. empirical aging trend, across testable TFs ---
        # meta-analyzed across the natural-aging discovery cohorts (per-cohort stats files don't exist for these)
        emp_stats_raw = retrieve_stats(analysis_name=REF_TFA_ANALYSIS, cell_type=cell_type, multi_cohort=True)
        emp_stats = emp_stats_raw.groupby('gene').agg(
            slope=('slope', 'mean'), p_value_adj=('meta_p_adj', 'first'), is_significant=('is_significant', 'first')
        )
        common_tfs = ulm_score.index.intersection(emp_stats.index)
        df = pd.DataFrame({
            'cell_type': cell_type,
            'tf': common_tfs,
            'ulm_score': ulm_score.loc[common_tfs].values,
            'ulm_padj': ulm_padj.loc[common_tfs].values,
            'emp_slope': emp_stats.loc[common_tfs, 'slope'].values,
            'emp_padj': emp_stats.loc[common_tfs, 'p_value_adj'].values,
            'emp_sig': emp_stats.loc[common_tfs, 'is_significant'].values,
        })
        df['ulm_sig'] = df['ulm_padj'] < sig_threshold
        df['agree'] = np.sign(df['ulm_score']) == np.sign(df['emp_slope'])
        concordance_dfs.append(df)

        both_sig = df[df['ulm_sig'] & df['emp_sig']]
        if len(both_sig) >= 4:
            table = pd.crosstab(np.sign(both_sig['ulm_score']).astype(int), np.sign(both_sig['emp_slope']).astype(int))
            table = table.reindex(index=[-1, 1], columns=[-1, 1], fill_value=0)
            odds_ratio, p_value = fisher_exact(table.values)
            n_agree = int(both_sig['agree'].sum())
            print(f"[{cell_type}] sign concordance (ULM & empirical both significant): "
                  f"{n_agree}/{len(both_sig)} agree ({n_agree/len(both_sig):.1%}), "
                  f"Fisher's exact OR={odds_ratio:.2f}, p={p_value:.3g}")
        else:
            print(f"[{cell_type}] only {len(both_sig)} TFs significant in both ULM and empirical stats; "
                  f"skipping concordance test (need >=4)")

    concordance_df = pd.concat(concordance_dfs, ignore_index=True)
    file_name = f'{PLOTS_DIR}/tf_regulation_concordance.csv'
    concordance_df.to_csv(file_name, index=False)
    print(f"Saving concordance table to {file_name}")

    print("Top genes dictionary:", top_genes_dict)
    print("Top TFs dictionary:", top_tfs_dict)
    wrapper_trend(top_genes_dict, top_weights_dict, feature_type='gene_expression', dataset=dataset)
    wrapper_trend(top_tfs_dict, top_acts_dict, feature_type='tf_activity', dataset=dataset)
    return concordance_df

if __name__ == "__main__":
    # plot_coeff()
    features_dict = features_stats()
    gsea(features_dict)
    tf_act_analysis(n_display=10)