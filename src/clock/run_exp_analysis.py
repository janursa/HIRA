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
from hira.src.clock.helper import save_clock_stats

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
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
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

    present_cts = [ct for ct in MAJOR_CTS if ct in pathway_scores['cell_type'].unique()]
    pathway_scores['cell_type'] = pd.Categorical(pathway_scores['cell_type'], categories=present_cts, ordered=True)
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
    plt.savefig(file_name, dpi=300, bbox_inches='tight')

def plot_feature_values(cell_type, features, feature_type, dataset='data1', ax=None, show_cbar=True, data_type='bulk', show_ylabels=True, min_age=None, bin_size=5, min_bin_n=None):
    from hira.src.feature_association.helper import retrieve_feature_data, bin_feature_values
    from hira.src.feature_association.plots import heatplot_age_trend

    # ponytail: only bulk+major callers exist here, so feature_type maps 1:1 to these analysis names
    analysis_name = {'gene_expression': REF_GE_ANALYSIS, 'tf_activity': REF_TFA_ANALYSIS}[feature_type]
    adata = retrieve_feature_data(dataset=dataset, cell_type=cell_type, analysis_name=analysis_name)
    # print(features)
    # aaa
    adata = adata[:, adata.var_names.isin(features)]
    if min_age is not None:
        adata = adata[adata.obs['age'] >= min_age]
    assert adata.shape[1]>0, f"Features {features} not found in dataset {dataset} for cell type {cell_type}"

    mean_expr = bin_feature_values(adata, bin_size=bin_size, min_bin_n=min_bin_n)
    # ponytail: clock features aren't restricted to net genes, so a few can be missing here
    missing = [f for f in features if f not in mean_expr.index]
    if missing:
        print(f"  [{cell_type}] dropping {len(missing)} feature(s) absent from {analysis_name}: {missing}")
    features = [f for f in features if f in mean_expr.index]
    mean_expr = mean_expr.loc[features]
    ages = list(mean_expr.columns)  # groupby already emits bins in ascending age order
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
    return features
    
    
from matplotlib.colors import ListedColormap

def top_clock_features(gene_stats, tf_stats, cell_type, feature_type, n=15, sig_threshold=0.05):
    """Top-n features the clock leans on, ranked from the persisted stats.

    Genes rank on |ridge coefficient| and are signed by their empirical aging direction;
    TFs rank on |ULM activity| among the ULM-significant ones.
    Returns (features, weights) with the weight giving the sign shown in the strip.
    """
    if feature_type == 'gene_expression':
        sub = gene_stats[gene_stats['cell_type'] == cell_type]
        top = sub.reindex(sub['clock_coef'].abs().sort_values(ascending=False).index).head(n)
        return top['gene'].tolist(), top['pooled_rho'].values
    df = tf_stats[tf_stats['cell_type'] == cell_type].set_index('tf')
    sig = df[df['padj'] < sig_threshold]
    top = (sig if len(sig) >= n else df)['score'].abs().sort_values(ascending=False).head(n).index
    return list(top), df.loc[top, 'score'].values


def plot_trend_panel(ax_sign, ax_heat, cell_type, features, weights, feature_type,
                     dataset=DISCOVERY_COHORTS[0], show_cbar=True,
                     sign_colors=('#E52B50', '#B0BF1A'), min_age=None, bin_size=5, min_bin_n=None):
    """Sign strip (age direction of the clock weight) next to the binned age-trend heatmap."""
    kept = plot_feature_values(cell_type, features, feature_type=feature_type, dataset=dataset,
                               ax=ax_heat, show_cbar=show_cbar, data_type='bulk', show_ylabels=True,
                               min_age=min_age, bin_size=bin_size, min_bin_n=min_bin_n)
    ax_heat.set_yticks([])

    df = pd.Series(np.sign(weights), index=features).loc[kept].to_frame('weight')
    sns.heatmap(df, cmap=ListedColormap(list(sign_colors)), cbar=False,
                ax=ax_sign, annot=False, vmin=-1, vmax=1)
    ax_sign.set_yticks(np.arange(len(kept)) + 0.5)
    ax_sign.set_yticklabels(kept, rotation=0)
    ax_sign.set(xlabel='', ylabel='')
    ax_sign.set_xticks([])


def wrapper_trend(top_features_dict, top_feature_values_dict, feature_type, dataset=DISCOVERY_COHORTS[0]):
    for cell_type, features in top_features_dict.items():
        print(cell_type)
        n_features = len(features)
        fig, axes = plt.subplots(1, 2, figsize=(2, .11*n_features + 1), width_ratios=[.1, 1], gridspec_kw={'wspace':0.05})
        plot_trend_panel(axes[0], axes[1], cell_type, features, top_feature_values_dict[cell_type],
                         feature_type=feature_type, dataset=dataset)
        file_name = f'{PLOTS_DIR}/feature_values_{cell_type}_{feature_type}_{dataset}.png'
        print(f"Saving figure to {file_name}")
        plt.savefig(file_name, dpi=300, bbox_inches='tight')

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
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()

def clock_gene_stats(cell_type):
    """One row per clock feature: its ridge coefficient and the empirical aging stats for
    the same gene.

    The clock's gene set is already restricted to age-significant genes at training time
    (retrieve_adata(only_sig_genes=True)), so significance isn't re-filtered here -- the
    caller just cross-checks that the two sets still agree.
    """
    from grnimmuneclock import retrieve_function
    from hira import retrieve_stats

    model, gene_names = retrieve_function(cell_type=cell_type, model_dir=CLOCKS_DIR if USE_LOCAL_CLOCK else None, version=CLOCK_V)

    emp_ge = retrieve_stats(analysis_name=REF_GE_ANALYSIS, cell_type=cell_type, multi_cohort=True).drop_duplicates(subset='gene').set_index('gene')
    gdf = pd.DataFrame({'gene': gene_names,
                        'clock_coef': model.named_steps['ridge'].coef_}).join(emp_ge, on='gene', how='inner')
    # magnitude from the ridge weight, direction from the empirical age trend: under
    # collinearity ridge hands correlated partners suppressor weights whose sign opposes
    # their own age trend, which would invert the inferred TF direction.
    gdf['signed_coef'] = gdf['clock_coef'].abs() * np.sign(gdf['pooled_rho'])
    return gene_names, gdf


def plot_clock_vs_empirical_consistency(concordance_df, save_suffix='clock_vs_empirical'):
    """One panel per cell type: signed significance of the clock-derived (ULM) TF activity
    against the empirical age trend, restricted to empirically significant TFs.
    """
    from hira.src.feature_association.plots import plot_directional_consistency_scatter

    df = concordance_df[concordance_df['emp_sig']].rename(columns={'tf': 'gene'}).copy()
    stats = df.assign(comparison='clock', slope=df['ulm_score'], p_value_adj=df['ulm_padj'])
    stats_ref = df.assign(comparison='empirical', slope=df['emp_slope'], meta_p_adj=df['emp_padj'])
    plot_directional_consistency_scatter(
        stats[['cell_type', 'gene', 'comparison', 'slope', 'p_value_adj']],
        stats_ref[['cell_type', 'gene', 'comparison', 'slope', 'meta_p_adj']],
        x_label='Clock-derived TF activity \n(signed significance)',
        y_label='Empirical TF activity \n(signed significance)',
        output_dir=PLOTS_DIR,
        save_suffix=save_suffix,
    )


def tf_act_analysis(sig_threshold=0.05, min_targets=10, n_display=15, n_genes=15):
    """Clock -> TF activity by ULM on the signed ridge-coefficient profile.

    The profile is each clock gene's |ridge coefficient| signed by its empirical aging
    direction, so a TF scores positive when the genes it regulates both weigh on the clock
    and rise with age. The sign has to come from the empirical trend rather than from the
    coefficient: under collinearity ridge gives correlated partners suppressor weights whose
    sign opposes their own age trend, which inverts the inferred TF direction.
    """
    from grnimmuneclock import tf_activity_from_coefs
    from hira import retrieve_stats, retrieve_sig_stats
    from scipy.stats import fisher_exact

    dataset = DISCOVERY_COHORTS[0]
    top_genes_dict = {}
    top_weights_dict = {}
    top_tfs_dict = {}
    top_acts_dict = {}
    concordance_dfs = []
    gene_stats, tf_stats = [], []
    for cell_type in ['CD8T', 'CD4T']:
        gene_names, gdf = clock_gene_stats(cell_type)

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

        # trend heatmap: the genes the clock leans on hardest, coloured by their age direction
        gene_stats.append(gdf.assign(cell_type=cell_type)[['cell_type', 'gene', 'clock_coef',
                                                            'signed_coef', 'pooled_rho']])
        top_genes_dict[cell_type], top_weights_dict[cell_type] = top_clock_features(
            gene_stats[-1], None, cell_type, 'gene_expression', n=n_genes)

        # ULM on the signed-importance profile against a freshly rebuilt consensus GRN
        # (force=True avoids a stale cached file). All TFs in the net are tested -- no
        # pre-filtering to age-significant ones, or the concordance check below is circular.
        net = retrieve_net_consensus(cell_type=cell_type, force=True)
        net = net[net['target'].isin(set(gdf['gene']))]

        tf_df = tf_activity_from_coefs(gdf['gene'].values, gdf['signed_coef'].values, net, min_targets=min_targets)
        tf_stats.append(tf_df.assign(cell_type=cell_type)[['cell_type', 'tf', 'score', 'padj']])
        ulm_score = tf_df.set_index('tf')['score']
        ulm_padj = tf_df.set_index('tf')['padj']
        tfs = tf_df['tf']
        print(f"[{cell_type}] {len(tfs)} TFs testable on the clock gene set "
              f"-> {(ulm_padj < sig_threshold).sum()} ULM-significant")

        # restrict top-n selection to TFs whose ULM score is itself significant
        top_tfs, top_acts = top_clock_features(None, tf_stats[-1], cell_type, 'tf_activity',
                                               n=n_display, sig_threshold=sig_threshold)
        top_tfs_dict[cell_type] = top_tfs
        top_acts_dict[cell_type] = top_acts

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

    plot_clock_vs_empirical_consistency(concordance_df)

    print("Top genes dictionary:", top_genes_dict)
    print("Top TFs dictionary:", top_tfs_dict)
    # persist the full per-gene and per-TF tables so assemble_figs can redo the top-n
    # selection itself without a clock rerun
    save_clock_stats(pd.concat(gene_stats, ignore_index=True), 'clock_gene_importance')
    save_clock_stats(pd.concat(tf_stats, ignore_index=True), 'clock_tf_activity')
    wrapper_trend(top_genes_dict, top_weights_dict, feature_type='gene_expression', dataset=dataset)
    wrapper_trend(top_tfs_dict, top_acts_dict, feature_type='tf_activity', dataset=dataset)
    return concordance_df

def tf_ora_analysis(sig_threshold=0.05, min_targets=5, cell_types=('CD8T', 'CD4T')):
    """Weight-free clock -> TF inference, and its agreement with the empirical TF-activity
    analysis.

    Splits the clock's features by their empirical aging direction and asks which regulons
    are overrepresented in each half (background: all clock features). ORA reads only set
    membership, so it is a check on the ULM result that the ridge weights cannot bias.

    Writes PLOTS_DIR/tf_ora_clock_vs_empirical.csv plus a concordance and a discrepancy plot.
    """
    from grnimmuneclock import regulon_ora
    from hira import retrieve_stats, retrieve_sig_stats
    from hira.src.clock.plots import plot_tf_ora_concordance, plot_tf_ora_discrepancy

    hits_all = []
    for cell_type in cell_types:
        _, gdf = clock_gene_stats(cell_type)

        net = retrieve_net_consensus(cell_type=cell_type, force=True)
        net = net[net['target'].isin(set(gdf['gene']))]

        emp = retrieve_stats(analysis_name=REF_TFA_ANALYSIS, cell_type=cell_type, multi_cohort=True).groupby('gene').agg(
            emp_slope=('slope', 'mean'), emp_padj=('meta_p_adj', 'min'))
        emp_sig = set(retrieve_sig_stats(analysis_name=REF_TFA_ANALYSIS, cell_type=cell_type, multi_cohort=True)['gene'].unique())

        for direction, sign in [('up', 1), ('down', -1)]:
            genes = gdf[np.sign(gdf['pooled_rho']) == sign]['gene']
            hits = regulon_ora(genes, gdf['gene'], net, min_targets=min_targets)
            hits = hits.rename(columns={'pval': 'ora_pval', 'padj': 'ora_padj'}).join(emp, on='tf')
            hits.insert(0, 'cell_type', cell_type)
            hits.insert(1, 'direction', direction)
            hits['n_genes'] = len(genes)
            hits['ora_sig'] = hits['ora_padj'] < sig_threshold
            hits['emp_sig'] = hits['tf'].isin(emp_sig)
            hits['dir_match'] = np.sign(hits['emp_slope']) == sign
            hits_all.append(hits)

    ora_df = pd.concat(hits_all, ignore_index=True)[
        ['cell_type', 'direction', 'tf', 'n_genes', 'n_targets', 'n_hit', 'odds',
         'ora_pval', 'ora_padj', 'ora_sig', 'emp_slope', 'emp_padj', 'emp_sig', 'dir_match']]
    file_name = f'{PLOTS_DIR}/tf_ora_clock_vs_empirical.csv'
    ora_df.to_csv(file_name, index=False)
    print(f"Saving clock-vs-empirical TF table to {file_name}")

    for (cell_type, direction), g in ora_df[ora_df['ora_sig']].groupby(['cell_type', 'direction'], sort=False):
        both = g[g['emp_sig']]
        print(f"[{cell_type}] {direction}: {len(g)} ORA-significant TFs; "
              f"{len(both)}/{len(g)} ({len(both)/len(g):.0%}) also empirically significant; "
              f"{int(both['dir_match'].sum())}/{len(both)} agree in direction"
              if len(both) else f"[{cell_type}] {direction}: {len(g)} ORA-significant TFs, none empirically significant")

    plot_tf_ora_concordance(ora_df, PLOTS_DIR)
    plot_tf_ora_discrepancy(ora_df, PLOTS_DIR)
    return ora_df


if __name__ == "__main__":
    # plot_coeff()
    features_dict = features_stats()
    gsea(features_dict)
    tf_act_analysis()
    tf_ora_analysis()