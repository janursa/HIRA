
# from hira.src.clock.train import wrapper_build_model_cell_type
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import anndata as ad
from hira import MAJOR_CTS, PLOTS_DIR, PRIOR_DIR, CLOCK_TRAINING_COHORTS, CLOCKS_DIR, CLOCK_V, CLOCK_CV_SCORING, TUNE_CLOCK, palette_major_cts, USE_LOCAL_CLOCK, DISCOVERY_COHORTS
from hira import retrieve_net_consensus
from hira.src.network_analysis.plots import dotplot_category_color
from hira.src.config import surrogate_names

def features_stats():
    features_dict = {}
    for cell_type in MAJOR_CTS:
        net = retrieve_net_consensus(cell_type=cell_type)
        print(cell_type, net.shape)
        features_dict[cell_type] = net['target'].unique().tolist()
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
       
    adata = retrieve_feature_data(dataset=dataset, cell_type=cell_type, data_type=data_type, feature_type=feature_type) 
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
            aa = plot_features_vs_datasets(cell_type=cell_type, features=features, feature_type=feature_type, sizes=(90, 100))
def plot_coeff():
    from grnimmuneclock import retrieve_function
    for cell_type in MAJOR_CTS:
        model, gene_names = retrieve_function(cell_type=cell_type, use_local_clocks=USE_LOCAL_CLOCK, version=CLOCK_V)
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

def tf_act_analysis(n_features=5):
    from grnimmuneclock import retrieve_function
    top_genes_dict = {}
    top_weights_dict = {}
    top_tfs_dict = {}
    top_acts_dict = {}
    for cell_type in ['CD8T', 'CD4T']:
        model, gene_names = retrieve_function(cell_type=cell_type, use_local_clocks=USE_LOCAL_CLOCK, version=CLOCK_V)
        coefs = model.named_steps["ridge"].coef_
        
        abs_coefs = np.abs(coefs)
        top_idx = np.argsort(abs_coefs)[-n_features:][::-1]
        top_genes = [str(gene_names[i]) for i in top_idx]
        
        top_genes_dict[cell_type] = top_genes
        top_weights_dict[cell_type] = coefs[top_idx]

        # calculate TF activity
        import decoupler as dc
        net = retrieve_net_consensus(cell_type=cell_type)
        
        if False: # select top genes
            top_idx = np.argsort(abs_coefs)[-100:][::-1]
            top_genes = [gene_names[i] for i in top_idx]
            gene_names = top_genes
            coefs = coefs[top_idx]
        obs = pd.DataFrame({'sample': [1]})
        var = pd.DataFrame({'genes': gene_names})
        var.index = var['genes']

        X=[[float(c) for c in coefs]]
        adata = ad.AnnData(np.asarray(X), obs=obs, var=var)
        dc.mt.ulm(adata, net, tmin=5)
        tf_acts_df = adata.obsm['score_ulm']
        tf_acts_X = tf_acts_df.values[0]

        abs_acts = np.abs(tf_acts_X)
        tfs = tf_acts_df.columns
        top_idx = np.argsort(abs_acts)[-n_features:][::-1]
        top_tfs= [tfs[i] for i in top_idx]

        top_tfs_dict[cell_type] = top_tfs
        top_acts_dict[cell_type] = tf_acts_X[top_idx]
    dataset = DISCOVERY_COHORTS[0]
    print("Top genes dictionary:", top_genes_dict)
    print("Top TFs dictionary:", top_tfs_dict)
    wrapper_trend(top_genes_dict, top_weights_dict, feature_type='gene_expression', dataset=dataset)
    wrapper_trend(top_tfs_dict, top_acts_dict, feature_type='tf_activity', dataset=dataset)

if __name__ == "__main__":
    # plot_coeff()
    # features_dict = features_stats()
    # gsea(features_dict)
    tf_act_analysis(n_features=10)