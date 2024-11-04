
import sys
import subprocess
import os
import anndata as ad
import scanpy as sc 
import os
import anndata as ad
import numpy as np 
import pandas as pd 
import seaborn as sns
from scipy import stats

from scipy.stats import spearmanr
import sys
import matplotlib.pyplot as plt
import scanpy as sc 
# import decoupler as dc 
from scipy.stats import pearsonr
import json
import itertools
import warnings
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import numpy as np
from scipy.stats import spearmanr, t
from statsmodels.stats.multitest import multipletests


sys.path.insert(0, '../')
from task_grn_inference.src.utils.util import basic_qc, read_gmt, quantile_transformation, zscore_transformation
from task_grn_inference.src.process_data.perturbation.normalization.script import normalize_func

def get_genesets():
    geneset_file = 'input/prior/h.all.v2024.1.Hs.symbols.gmt'
    genesets_all = read_gmt(geneset_file) 
    genesets_all = {key:gs['genes'] for key, gs in genesets_all.items()}
    # extract relevant sets 
    gene_sets = {}
    map_dict = {
        'HALLMARK_PI3K_AKT_MTOR_SIGNALING': 'PI3K/AKT/MTOR',
        'HALLMARK_MTORC1_SIGNALING': 'MTORC1',
        'HALLMARK_P53_PATHWAY': 'P53',
        'HALLMARK_TNFA_SIGNALING_VIA_NFKB': 'TNFA/NFKB',
        'HALLMARK_TGF_BETA_SIGNALING': 'TGF-Beta',
        'HALLMARK_WNT_BETA_CATENIN_SIGNALING': 'WNT-Beta Catenin',
        'HALLMARK_OXIDATIVE_PHOSPHORYLATION': 'Oxidative Phos.'
    }
    for key, key_simple in map_dict.items():
        gene_sets[key_simple] = genesets_all[key]
#     return gene_sets
# def get_genesets_andreas():
    geneset_file = 'input/prior/c5.all.v2024.1.Hs.symbols.gmt'
    genesets_all = read_gmt(geneset_file) 
    genesets_all = {key:gs['genes'] for key, gs in genesets_all.items()}
    # extract relevant sets 
    gene_sets_andreas = {}
    map_dict = {'GOMF_ANTIGEN_BINDING': 'Antigen binding', 
                'GOCC_NUCLEOSOME': 'Nucleosome', 
                'GOMF_EXTRACELLULAR_MATRIX_BINDING': 'ECM-binding', 
                'GOCC_RNA_POLYMERASE_II_CORE_COMPLEX': 'RNA polymerase 2'}
    for key, key_simple in map_dict.items():
        gene_sets_andreas[key_simple] = genesets_all[key]

    gene_sets = {**gene_sets, **gene_sets_andreas}
    return gene_sets

def link_to_matrix(df):
    df[['source', 'target']] = df['link'].str.split('_', expand=True)

    sources = df.source.tolist()
    targets = df.target.tolist()
    weights = df.weight.tolist()

    genes = np.unique(sources + targets)

    # Create index mapping for genes
    gene_idx = {gene: idx for idx, gene in enumerate(genes)}
    # Initialize the gene-gene matrix with zeros
    n_genes = len(genes)
    g_g_matrix = np.zeros((n_genes, n_genes))

    # Populate the matrix with weights
    for source, target, value in zip(sources, targets, weights):
        i, j = gene_idx[source], gene_idx[target]
        g_g_matrix[i, j] = value
        g_g_matrix[j, i] = value  # Make it symmetric
    net = pd.DataFrame(g_g_matrix, columns= genes, index=genes)
    return net 

def determine_sig_links(results_df, fold_change_t=2, pvalue_t=0.05):
    abs_min = results_df[['mean', 'mean_ctr']].values.ravel().min()
    results_df['mean_scalled'] = results_df['mean']+np.abs(abs_min)
    results_df['mean_ctr_scalled'] = results_df['mean_ctr']+np.abs(abs_min)

    results_df['fold_change'] = results_df['mean_scalled']/results_df['mean_ctr_scalled']

    flag = (results_df.pvalue<pvalue_t)&((results_df['fold_change']>fold_change_t) | (results_df['fold_change']<(1/fold_change_t)))
    results_df['sig'] = flag
    return results_df
def volcanic_plot_diff(df, diff_t=.1, pvalue_t=0.05):
    # Calculate log fold change and negative log10 p-value
    df['neg_log10_pvalue'] = -np.log10(df['pvalue'])  # Negative log10 of p-value

    # Create the volcano plot
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x='diff', y='neg_log10_pvalue', hue='sig_diff', style='sig', s=100, palette={False: 'grey', True: 'red'})

    # Add labels and title
    # plt.title('Volcano Plot')
    plt.xlabel('Mean Change')
    plt.ylabel('-Log10 P-value')
    plt.axhline(y=-np.log10(pvalue_t), color='red', linestyle='--', label=f'p-value = {pvalue_t}')
    plt.axvline(x=diff_t, color='red', linestyle='-.', label=f'Mean change = {diff_t}')  # Vertical line at x=0
    plt.axvline(x=-diff_t, color='red', linestyle='-.', label=f'Mean change = {diff_t}')  # Vertical line at x=0
    # plt.legend(title='Significance')
    plt.grid()

def volcanic_plot(df, fold_change_t=2, pvalue_t=0.05):
    # Calculate log fold change and negative log10 p-value
    df['log_fold_change'] = np.log2(df['fold_change'])  # Log2 transformation of fold change
    df['neg_log10_pvalue'] = -np.log10(df['pvalue'])  # Negative log10 of p-value

    # Create the volcano plot
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x='log_fold_change', y='neg_log10_pvalue', hue='sig', style='sig', s=100, palette={False: 'grey', True: 'red'})

    # Add labels and title
    # plt.title('Volcano Plot')
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10 P-value')
    plt.axhline(y=-np.log10(pvalue_t), color='red', linestyle='--', label=f'p-value = {pvalue_t}')
    plt.axvline(x=np.log2(fold_change_t), color='red', linestyle='-.', label=f'Fold change = {fold_change_t}')  # Vertical line at x=0
    plt.axvline(x=-np.log2(fold_change_t), color='red', linestyle='-.', label=f'Fold change = {fold_change_t}')  # Vertical line at x=0
    # plt.legend(title='Significance')
    plt.grid()

def calculate_coexp_all(adata_dir='input/adata.h5ad', normalize='sla', corr_method='spearman', write_file='output/coexp_adata_sla_spearman.h5ad', denoise=False, targeted=True):
    
    genesets = get_genesets()
    target_genes = np.unique(np.concatenate(list(genesets.values())))

    adata_all = ad.read_h5ad(adata_dir) 
    if targeted:
        adata_all = adata_all[:, adata_all.var_names.isin(target_genes)]

    i_all = 0
    for i, age_group in enumerate(adata_all.obs.age_group.unique()):
        adata_age_group= adata_all[adata_all.obs.age_group==age_group]
        
        coexp_age_group = calculate_coexp(adata_age_group, corr_method=corr_method, denoise=denoise, normalize=normalize)

        if i_all == 0:
            coexp_all = coexp_age_group
        else:
            coexp_all = pd.concat([coexp_all, coexp_age_group], axis=0).fillna(0)
        i_all+=1

    # to adata
    obs = coexp_all.index
    obs = obs.to_frame().reset_index(drop=True)
    var = pd.DataFrame(index=coexp_all.columns)
    coexp_adata = ad.AnnData(X=coexp_all.values, obs=obs, var=var)
    coexp_adata.write(write_file)


def denoise_func(X):
    from sklearn.decomposition import PCA, TruncatedSVD
    pca = TruncatedSVD(n_components=100) 
    X_pca = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_pca)
    return X_reconstructed

def corr_latent_space(X, n_components):
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_standardized = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X_standardized)  
    W_k = pca.components_.T 
    covariance_matrix = W_k @ W_k.T  
    std_dev = np.sqrt(np.diag(covariance_matrix))
    correlation_matrix = covariance_matrix / np.outer(std_dev, std_dev)
    return correlation_matrix
# Function to compute Pearson correlation matrix and approximate p-values
def fast_pearson_with_pvalues(X_subset):
    n_samples, n_vars = X_subset.shape
    
    # Compute the correlation matrix
    corr_matrix = np.corrcoef(X_subset, rowvar=False)
    
    # Calculate degrees of freedom for the t-distribution
    dof = n_samples - 2
    
    # Compute the t-statistic for each correlation value
    t_stats = np.zeros_like(corr_matrix)
    mask = (np.abs(corr_matrix) < 1)  # Mask for values < 1 in absolute
    t_stats[mask] = corr_matrix[mask] * np.sqrt(dof / (1 - corr_matrix[mask]**2))
    t_stats[~mask] = np.inf  # Set t-stats to infinity for perfect correlations
    
    
    # Convert t-statistics to p-values using survival function (one-sided p-value * 2 for two-tailed)
    p_values = 2 * t.sf(np.abs(t_stats), dof)
    
    return corr_matrix, p_values

def calculate_coexp(adata, layer=None, group='age_donor', corr_method='pearson', denoise=False, normalize='sla'):

        for i_donor, age_donor in enumerate(tqdm(adata.obs[group].unique())):
            adata_age = adata[adata.obs[group].eq(age_donor), :]

            adata_age = basic_qc(adata_age, min_cells_per_gene=100, min_genes_per_cell=10)

            if normalize=='sla':
                sc.pp.normalize_total(adata_age)
                sc.pp.log1p(adata_age)
            elif normalize=='apr':
                sc.experimental.pp.normalize_pearson_residuals(adata_age)

            X_subset = adata_age.X
            try:
                X_subset = X_subset.todense().A
            except:
                X_subset = X_subset
            
            if denoise:
                X_subset = denoise_func(X_subset)
                # corr_matrix = corr_latent_space(X_subset, n_components=100)
   
            if corr_method=='pearson':
                # corr_matrix = np.corrcoef(X_subset.T)
                corr_matrix, p_values = fast_pearson_with_pvalues(X_subset)
            elif corr_method=='spearman':
                corr_matrix, p_values = spearmanr(X_subset, nan_policy='raise')
        
            # - correct the p value
            p_values_flat = p_values.flatten()
            _, p_values_corrected_flat, _, _ = multipletests(p_values_flat, method='fdr_bh')
            p_values_corrected = p_values_corrected_flat.reshape(p_values.shape)

            mask_non_sig =  p_values_corrected>=0.05
            corr_matrix[mask_non_sig] = 0

            # - melt
            net = efficient_melting(corr_matrix, adata_age.var_names)

            net['link'] = ['_'.join(sorted([str(src), str(tgt)])) for src, tgt in zip(net.source, net.target)]

            # Create a MultiIndex from the specified columns in adata_age_donor
            index_df = pd.DataFrame({'age_group':adata_age.obs.age_group.unique(), 'batch_group':adata_age.obs.batch_group.unique(), 'age_donor':adata_age.obs.age_donor.unique(), 'age':adata_age.obs.age.unique(), 'cell_count': [len(adata_age)]})
            index = pd.MultiIndex.from_frame(index_df, names=['age_group', 'batch_group', 'age_donor', 'age', 'cell_count'])

            # Create a DataFrame with weights using the new 'link' and the MultiIndex
            coexp_df = pd.DataFrame([net.weight.values], 
                                    columns=net.link, index=index)

            if i_donor == 0:
                coexp_all = coexp_df
            else:
                coexp_all = pd.concat([coexp_all, coexp_df], axis=0).fillna(0)

        return coexp_all

def sig_test_all(coexp_adata_file='output/coexp_adata_apr_spearman.h5ad', ctr_group='34-', col_contrast='age_group', col_link='link', save_file='output/links_pvalues_vs_34.csv'):
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from scipy import stats
    coexp_adata = ad.read_h5ad(coexp_adata_file)
    coexp_adata.obs.age_group = coexp_adata.obs.age_group.replace('-34', '34-')
    coexp_adata.obs.age_group = coexp_adata.obs.age_group.astype('str').astype('category')
    print(coexp_adata)

    # - keep only that are present in all 
    mask_zeros = (coexp_adata.X==0).any(axis=0)
    coexp_adata = coexp_adata[:, ~mask_zeros]
    print(coexp_adata)
    print(coexp_adata.obs.nunique())
    # - make a df 
    coexp_df = pd.DataFrame(coexp_adata.X, columns=coexp_adata.var_names, index=pd.MultiIndex.from_frame(coexp_adata.obs, names=coexp_adata.obs.columns))

    coexp_long_df = coexp_df.reset_index(level=['batch_group', 'age_group', 'age_donor']).melt(id_vars=['batch_group', 'age_group', 'age_donor'], var_name='link', value_name='weight')
    coexp_long_df.head()

    links = coexp_long_df[col_link].unique()
    contrast_groups = coexp_long_df[col_contrast].unique()

    # Initialize list for storing results
    data_store = []

    # Group by link and batch_group at the start
    grouped = coexp_long_df.groupby([col_link, col_contrast])

    for link in tqdm(links):
        # Compute control (34-) group once
        ctr_values = grouped.get_group((link, ctr_group)).weight.values
        ctr_mean = np.mean(ctr_values)
        ctr_median = np.median(ctr_values)
        # Iterate through age groups
        for group in contrast_groups:
            if group == ctr_group:
                continue
            df_a = grouped.get_group((link, group))
            sample_values = df_a.weight.values
            sample_mean = np.mean(sample_values)
            sample_median = np.median(sample_values)

            # Calculate p-value
            pvalue = stats.mannwhitneyu(ctr_values, sample_values)[1]
            # pvalue = stats.ttest_ind(ctr_values, sample_values)[1]

            # Store results
            data_store.append({
                col_link: link,
                col_contrast: group,
                'pvalue': pvalue,
                'mean': sample_mean,
                'median': sample_median,
                'mean_ctr': ctr_mean,
                'mean_median': ctr_median,
                'diff_median': sample_median - ctr_median,
                'diff': sample_mean-ctr_mean
            })
        # print(pd.DataFrame(data_store))
        # aa

    results_df = pd.DataFrame(data_store)
    results_df.to_csv(save_file)
def exp_plots(groups, cell_type=True):
        
    # - plot dist of sex and cell types
    def norm_size(series):
        normalized = series.value_counts(normalize=True)
        normalized.index.name='index'
        return normalized
    
    cellcount_dist = groups.size().reset_index(name='cell_count')
    donor_dist = groups['donor_id'].nunique().reset_index(name='donor_n')
    age_donor_dist = groups['age_donor'].nunique().reset_index(name='age_donor')
    age2donor_dist = groups.apply(lambda df: df.groupby('donor_id')['age'].nunique()).reset_index(name='count')


    sex_ratio = groups['sex'].apply(norm_size).reset_index(name='ratio')
    sex_ratio = sex_ratio.rename(columns={'index':'sex'})

    cell_type_ratio = groups['cell_type'].apply(norm_size).reset_index(name='ratio')
    cell_type_ratio = cell_type_ratio.rename(columns={'index':'cell_type'})

    cell_count_donors = groups.apply(lambda df: df.groupby('age_donor').size()).reset_index(name='count')
    

    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 7), gridspec_kw={'width_ratios':[1, 1, 1, 2]})


    # distribution of cell count
    ax = axes[0][0]
    sns.barplot(data=cellcount_dist, x='age_group', y='cell_count', ax=ax)
    ax.set_title('Cell count')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)

    # distribution of age 2 donor 
    ax = axes[1][2]
    sns.stripplot(data=age2donor_dist, x='age_group', y='count', ax=ax)
    ax.set_title('Age 2 donor')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)

    # distribution of donors
    ax = axes[0][1]
    sns.barplot(data=donor_dist, x='age_group', y='donor_n', ax=ax)
    ax.set_title('Donor count')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)

    # distribution of age donor
    ax = axes[0][2]
    sns.barplot(data=age_donor_dist, x='age_group', y='age_donor', ax=ax)
    ax.set_title('Age donor count')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)

    # Plot for sex
    ax = axes[1][0]
    sns.barplot(data=sex_ratio, x='age_group', y='ratio', hue='sex', ax=ax)
    ax.set_title('Ratio of Sex per Age Group')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Ratio')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc=(.5,.5))

    # Plot for cell type
    if cell_type:
        ax = axes[0][3]
        sns.barplot(data=cell_type_ratio, x='age_group', y='ratio', hue='cell_type', ax=ax)
        ax.set_title('Ratio of Cell Type per Age Group')
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Ratio')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(loc=(1.1,.2))

    # cell count distribution for donor-age 
    ax = axes[1][1]
    sns.stripplot(data=cell_count_donors, x='age_group', y='count', ax=ax)
    ax.set_title('Counts seg. by donor-age')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Cell count')
    ax.tick_params(axis='x', rotation=45)


    # Adjust layout
    plt.tight_layout()

    # Show the plots
    plt.show()
def plot_centrality_cluster(groups, normalize=True, save_name='output/figs/degree.png', figsize=(4, 20), degree_t = 10, target_groups=['45_54','65_75']):
    def link_to_centrality(df):
        df['weight'] = 1
        gg_mat = link_to_matrix(df)
        # Calculate degree centrality (sum of weights for each node)
        degree = gg_mat.sum(axis=1)  # sum by row to get degree centrality
        
        return degree

    # Apply the function to the groups
    centrality_df = groups.apply(link_to_centrality).reset_index(level=0, name='degree').pivot(columns='age_group', values='degree').fillna(0)

    
    centrality_df = centrality_df[(centrality_df>degree_t)[target_groups].any(axis=1)]

    if normalize:
        # Normalize the data (optional, for better visual contrast)
        df = (centrality_df - centrality_df.min()) / (centrality_df.max() - centrality_df.min())
    else:
        df = centrality_df

    # Create the clustermap with row clustering
    g = sns.clustermap(df, cmap="Blues", row_cluster=True, col_cluster=False, 
                    figsize=figsize, linewidths=0.01, xticklabels=True, yticklabels=True, 
                    cbar_pos=(0.95, 0.2, 0.03, 0.45))
    # g.ax_heatmap.set_position([0.05, 0.05, 0.8, 0.8])
    # Make the x and y tick labels (gene names) smaller
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), fontsize=5)  # Adjust x-axis gene names
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), fontsize=5)  # Adjust y-axis gene names
    plt.tight_layout()
    # g.fig.suptitle('', y=0.8)

    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
def heatmap_pathways(g_g_matrix:pd.DataFrame, pathway_df:pd.DataFrame):
    '''
    g_g_matrix: df with genes names on index and column
    pathway_df: gene to pathway connection df, where index is gene name and pathway column is pathways
    '''
    # Sort the genes based on the pathways for better visualization

    pathway_df = pathway_df[pathway_df.index.isin(g_g_matrix.index)]
    sorted_genes = pathway_df.index
    expanded_g_g_matrix = pd.DataFrame(index=sorted_genes, columns=sorted_genes)

    # Loop over the gene pairs to populate expanded_g_g_matrix from g_g_matrix
    for i, gene_i in enumerate(sorted_genes):
        for j, gene_j in enumerate(sorted_genes):
            # Use g_g_matrix values if both genes are present, otherwise keep NaN
            if gene_i in g_g_matrix.index and gene_j in g_g_matrix.columns:
                expanded_g_g_matrix.iloc[i, j] = g_g_matrix.loc[gene_i, gene_j]
            else:
                expanded_g_g_matrix.iloc[i, j] =0  # or use 0 if you prefer

    # Convert the matrix to numeric, ensuring all values are floats
    expanded_g_g_matrix = expanded_g_g_matrix.apply(pd.to_numeric, errors='coerce')

    # Fill NaNs with a small value or 0 (depending on your preferences)
    expanded_g_g_matrix = expanded_g_g_matrix.fillna(0)  # or use np.nan if you prefer

    pathway_colors = sns.color_palette('Set2', len(pathway_df['pathway'].unique()))
    pathway_color_map = dict(zip(pathway_df['pathway'].unique(), pathway_colors))
    row_colors = pathway_df['pathway'].map(pathway_color_map)

    # Set figure size
    figsize = (10, 10)

    # Create the heatmap with the gene-to-gene connection strengths
    g = sns.clustermap(expanded_g_g_matrix, row_colors=row_colors, col_colors=row_colors, 
                    cmap="coolwarm", linewidths=0.01, xticklabels=True, yticklabels=True, 
                    row_cluster=False, col_cluster=False,
                    cbar_pos=(.1, 0.2, 0.03, 0.6))  # cbar_pos to the left
    # Make the x and y tick labels (gene names) smaller
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), fontsize=4)  # Adjust x-axis gene names
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), fontsize=4)  # Adjust y-axis gene names

    # Adjust the figure size
    g.fig.set_size_inches(figsize)

    # Create a custom legend showing pathway names and their colors
    for pathway, color in pathway_color_map.items():
        plt.plot([], [], marker="o", ms=10, ls="", mec=None, color=color, label=f"{pathway}")

    # Customize plot labels and add the legend on the right
    plt.legend(loc='lower left', title="Pathways", bbox_to_anchor=(30, 0.5), borderaxespad=0)

    # Show the plot
    plt.tight_layout()
def G_plot(df, ax=None):
    import networkx as nx

    df[['source', 'target']] = df['link'].str.split('_', expand=True)
    if 'mean' not in df.columns:
        df['mean'] = 1
    G = nx.from_pandas_edgelist(df, 'source', 'target', edge_attr='mean')
    pos = nx.circular_layout(G)  
    nx.draw(G, pos, with_labels=True, node_size=200, node_color='lightblue', font_size=6, font_weight='normal', edge_color='gray', ax=ax)

def efficient_melting(net, gene_names):
    '''to replace pandas melting'''
    upper_triangle_indices = np.triu_indices_from(net, k=1)

    # Extract the source and target gene names based on the indices
    sources = np.array(gene_names)[upper_triangle_indices[0]]
    targets = np.array(gene_names)[upper_triangle_indices[1]]

    # Extract the corresponding correlation values
    weights = net[upper_triangle_indices]

    # Create a structured array
    data = np.column_stack((targets, sources, weights))

    # Convert to DataFrame
    # print('convert to df')
    net = pd.DataFrame(data, columns=['source', 'target', 'weight'])
    return net
def run_classifier(adata, model_type='GB', confounder='donor_id', covariate='age', normalize=True):
    import lightgbm as lgb
    from sklearn.linear_model import RidgeClassifier, Ridge
    from sklearn.metrics import r2_score, make_scorer, accuracy_score
    from sklearn.model_selection import cross_validate
    from sklearn.preprocessing import MaxAbsScaler 
    import scanpy as sc  
    
    if normalize:
        print('normalize start')
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.scale(adata)
        print('normalize end')
    
    scores = {}
    # classifer for confounder
    y = adata.obs[confounder]
    X = adata.X
    print('Batch classifier')
    if model_type=='ridge':
        model = RidgeClassifier(alpha=1)
        X = MaxAbsScaler().fit_transform(X)
    elif model_type=='GB':
        model = lgb.LGBMClassifier(silent=True, verbose=-1, n_jobs=20)
    else: 
        raise ValueError('Define the classifier')
    
    scoring = {
        'accuracy_score': make_scorer(accuracy_score)
    }
    score = 1 - cross_validate(model, X, y, cv=5, scoring=scoring, return_train_score=False)['test_accuracy_score'].mean()
    scores[confounder] = score
    scores[f"{confounder}_random_classifer_score"] = 1 - (1/y.nunique())
    # regressor for covariate
    print('Age regressor')
    if model_type=='ridge':
        model = Ridge(alpha=1)
        X = MaxAbsScaler().fit_transform(adata.X)
    elif model_type=='GB':
        model = lgb.LGBMRegressor(silent=True, verbose=-1, n_jobs=20)
    else: 
        raise ValueError('Define the classifier')
    y = adata.obs[covariate]
    scoring = {
        'r2_score': make_scorer(r2_score)
    }
    score = cross_validate(model, X, y, cv=5, scoring=scoring, return_train_score=False)['test_r2_score'].mean()
    scores[covariate] = score
    return scores
# run_classifier(adata)
if __name__ == '__main__': # srun --time 01:00:00  --mem 250g python src/helper.py 
    # par = {
    # 'input_dir': 'input/',
    # 'batch1': ['-34_0', '35_44_0', '45_54_0', '55_64_0', '65_75_0'], 
    # 'batch2': ['-34_1', '35_44_1', '45_54_1', '55_64_1', '65_75_1'], 
    
    # 'models': ['-34_0', '35_44_0', '45_54_0', '55_64_0', '65_75_0', '-34_1', '35_44_1', '45_54_1', '55_64_1', '65_75_1'],
    # # 'models': ['35_44_1', '45_54_1'],
    # }
    # subset_data()
    # corr_genesets(par)
    # batch_correction(par)
    # corr_genesets(par)
    for denoise in [False, True]:
        for normalize in ['apr']:
            for corr_method in ['pearson','spearman']:
                calculate_coexp_all(adata_dir='input/adata_bootstrapped.h5ad', normalize=normalize, corr_method=corr_method, write_file=f'output/coexp_adata_{normalize}_{corr_method}_{denoise}.h5ad', denoise=denoise, targeted=True)
                sig_test_all(coexp_adata_file=f'output/coexp_adata_{normalize}_{corr_method}_{denoise}.h5ad', ctr_group='34-', col_contrast='age_group', col_link='link', save_file=f'output/links_pvalues_vs_34_{normalize}_{corr_method}_{denoise}.csv')

    