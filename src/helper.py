
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
import networkx as nx
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
# from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from scipy import stats

import numpy as np
from scipy.stats import spearmanr, t
from statsmodels.stats.multitest import multipletests


cell_type_mapping = {
    "CD4Naive": "CD4+ T cells",
    "CD4TCM": "CD4+ T cells",
    "CD4TEM": "CD4+ T cells",
    "CD4CTL": "CD4+ T cells",
    "Treg": "CD4+ T cells",
    "CD4Proliferating": "CD4+ T cells",
    "CD8Naive": "CD8+ T cells",
    "CD8TCM": "CD8+ T cells",
    "CD8TEM": "CD8+ T cells",
    "TRAV1-2- CD8+ T cells": "CD8+ T cells",
    "CD8Proliferating": "CD8+ T cells",
    "NK": "NK cells",
    "NK_CD56bright": "NK cells",
    "NKProliferating": "NK cells",
    "Bnaive": "B cells",
    "Bmemory": "B cells",
    "Bintermediate": "B cells",
    "Plasmablast": "B cells",

    "CD14Mono": "Myeloid cells",
    "CD16Mono": "Myeloid cells",
    "cDC1": "Myeloid cells",
    "cDC2": "Myeloid cells",
    'MONO': 'Myeloid cells',
    'DC': 'Other',
    "pDC": "Other",
    "ASDC": "Other",

    "gdT": "gd T cells",
    "MAIT": "MAIT cells",
    "HSPC": "Progenitor cells",
    "Platelet": "Platelet",
    "Eryth": "Erythroid cells",
    "ILC": "ILC",
    "Doublet": "Doublet",
    "dnT": "DN T cells",
    "DN T cells": "DN T cells",

}
colors_blind = [
          '#E69F00',  # Orange
          '#56B4E9',  # Sky Blue
          '#009E73',  # Bluish Green
          '#F0E442',  # Yellow
          '#0072B2',  # Blue
          '#D55E00',  # Vermillion
          '#CC79A7']  # Reddish Purple


major_cell_types = ['B cells', 'CD4+ T cells', 'CD8+ T cells', 'Myeloid cells', 'NK cells']
cell_type_palette = {name: color for name, color in zip(major_cell_types, colors_blind[:len(major_cell_types)])}

map_cell_type_genernib = {
        'CD4+ T cells': 'T cells',
        'TRAV1-2- CD8+ T cells': 'T cells',
        'CD8+ T cells': 'T cells',
        'gd T cells': 'T cells',
        'DN T cells': 'T cells',
        'MAIT cells': 'T cells',
        'Progenitor cells': 'Myeloid cells',
        'B cells': 'B cells',
        'NK cells': 'NK cells',
        'Myeloid cells': 'Myeloid cells'
    }

sys.path.insert(0, '../../')
from task_grn_inference.src.utils.util import basic_qc, read_gmt

surrogate_names = {'batch_1':'Batch 1', 'batch_2':'Batch 2', 'all_batches':'All batches', 
                    '34-':'35 below', '35_44':'35-45', '45_54':'45-55', '55_64':'55-65', '65_75':'65-75',
                    'data1_male': 'External validation', 'pbmc_ageing_downsample_male': 'Downsampled data',
                    'data1': 'Dataset 1', 'data7_allTPs_jalil': 'Dataset 2'}

def determine_centrality(net, use_weight=True):
    """
    Determine centrality based on degree or weight.

    Parameters:
        net (pd.DataFrame): DataFrame containing network edges with 'source', 'target', and 'weight' columns.
        use_weight (bool): If True, calculate weighted degree centrality; otherwise, calculate degree centrality.

    Returns:
        pd.DataFrame: DataFrame with centrality values.
    """
    if use_weight:
        net['abs_weight'] = net['weight'].abs()
        G = nx.from_pandas_edgelist(
            net,
            source="source",
            target="target",
            edge_attr="abs_weight",
            create_using=nx.DiGraph()
        )
        centrality = dict(G.degree(weight="abs_weight")) 
        # centrality = nx.degree_centrality(G)  # Use weighted degree centrality
    else:
        G = nx.from_pandas_edgelist(
            net,
            source="source",
            target="target",
            create_using=nx.DiGraph()
        )
        # centrality = dict(G.degree())  # Regular degree centrality (ignores weights)
        centrality = nx.degree_centrality(G)

    # Convert centrality results to DataFrame
    net_centrality = pd.DataFrame.from_dict(centrality, orient="index", columns=["centrality"])
    return net_centrality



# def determine_centrality_all(net_all, col1='g1',col2='g2'):
#     # - calculaye centrality 
#     centrality_data = []
#     for (cell_type, batch_group, age_group), group_df in net_all.groupby(['cell_type', 'batch_group', 'age_group']):
#         G = nx.Graph()
#         G.add_edges_from(group_df[[col1, col1]].values)

#         raw_degrees = dict(G.degree())  # Get raw degree for each gene

#         centrality_data.extend([
#             {
#                 'gene': gene,
#                 'cell_type': cell_type,
#                 'batch_group': batch_group,
#                 'age_group': age_group,
#                 'centrality': degree  # Note: Key changed to 'degree' for clarity
#             }
#             for gene, degree in raw_degrees.items()
#         ])
#     centrality_df = pd.DataFrame(centrality_data)
#     return centrality_df

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



def topology_stats(par):
    from task_grn_inference.src.exp_analysis.helper import Exp_analysis
    exp_objs_dict_dict = {}
    

    exp_objs_dict = {}
    nets_dict = {}

    for model in par['grn_models']:
        par['grn_model'] = f"{par['grn_models_dir']}/{model}.csv"
        if not os.path.exists(par['grn_model']):
            print(model, ' is skipped')
            continue
        net = pd.read_csv(par['grn_model'])

        net = net.drop_duplicates()
        
        nets_dict[model] = net

        
        peak_gene_net = None
        print(model, len(net))
        obj = Exp_analysis(net, peak_gene_net)
        obj.calculate_basic_stats()
        obj.calculate_centrality()

        exp_objs_dict[model] = obj

    # regulatory links
    links_n = {}
    source_n = {}
    target_n = {}
    nets = {}

    for name, obj in exp_objs_dict.items():
        net = obj.net
        if 'cell_type' in net.columns: # for cell specific grn models, take the mean
            n_grn = net.groupby('cell_type').size().mean()
        else:
            n_grn = len(net)

        links_n[name] = n_grn
        source_n[name] = obj.stats['n_source']
        target_n[name] = obj.stats['n_target']
    # Prepare data for plotting
    data = {
        'Model': [],
        'Count': [],
        'Type': []
    }

    # Populate the data dictionary for each metric
    for model in links_n.keys():
        data['Model'].append(model)
        data['Count'].append(links_n[model])
        data['Type'].append('Putative links')

    for model in source_n.keys():
        data['Model'].append(model)
        data['Count'].append(source_n[model])
        data['Type'].append('Putative TFs')

    for model in target_n.keys():
        data['Model'].append(model)
        data['Count'].append(target_n[model])
        data['Type'].append('Putative target genes')

    # Create a DataFrame from the data dictionary
    topology_stats = pd.DataFrame(data)
    return topology_stats

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
def predict_celltype_ratio_from_coexp(reg_type='ridge', save_tag=''):
    # - fit a regression model to predict g-g-corr from cell type ratio
    # Define the input features and targets
    X_df = pd.read_csv('output/feature_importances/X_df.csv', index_col=0)
    X = X_df.values

    Y_df = pd.read_csv('output/feature_importances/Y_df.csv', index_col=0)
    target_links = Y_df.columns
    Y = Y_df.values

    # Initialize K-Fold cross-validation
    kf = KFold(n_splits=5, random_state=32, shuffle=True)
    test_indices_per_fold = [test_index for _, test_index in kf.split(X)]

    # Initialize dictionary to store results
    r2_scores = {}
    feature_importances = {}

    # Scale features
    if reg_type=='ridge':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        model = Ridge(alpha=1)
    else:
        # Set up the LightGBM model with 20 CPUs
        model = LGBMRegressor(n_jobs=20, verbose=-1, min_child_samples=5,       # Relaxing minimum child samples
            min_split_gain=0.01        # Small positive gain for flexibility in splits
        )

    # Iterate over each target link
    for i, target_link in enumerate(tqdm(target_links)):
        Y_i = Y[:, i]
        scores = []
        fold_importances = []

        # Perform cross-validation
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y_i[train_index], Y_i[test_index]

            # Fit the model and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate R² score for this fold
            scores.append(r2_score(y_test, y_pred))

            # Collect the feature importances for each fold
            if reg_type=='ridge':
                fold_importances.append(model.coef_)
            else:
                fold_importances.append(model.feature_importances_)

        # Store R² scores and average feature importances for this target
        r2_scores[target_link] = scores
        feature_importances[target_link] = np.mean(fold_importances, axis=0)

    # Convert the results into DataFrames 
    r2_scores_df = pd.DataFrame(r2_scores)
    feature_importances_df = pd.DataFrame(feature_importances, index=X_df.columns) 


    feature_importances_df.to_csv(f'output/feature_importances/feature_importances_df_{reg_type}{save_tag}.csv')
    r2_scores_df.to_csv(f'output/feature_importances/r2_scores_df_{reg_type}{save_tag}.csv')


if __name__ == '__main__': # srun --time 01:00:00  --mem 250g python src/helper.py 
    # if False: # correlation analysis
    #     for denoise in [False, True]:
    #         for normalize in ['apr']:
    #             for corr_method in ['pearson','spearman']:
    #                 calculate_coexp_all(adata_dir='input/adata_bootstrapped.h5ad', normalize=normalize, corr_method=corr_method, write_file=f'output/coexp_adata_{normalize}_{corr_method}_{denoise}.h5ad', denoise=denoise, targeted=True)
    #                 sig_test_all(coexp_adata_file=f'output/coexp_adata_{normalize}_{corr_method}_{denoise}.h5ad', ctr_group='34-', col_contrast='age_group', col_link='link', save_file=f'output/links_pvalues_vs_34_{normalize}_{corr_method}_{denoise}.csv')

    # if False:
    #     predict_celltype_ratio_from_coexp('GB')
    
    # if True:
    #     batch_correction()
    # create_dataset12()
    diff_corr_all()