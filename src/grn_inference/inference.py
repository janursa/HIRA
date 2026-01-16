import sys
import os 
from tqdm import tqdm
import argparse
import anndata as ad
import scanpy as sc 
import numpy as np 
import pandas as pd

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from scipy.sparse import issparse
from hiara import PRIOR_DIR


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
    net = pd.DataFrame(data, columns=['source', 'target', 'weight'])
    return net
def efficient_melting_full(net, gene_names):
    '''includes both directions A->B and B->A'''
    n = len(gene_names)
    source, target = np.meshgrid(gene_names, gene_names, indexing='ij')
    weights = net.flatten()
    data = np.column_stack((source.flatten(), target.flatten(), weights))

    df = pd.DataFrame(data, columns=['source', 'target', 'weight'])
    df = df[df['source'] != df['target']]  # remove self-pairs if needed
    return df

def infer_grn(X, gene_names, p_value_filter=True):
    std_devs = sparse_std(X)
    nonzero_mask = std_devs != 0
    print(f"Number of genes removed due to zero variance: {np.sum(~nonzero_mask)}", flush=True)
    gene_names = gene_names[nonzero_mask]
    X_filtered = X[:, nonzero_mask]
    
    if issparse(X_filtered):
        X_filtered = X_filtered.todense().A
    print(f"Computing Spearman correlation for matrix shape: {X_filtered.shape}")
    corr, p_values = spearmanr(X_filtered, nan_policy='raise')
    print(corr.shape)
    
    # Melt correlation and p-value matrices into edge list format
    corr_df = efficient_melting_full(corr, gene_names)
    
    # FDR correction
    if p_value_filter:
        pval_df = efficient_melting_full(p_values, gene_names).rename(columns={'weight': 'p_value'})
        _, fdr_corrected, _, _ = multipletests(pval_df['p_value'], method='fdr_bh')
        corr_df = corr_df[fdr_corrected < 0.05]
        net = corr_df
    else:
        net = corr_df


    assert (net['weight'] <= 1).all(), "Correlation values should be in [-1, 1]"
    return net 
def sparse_std(X):
    from sklearn.preprocessing import StandardScaler
    scalar = StandardScaler(with_mean=False)
    scalar.fit(X)
    X_var = scalar.var_
    return X_var
def sparse_corrcoef(A, B=None):
    if B is not None:
        A = sparse.vstack((A, B), format='csr')
    A = A.astype(np.float64)
    n = A.shape[1]
    # Compute the covariance matrix
    rowsum = A.sum(1)
    centering = rowsum.dot(rowsum.T.conjugate()) / n
    C = (A.dot(A.T.conjugate()) - centering) / (n - 1)
    # The correlation coefficients are given by
    # C_{i,j} / sqrt(C_{i} * C_{j})
    d = np.diag(C)
    coeffs = C / np.sqrt(np.outer(d, d))
    return coeffs


def main(expression_sample, gene_names, weight_t):
    net = infer_grn(expression_sample, gene_names, p_value_filter=True) 
    if False:
        net = net[net['weight'].abs() > weight_t]

    tf_all = np.loadtxt(f"{PRIOR_DIR}/tf_all.csv", dtype=str)
    net = net[net['source'].isin(tf_all)]
    
    if True:
        skeleton = pd.read_csv(f'{PRIOR_DIR}/skeleton_promotor.csv')
        net['edge'] = net['source'] + '_' + net['target']
        net['promotor_based'] = net['edge'].isin(skeleton['edge'])
        net = net.drop('edge', axis=1)

    return net

# if __name__ == '__main__':
    
#     print(par)
#     main(par)
