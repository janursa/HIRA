import sys
import os 
from tqdm import tqdm
import argparse
import anndata as ad
import scanpy as sc 
import numpy as np 
from scipy.stats import spearmanr
import scipy.sparse as sp
import pandas as pd

parser = argparse.ArgumentParser(description='Infer GRN')
parser.add_argument('--rna', type=str, required=True, help='Input AnnData file')
parser.add_argument('--prediction', type=str, required=True, help='Output file')
parser.add_argument('--min_cells_per_gene',type=int, default=100, help='Minimum number of cells per gene')
parser.add_argument('--min_genes_per_cell', type=int, default=10, help='Minimum number of genes per cell')
parser.add_argument('--weight_t', type=float, default=.05, help='Minimum correlation coefficient to retain an edge.') 
args = parser.parse_args()
par = vars(args)

meta = {
    'resources_dir' : 'src/utils/'
}
sys.path.append(meta['resources_dir'])
from util import basic_qc
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

# def efficient_melting(net, pvals, gene_names, alpha=0.05):
#     '''Efficiently extract upper triangle and apply FDR correction'''
#     upper_triangle_indices = np.triu_indices_from(net, k=1)

#     sources = np.array(gene_names)[upper_triangle_indices[0]]
#     targets = np.array(gene_names)[upper_triangle_indices[1]]
#     weights = net[upper_triangle_indices]
#     pvals_flat = pvals[upper_triangle_indices]

#     # FDR correction
#     _, pvals_adj, _, _ = multipletests(pvals_flat, alpha=alpha, method='fdr_bh')

#     # Filter by adjusted p-value
#     mask = pvals_adj < alpha

#     # Create filtered DataFrame
#     data = np.column_stack((sources[mask], targets[mask], weights[mask], pvals_adj[mask]))
#     net_df = pd.DataFrame(data, columns=['source', 'target', 'weight', 'p_adj'])
#     net_df['weight'] = net_df['weight'].astype(float)
#     net_df['p_adj'] = net_df['p_adj'].astype(float)
#     return net_df

# def infer_grn(X, gene_names, alpha=0.05):
#     std_devs = sparse_std(X)
#     mask_zero_std = std_devs == 0
#     gene_names_filtered = gene_names[~mask_zero_std]
#     X_filtered = X[:, ~mask_zero_std]
#     if X_filtered.shape[1] < 2:
#         raise ValueError("Not enough genes with non-zero variance to compute correlation.")

#     print(X_filtered.shape, type(X_filtered))
#     if sp.issparse(X_filtered):
#         X_filtered = X_filtered.todense()

#     corr, p_value = spearmanr(X_filtered, nan_policy='raise')

#     try:
#         net = efficient_melting(corr.A, p_value, gene_names_filtered, alpha)
#     except:
#         net = efficient_melting(corr, p_value, gene_names_filtered, alpha)

#     assert (net['weight'] <= 1).all()
#     return net
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

def infer_grn(X, gene_names):
    from scipy.stats import spearmanr
    std_devs = sparse_std(X)
    mask_zero_std = std_devs == 0
    gene_names = gene_names[~mask_zero_std]
    X_filtered = X[:, ~mask_zero_std]
    if False:
        corr, p_value = spearmanr(X_filtered, nan_policy='raise')
    else:
        print('start corr calculation')
        corr = sparse_corrcoef(X_filtered.T)
        print(corr.shape)
    try:
        net = efficient_melting_full(corr.A, gene_names)
    except:
        net = efficient_melting_full(corr, gene_names)
    assert (net['weight']<=1).all()
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


def main(par):
    print(par['rna'])
    adata = ad.read_h5ad(par['rna'])
    # Subset and QC
    adata = basic_qc(adata, min_cells_per_gene=par['min_cells_per_gene'], min_genes_per_cell=par['min_genes_per_cell'])
    if adata.shape[0] == 0 or adata.shape[1] == 0:
        print('No cells or genes left after filtering.')
        return 

    assert sp.isspmatrix(adata.X)
    
    # Normalize
    X_norm = sc.pp.normalize_total(adata, inplace=False)['X']
    expression_sample = sc.pp.log1p(X_norm, copy=True)
    
    gene_names = adata.var_names

    net = infer_grn(expression_sample, gene_names)
    net['weight'] = pd.to_numeric(net['weight'], errors='coerce')
    net = net[net['weight'].abs() > par['weight_t']]

    tf_all = np.loadtxt(f"/vol/projects/jnourisa/prior/tf_all.csv", dtype=str)
    net = net[net['source'].isin(tf_all)]
    
    if True:
        skeleton = pd.read_csv(f'/vol/projects/jnourisa/prior/skeleton_promotor.csv')
        net['edge'] = net['source'] + '_' + net['target']
        net['promotor_based'] = net['edge'].isin(skeleton['edge'])
        net = net.drop('edge', axis=1)

    net.to_csv(par['prediction'], index=False)

if __name__ == '__main__':
    
    print(par)
    main(par)
