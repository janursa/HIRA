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
parser.add_argument('--data_type', type=str, default='sc', help='Type of data: bulk or single-cell')
args = parser.parse_args()
par = vars(args)

# meta = {
#     'resources_dir' : 'src/utils/'
# }
# sys.path.append(meta['resources_dir'])
from ciim.src.utils.util import basic_qc
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests


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

def infer_grn(X, gene_names, p_value_filter=False):
    from scipy.stats import spearmanr
    from statsmodels.stats.multitest import multipletests
    from scipy.sparse import issparse

    # Remove genes with zero variance
    std_devs = sparse_std(X)
    nonzero_mask = std_devs != 0
    gene_names = gene_names[nonzero_mask]
    X_filtered = X[:, nonzero_mask]

    if p_value_filter:
        # Compute Spearman correlation and p-values
        
        if issparse(X_filtered):
            X_filtered = X_filtered.todense().A
        corr, p_values = spearmanr(X_filtered, nan_policy='raise')

        # print(corr.shape)
        # print(p_values.shape)
        
        # Melt correlation and p-value matrices into edge list format
        corr_df = efficient_melting_full(corr, gene_names)
        pval_df = efficient_melting_full(p_values, gene_names).rename(columns={'weight': 'p_value'})

        # FDR correction
        _, fdr_corrected, _, _ = multipletests(pval_df['p_value'], method='fdr_bh')
        corr_df = corr_df[fdr_corrected < 0.05]
        net = corr_df

    else:
        print("Start correlation calculation")
        corr = sparse_corrcoef(X_filtered.T)
        print(f"Correlation matrix shape: {corr.shape}")

        # Convert sparse matrix to dense if needed
        if hasattr(corr, 'A'):
            corr = corr.A
        
        net = efficient_melting_full(corr, gene_names)

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


def main(par):
    print(par['rna'])
    adata = ad.read_h5ad(par['rna'])
    # Subset and QC
    data_type = par['data_type']
    if data_type == 'sc':
        adata = basic_qc(adata, min_cells_per_gene=par['min_cells_per_gene'], min_genes_per_cell=par['min_genes_per_cell'])
        if adata.shape[0] == 0 or adata.shape[1] == 0:
            print('No cells or genes left after filtering.')
            return 

        assert sp.isspmatrix(adata.X)
        
        # Normalize
        X_norm = sc.pp.normalize_total(adata, inplace=False)['X']
        expression_sample = sc.pp.log1p(X_norm, copy=True)

    else:
        expression_sample = adata.X
    
    gene_names = adata.var_names
    if False:
        net = infer_grn(expression_sample, gene_names)
        net['weight'] = pd.to_numeric(net['weight'], errors='coerce')
        
    else:
        net = infer_grn(expression_sample, gene_names, p_value_filter=True)
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
