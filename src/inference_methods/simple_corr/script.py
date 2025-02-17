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
from util import efficient_melting, basic_qc


def infer_grn(X, gene_names):
    from scipy.stats import spearmanr
    std_devs = sparse_std(X)
    mask_zero_std = std_devs == 0
    gene_names = gene_names[~mask_zero_std]
    X_filtered = X[:, ~mask_zero_std]
    if False:
        corr, _ = spearmanr(X_filtered, nan_policy='raise')
    else:
        print('start corr calculation')
        corr = sparse_corrcoef(X_filtered.T)
        print(corr.shape)
    
    try:
        net = efficient_melting(corr.A, gene_names)
    except:
        net = efficient_melting(corr, gene_names)
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
    net.to_csv(par['prediction'], index=False)

if __name__ == '__main__':
    
    print(par)
    main(par)
