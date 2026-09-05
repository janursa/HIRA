import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.sparse import issparse
from hira import PRIOR_DIR


def sparse_std(X):
    from sklearn.preprocessing import StandardScaler
    scalar = StandardScaler(with_mean=False)
    scalar.fit(X)
    X_var = scalar.var_
    return X_var


def bh_cutoff(p, alpha=0.05):
    '''largest p passing Benjamini-Hochberg, ignoring NaNs'''
    ps = np.sort(p, axis=None)  # NaNs sort last
    n = int(np.isfinite(ps).sum())
    ps = ps[:n]
    passed = np.flatnonzero(ps <= alpha * np.arange(1, n + 1) / n)
    return ps[passed[-1]] if passed.size else -np.inf


def infer_grn(X, gene_names, tf_all, p_value_filter=True):
    gene_names = np.asarray(gene_names)
    nonzero_mask = sparse_std(X) != 0
    print(f"Number of genes removed due to zero variance: {np.sum(~nonzero_mask)}", flush=True)
    gene_names = gene_names[nonzero_mask]
    X_filtered = X[:, nonzero_mask]

    if issparse(X_filtered):
        X_filtered = X_filtered.todense().A
    print(f"Computing Spearman correlation for matrix shape: {X_filtered.shape}", flush=True)
    corr, p_values = spearmanr(X_filtered, nan_policy='raise')

    # Self-pairs are not part of the network, so they stay out of the FDR denominator too.
    np.fill_diagonal(p_values, np.nan)
    keep = np.isfinite(p_values) if not p_value_filter else p_values <= bh_cutoff(p_values)

    # Only TFs can be sources, so melt those rows alone: building the full n^2 edge list first
    # costs ~7x the rows and lands them in an object-dtype frame (numpy upcasts str+float).
    rows = np.flatnonzero(np.isin(gene_names, tf_all))
    i, j = np.nonzero(keep[rows])
    net = pd.DataFrame({'source': gene_names[rows][i],
                        'target': gene_names[j],
                        'weight': corr[rows][i, j]})

    assert (net['weight'].abs() <= 1).all(), "Correlation values should be in [-1, 1]"
    return net


def main(expression_sample, gene_names):
    tf_all = np.loadtxt(f"{PRIOR_DIR}/tf_all.csv", dtype=str)
    net = infer_grn(expression_sample, gene_names, tf_all, p_value_filter=True)

    # Annotate motif support; filtering on it is a load-time choice (see retrieve_net).
    net['edge'] = net['source'] + '_' + net['target']
    for col, fname in [('promotor_based', 'skeleton_promotor.csv'), ('skeleton_based', 'skeleton_atac.csv')]:
        skeleton = pd.read_csv(f'{PRIOR_DIR}/{fname}', usecols=['edge'])
        net[col] = net['edge'].isin(skeleton['edge'])
    net = net.drop('edge', axis=1)

    return net
