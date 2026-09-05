"""DE-first TF activity: differential expression on consensus-GRN target genes, then ULM
on the per-gene contrast statistic. Avoids the rank-1 collapse of testing per-sample TF
activity (all TFs share the same gene space -> near-identical contrasts)."""
import argparse, os
import numpy as np, pandas as pd, anndata as ad, decoupler as dc

from hira.src.config import get_config, get_config_fa, TF_MIN_TARGET, CONSENSUS_MIN_DEGREE, DISCOVERY_COHORTS, HIRA_DIR
from hira.src.utils.util import retrieve_adata, retrieve_net
from hira.src.feature_association.helper import associate_with_condition, retrieve_feature_data

CACHE = f'{HIRA_DIR}/temp/consensus_skeleton'


def skeleton_consensus(cell_type, min_degree=CONSENSUS_MIN_DEGREE):
    """Consensus GRN built from skeleton (ATAC+motif) edges only. Cached separately so the
    repo-wide consensus_net_*.csv (built under NET_SKELETON=None) stays untouched."""
    os.makedirs(CACHE, exist_ok=True)
    f = f'{CACHE}/consensus_net_{cell_type}.csv'
    if os.path.exists(f):
        return pd.read_csv(f)
    nets = pd.concat([retrieve_net(d, cell_type, skeleton='skeleton').assign(dataset=d) for d in DISCOVERY_COHORTS])
    nets['link'] = nets['source'] + '_' + nets['target']
    sign_ok = nets.groupby('link')['weight'].apply(lambda x: np.sign(x).nunique() == 1)
    deg_ok = nets.groupby('link')['dataset'].nunique() >= min_degree
    nets = nets[nets['link'].isin(sign_ok[sign_ok].index) & nets['link'].isin(deg_ok[deg_ok].index)]
    net = nets.groupby(['source', 'target'], as_index=False)['weight'].mean()
    net.to_csv(f, index=False)
    return net


def de_stats(dataset, cell_type, net, data_type='bulk'):
    """Per-gene differential expression (dataset's own test_type) restricted to GRN targets."""
    config = get_config(dataset)
    adata = retrieve_adata(dataset=dataset, data_type=data_type, cell_type=cell_type)
    adata = adata[:, adata.var_names.isin(net['target'].unique())].to_memory().copy()
    adata.obs['dataset'] = dataset
    return adata, associate_with_condition(adata, config, test_type=config.test_type)


def tfa_from_de(de, net, tmin=TF_MIN_TARGET, method='mlm'):
    """Enrichment of each contrast's signed -log10(p) gene vector against the GRN.

    method='ulm' fits one TF at a time -- with a dense GRN every TF picks up the same
    global shift, which is the rank-1 collapse. 'mlm' fits all TFs jointly, so shared
    targets are partialled out."""
    fn = getattr(dc.mt, method)
    out = []
    for comp, d in de.groupby('comparison'):
        d = d.drop_duplicates('gene').set_index('gene')
        stat = np.sign(d['slope']) * -np.log10(d['p_value'].clip(lower=1e-300))
        a = ad.AnnData(X=stat.values[None, :].astype(float), var=pd.DataFrame(index=stat.index),
                       obs=pd.DataFrame(index=[comp]))
        fn(a, net, tmin=tmin)
        out.append(pd.DataFrame({'gene': a.obsm[f'score_{method}'].columns,
                                 'slope': a.obsm[f'score_{method}'].values[0],
                                 'p_value_adj': a.obsm[f'padj_{method}'].values[0],  # dc already BH-adjusts per row
                                 'comparison': comp}))
    return pd.concat(out, ignore_index=True)


def prune_redundant_tfs(net, tol=1e-8):
    """Keep a maximal set of TFs whose target-weight vectors are linearly independent.

    The consensus GRN is rank-deficient (many TFs share nearly the same targets), which is
    what collapses every per-TF test onto one factor and makes a joint (MLM) fit impossible.
    Rank-revealing QR picks the independent columns; the dropped TFs are linear combinations
    of the kept ones, i.e. not separately identifiable from this network at all."""
    from scipy.linalg import qr
    W = net.pivot_table(index='target', columns='source', values='weight', fill_value=0)
    _, r, piv = qr(W.values, mode='economic', pivoting=True)
    d = np.abs(np.diag(r))
    keep = W.columns[piv[:int((d > d[0] * tol).sum())]]
    return net[net['source'].isin(keep)]


def net_effective_rank(net):
    """exp(entropy of the singular-value spectrum) of the TF x target weight matrix.
    Near 1 => the GRN is effectively one factor and every TF sees the same signal."""
    W = net.pivot_table(index='source', columns='target', values='weight', fill_value=0).values
    s = np.linalg.svd(W, compute_uv=False)
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum())), W.shape


def pc1_var(X):
    from scipy.sparse import issparse
    X = np.asarray(X.toarray() if issparse(X) else X, dtype=float)
    X = X - X.mean(0)
    s = np.linalg.svd(X, compute_uv=False)
    return float(s[0] ** 2 / (s ** 2).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='CXCL9')
    p.add_argument('--cell-types', nargs='+', default=['CD4T', 'CD8T'])
    p.add_argument('--old-analysis', default='tfa_major_b', help='cached per-sample TF activity, for the rank-1 comparison')
    p.add_argument('--method', default='mlm', choices=['mlm', 'ulm'])
    p.add_argument('--prune-redundant', action='store_true',
                   help='drop TFs that are linear combinations of others (required for mlm)')
    p.add_argument('--out', default=f'{HIRA_DIR}/temp/de_then_tfa')
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for ct in args.cell_types:
        net = skeleton_consensus(ct)
        if args.prune_redundant:
            n0 = net.source.nunique()
            net = prune_redundant_tfs(net)
            print(f'{ct}: pruned {n0} -> {net.source.nunique()} linearly independent TFs')
        adata, de = de_stats(args.dataset, ct, net)
        stats = tfa_from_de(de, net, method=args.method)
        stats['cell_type'], stats['dataset'] = ct, args.dataset
        de.to_csv(f'{args.out}/de_{args.dataset}_{ct}.csv', index=False)
        stats.to_csv(f'{args.out}/tfa_from_de_{args.method}_{args.dataset}_{ct}.csv', index=False)

        old = retrieve_feature_data(dataset=args.dataset, cell_type=ct, analysis_name=args.old_analysis)
        print(f'\n=== {args.dataset} {ct} | skeleton consensus: {net.source.nunique()} TFs, '
              f'{net.target.nunique()} targets, {len(net)} edges ===')
        erank, shape = net_effective_rank(net)
        print(f'PC1 %var  old per-sample TF activity : {100*pc1_var(old.X):.1f}%'
              f'  | GRN-target gene expression: {100*pc1_var(adata.X):.1f}%')
        print(f'GRN effective rank: {erank:.1f} of {min(shape)} ({shape[0]} TFs x {shape[1]} targets)')
        for comp, g in stats.groupby('comparison'):
            sig = g[g.p_value_adj < 0.05]
            ng = de[(de.comparison == comp) & (de.p_value_adj < 0.05)]
            frac = 0 if len(sig) == 0 else max((sig.slope > 0).mean(), (sig.slope < 0).mean())
            print(f'  {comp:28s} DE genes {len(ng):5d}/{de.comparison.eq(comp).sum():5d} | '
                  f'TFs {len(sig):4d}/{len(g):4d} ({100*len(sig)/len(g):.0f}%) | same-sign {100*frac:.0f}%')
        print(f'  -> {args.out}/tfa_from_de_{args.method}_{args.dataset}_{ct}.csv')


if __name__ == '__main__':
    main()
