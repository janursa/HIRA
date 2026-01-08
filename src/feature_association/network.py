"""
Network analysis utilities for TF-target interactions.
"""

import pandas as pd
import numpy as np
from ciim.src.utils.util import retrieve_net
from ciim.src.feature_association.helper import retrieve_sig_stats


def select_top_targets(cell_type, tf, datasets, all_targets, n_top=30, feature_type='gene_expression'):
    """
    Select top targets for a TF across datasets based on weight and significance.
    
    Parameters
    ----------
    cell_type : str
        Cell type to analyze
    tf : str
        Transcription factor name
    datasets : list
        List of dataset names
    all_targets : list or array
        List of target genes to consider
    n_top : int, optional
        Number of top targets to return, default 30
    feature_type : str, optional
        Feature type for statistics, default 'gene_expression'
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: source, target, weight, p_value_adj, slope, 
        neg_log10_adj_pval, dataset
    """
    stats_targets = retrieve_sig_stats(data_type='bulk', feature_type=feature_type, race='both')
    
    net_store = []
    for dataset in datasets:
        net = retrieve_net(dataset, cell_type)
        net = net[(net['source'] == tf) & (net['target'].isin(all_targets))].copy()
        
        # Select targets
        stats = stats_targets[
            (stats_targets['target'].isin(all_targets)) & 
            (stats_targets['dataset'] == dataset) & 
            (stats_targets['cell_type'] == cell_type)
        ]
        net = net.merge(
            stats[['target', 'p_value_adj', 'slope']], 
            on='target', 
            how='left'
        )
        net['neg_log10_adj_pval'] = -np.log10(net['p_value_adj'])
        net['dataset'] = dataset

        net_store.append(net)
    
    net_tf = pd.concat(net_store)
    return net_tf
