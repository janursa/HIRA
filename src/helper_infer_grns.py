
import sys
import os 
from tqdm import tqdm
import argparse
import anndata as ad

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.join(parent_directory, '../'))
from src.helper_infer_grns import infer_grns_all
from src.helper_dataset import process_dataset


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
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from scipy.stats import spearmanr
import sys
import matplotlib.pyplot as plt
import scanpy as sc 
# import decoupler as dc 
import json
import warnings
from tqdm import tqdm
from scipy import stats
import numpy as np
from scipy.stats import spearmanr, t
from concurrent.futures import ProcessPoolExecutor
from functools import partial

sys.path.insert(0, '../')
from task_grn_inference.src.utils.util import basic_qc
sys.path.insert(0, './')
from src.helper import efficient_melting



import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

def enrich_tf_local(net, adata_bulk, tf_all):
    net = net.pivot(index='source', columns='target', values='weight').fillna(0)
    net = net[[g for g in adata_bulk.var_names if g in net.columns]]
    tfs_present = np.intersect1d(net.index, tf_all)
    net = net[net.index.isin(tfs_present)]
    print('ratio of porosity: ', (net==0).sum().sum()/net.size)
    # - subset the adata
    adata_bulk = adata_bulk[:, adata_bulk.var_names.isin(net.columns)]
    # - enrich tfs 
    mat = adata_bulk.X.todense().T
    
    tf_acts = np.dot(net, mat)
    # - format
    tf_acts = pd.DataFrame(tf_acts, index=net.index, columns=adata_bulk.obs.index)
    tf_acts = tf_acts.reset_index().melt(id_vars='source', var_name='sample', value_name='activity')
    tf_acts = tf_acts.set_index('sample').merge(adata_bulk.obs[['cell_type', 'donor_id', 'cell_count', 'age']], left_index=True, right_index=True).reset_index(drop=False)
    print(f"net: {net.shape}, mat: {mat.shape}, source: {tf_acts['source'].nunique()}")
    # Calculate ranks within each sample
    tf_acts['rank'] = tf_acts.groupby('sample')['activity'].transform(lambda x: x.abs().rank(method='dense', ascending=False)) 

    return tf_acts





def enrich_tfs(adata_sample, net, tf_all):
    import decoupler
    from scipy.stats import zscore

    # - pseudobulk cell type-donor
    sys.path.insert(0, '../')
    from task_grn_inference.src.process_data.perturbation.opsca.script import sum_by

    adata_sample.obs['sum_by'] = '_' + adata_sample.obs['cell_type'].astype(str) + '_' + adata_sample.obs['donor_id'].astype(str)
    adata_sample.obs['sum_by'] = adata_sample.obs['sum_by'].astype('category')
    adata_bulk = sum_by(adata_sample, 'sum_by', unique_mapping=False)
    cell_count_df = adata_sample.obs.groupby('sum_by').size().reset_index(name='cell_count')
    adata_bulk.obs = adata_bulk.obs.merge(cell_count_df, on='sum_by')
    adata_bulk = adata_bulk[adata_bulk.obs['cell_count']>par['n_cells_t']]

    # - normalize
    if par['normalize']:
        sc.pp.normalize_total(adata_bulk)
        sc.pp.log1p(adata_bulk)

    # -enrich TFs
    net = net[net['source'].isin(tf_all)]

    if False:
        mat = pd.DataFrame(
            data=adata_bulk.X.todense(),  
            columns=adata_bulk.var_names,  
            index=adata_bulk.obs.index  
        )

        tf_acts, tf_pvals = decoupler.run_ulm(mat, net, source='source', target='target', weight='weight', use_raw=False)
        # - formatize
        tf_acts = tf_acts.reset_index().melt(id_vars='index', var_name='source', value_name='activity')
        obs = adata_bulk.obs[['cell_type', 'donor_id', 'cell_count', 'age']]
        obs = obs.reset_index()
        
        tf_acts['index'] = tf_acts['index'].astype(str)
        obs['index'] = obs['index'].astype(str)
        tf_acts = tf_acts.merge(obs, on='index', how='left').drop('index', axis=1)
        assert tf_acts.shape[0]==tf_acts.shape[0]
    else:
        tf_acts = enrich_tf_local(net, adata_bulk, tf_all)

    return tf_acts
