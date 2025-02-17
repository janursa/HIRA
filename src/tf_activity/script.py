import anndata as ad
import pandas as pd
import sys
import numpy as np

import argparse
meta = {
    'helper_dir': '../src/tf_activity/'
}
sys.path.append(meta['helper_dir'])
from helper import enrich_tfs



# def main(par):
#     adata_bulk = ad.read_h5ad(par['adata_bulk'], backed='r')
#     tf_all = np.loadtxt(par['tf_all'], dtype=str)

#     # - one net for each cell type
#     net = par['net']
#     net = pd.read_csv(net)

#     # - get the expression for the cell type
#     mask_sample = adata_bulk.obs['cell_type']==par['cell_type'] 
#     adata_sample = adata_bulk[mask_sample, :].to_memory()   
#     adata_sample = adata_sample[adata_sample.obs['cell_count']>par['n_cells_t']]

#     tf_acts = enrich_tfs(adata_sample, net, tf_all)

#     return tf_acts

# if __name__=="__main__":
#     argparser = argparse.ArgumentParser()
#     argparser.add_argument('--adata_bulk', type=str, required=False, help='Path to the bulk data')
#     argparser.add_argument('--net', type=str,required=False, help='Path to the net')
#     argparser.add_argument('--cell_type', type=str, required=False,help='Subset to this cell type')

#     args = argparser.parse_args()
#     par_input = vars(args)

    

#     par = {
#         'n_cells_t': 10,
#         'tf_all': "../../task_grn_inference/resources/grn_benchmark/prior/tf_all.csv"
#     }
#     for key in par_input.keys():
#         if par_input[key] is not None:
#             par[key] = par_input[key]
#     main(par)