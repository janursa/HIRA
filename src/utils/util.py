
import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import sys
import scanpy as sc
import matplotlib.pyplot as plt

from task_grn_inference.src.utils.util import sum_by, read_gmt


def plot_umap(adata, color='', palette=None, ax=None, X_label='X_umap', on_data=False,
              bbox_to_anchor=None, legend=True, legend_title='', **kwrds):
    latent = adata.obsm[X_label]
    var_unique_sorted = sorted(adata.obs[color].unique())
    legend_handles = []
    
    for i_group, group in enumerate(var_unique_sorted):
        mask = adata.obs[color] == group
        sub_data = latent[mask]
        if palette is None:
            c = None 
        else:
            c = palette[group]
        # Plot scatter points
        scatter = ax.scatter(sub_data[:, 0], sub_data[:, 1], label=group, c=c, **kwrds)
        if palette is None:
            solid_color = scatter.get_facecolor()[0]
        else:
            solid_color = c
        # plot legend
        legend_handles.append(plt.Line2D([0], [0], linestyle='none', marker='o', markersize=8, color=solid_color))
        if on_data:
            mean_x = np.mean(sub_data[:, 0])
            mean_y = np.mean(sub_data[:, 1])
            ax.text(mean_x, mean_y, group, fontsize=9, ha='center', va='top', color='black', weight='bold')
    ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])

    if legend and not on_data:
        legend = ax.legend(handles=legend_handles, labels=var_unique_sorted, loc=(1.4,.3), 
                           bbox_to_anchor=bbox_to_anchor, frameon=False, title=legend_title, 
                           title_fontproperties={'weight': 'bold', 'size': 9})
        legend.get_title().set_ha('left')
        legend._legend_box.align = "left" 
    
def get_canonical_pathways():
    geneset_file = '/home/jnourisa/projs/ongoing/ciim/input/prior/h.all.v2024.1.Hs.symbols.gmt'
    genesets_all = read_gmt(geneset_file) 
    genesets_all = {key: gs['genes'] for key, gs in genesets_all.items()}

    # Create a list of gene-to-pathway mappings (one-to-one mapping)
    gene_to_pathway_list = [
        (gene, pathway)
        for pathway, genes in genesets_all.items()
        for gene in genes
    ]

    # Convert the list to a DataFrame
    df_pathway = pd.DataFrame(gene_to_pathway_list, columns=["gene", "pathway"])
    df_pathway = df_pathway.set_index("gene")
    df_pathway['pathway'] = df_pathway['pathway'].str.replace('HALLMARK_','')
    df_pathway['pathway'] = df_pathway['pathway'].str.replace('_',' ')
    df_pathway['pathway'] = df_pathway['pathway'].str.title()

    return df_pathway

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
def basic_qc(adata, min_genes_per_cell = 200, max_genes_per_cell = 5000, min_cells_per_gene = 10):
    mt = adata.var_names.str.startswith('MT-')
    print('shape before ', adata.shape)
    # 1. stats
    total_counts = adata.X.sum(axis=1)
    n_genes_by_counts = (adata.X > 0).sum(axis=1)
    # mt_frac = adata[:, mt].X.sum(axis=1) / total_counts
    
    low_gene_filter = (n_genes_by_counts < min_genes_per_cell)
    high_gene_filter = (n_genes_by_counts > max_genes_per_cell)
    # mt_filter = mt_frac > max_mt_frac

    # 2. Filter cells
    # print(f'Number of cells removed: below min gene {low_gene_filter.sum()}, exceed max gene {high_gene_filter.sum()}')
    mask_cells=  (~low_gene_filter)& \
                 (~high_gene_filter)
                #  (~mt_filter)
    # 3. Filter genes
    n_cells = (adata.X!=0).sum(axis=0)
    mask_genes = n_cells>min_cells_per_gene
    adata_f = adata[mask_cells, mask_genes]
    print('shape after ', adata_f.shape)
    return adata_f

def bulkify_adata(adata):
    adata.obs['sum_by'] = '_' + adata.obs['cell_type'].astype(str) + '_' + adata.obs['donor_id'].astype(str) + '_' + adata.obs['age'].astype(str) 
    adata.obs['sum_by'] = adata.obs['sum_by'].astype('category')
    adata_bulk = sum_by(adata, 'sum_by', unique_mapping=False)
    cell_count_df = adata.obs.groupby('sum_by').size().reset_index(name='cell_count')
    adata_bulk.obs = adata_bulk.obs.merge(cell_count_df, on='sum_by')

    
    sc.pp.normalize_total(adata_bulk)
    sc.pp.log1p(adata_bulk)

    return adata_bulk
