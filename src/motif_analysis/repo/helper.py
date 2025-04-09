
import warnings
import sys 
import os 
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import argparse
from pybedtools import BedTool
from joblib import Parallel, delayed
import tqdm

from task_grn_inference.src.utils.util import read_gene_annotation


def bedify_peaks(peak_df, col_name='peak'):
    other_cols = [col for col in peak_df.columns if col != col_name]
    peak_df[['chromosome', 'start', 'end']] = peak_df[col_name].str.split(r':|-', expand=True)
    peak_df = peak_df[['chromosome', 'start', 'end']+other_cols]
    peak_df.drop_duplicates(inplace=True)
    peaks_bed = BedTool.from_dataframe(peak_df)
    return peaks_bed

def run_consensus_peak(par):
    """
    Find consensus peaks for each cell type.
    """
    print("Running consensus peak...")
    adata = ad.read_h5ad(par['atac'])
    threshold = 0.05  # Fraction of cells where a peak is present
    consensus_list = []
    for cell_type in adata.obs['cell_type'].unique():
        # Select cells of the given cell type
        cell_mask = adata.obs['cell_type'] == cell_type
        sub_adata = adata[cell_mask, :]

        # Compute fraction of cells where each peak is present
        peak_presence = np.array(sub_adata.X.sum(axis=0)).flatten() / sub_adata.shape[0]
        
        # Identify consensus peaks
        consensus_peaks = adata.var.index[peak_presence >= threshold]
        
        # Store results as (peak, cell_type) pairs
        consensus_list.extend([(peak, cell_type) for peak in consensus_peaks])
    consensus_peak = pd.DataFrame(consensus_list, columns=['peak', 'cell_type'])
    consensus_peak.to_csv(par['consensus_peak'], index=False)



def run_region2promotors(par):
    """
    Connect regions of motif database to promotor regions.
    """
    print("Running region2promotors...")

    # - read consensus peaks
    consensus_peak = pd.read_csv(par['consensus_peak'])
    consensus_peak = consensus_peak[['peak']].drop_duplicates()
    peaks_bed = bedify_peaks(consensus_peak)
    # - read promotor data
    promoters_df = pd.read_csv(par['promoter_df'])
    promoter_bed = bedify_peaks(promoters_df[['promoter', 'gene_name']], col_name='promoter')
    # - read motif score database #TODO: this can be generated from the peak data
    motif_scores = pd.read_feather(par['motif_scores_db'])
    motif_scores = motif_scores.set_index('motifs')
    # print(f'Number of motifs: {motif_scores.shape[0]}, Number of regions: {motif_scores.shape[1]}')

    # - only take the regions (columns): we do not use the other information at this step
    regions_df = motif_scores.columns.to_series().to_frame(name='region')
    regions_bed = bedify_peaks(regions_df[['region']], col_name='region')

    # Use bedtools window to find all promoters within given distance
    regions2promoters = regions_bed.window(promoter_bed, w=par['enhancer_distance']).to_dataframe(
        names=['chromosome', 'start', 'end', 'promoter_chr', 'promoter_start', 'promoter_end', 'gene_name']
    )
    regions2promoters['region'] = regions2promoters.apply(lambda x: f"{x['chromosome']}:{x['start']}-{x['end']}", axis=1)
    regions2promoters['promoter'] = regions2promoters.apply(lambda x: f"{x['promoter_chr']}:{x['promoter_start']}-{x['promoter_end']}", axis=1)

    regions2promoters = regions2promoters[['region', 'promoter', 'gene_name']]
    regions2promoters = regions2promoters.merge(promoters_df[['promoter', 'promoter_open']], on='promoter', how='left')
    regions2promoters.to_csv(par['regions2promoters'], index=False)


def identify_promoter_region(df, upstream=1000, downstream=200):
    """
    Identify promoter region with given upstream and downstream distance.
    """
    df["start"] = df.apply(lambda x: x["TSS"] - downstream if x["strand"] == "+" else x["TSS"] - upstream, axis=1)
    df["end"] = df.apply(lambda x: x["TSS"] + upstream if x["strand"] == "+" else x["TSS"] + downstream, axis=1)
    promoters_df = df[["chromosome", "start", "end", "gene_name"]]
    return promoters_df

def run_promotor_analysis(par):
    """
    Define promotor regions and find open promotors based on peak data.
    """
    print('Running promotor analysis...')
    # - read consensus peaks
    consensus_peak = pd.read_csv(par['consensus_peak'])
    consensus_peak = consensus_peak[['peak']].drop_duplicates()
    peaks_bed = bedify_peaks(consensus_peak)
    # - read annotation file
    annotation = read_gene_annotation(par['annotation_file'])  # ["chromosome", "start", "end", "strand", "gene_name"]
    print(f"Number of genes: {annotation['gene_name'].nunique()}")
    promoters_df = identify_promoter_region(annotation)
    promoter_bed = BedTool.from_dataframe(promoters_df)
    promoters_df['promoter'] = promoters_df.apply(lambda x: f"{x['chromosome']}:{x['start']}-{x['end']}", axis=1)
    promoters_df = promoters_df[['promoter', 'gene_name']]
    # - find open promoters. #TODO: this can be improved by finding enriched promoters in peaks
    open_promoters = promoter_bed.intersect(peaks_bed, wa=True, wb=True).to_dataframe(
                names=['chromosome', 'start', 'end', 'gene_name', 'peak_chr', 'peak_start', 'peak_end'])
    open_promoters['promoter'] = open_promoters.apply(lambda x: f"{x['chromosome']}:{x['start']}-{x['end']}", axis=1)
    open_promoters = open_promoters[['promoter', 'gene_name']]

    promoters_df['promoter_open'] = False
    promoters_df.loc[promoters_df['promoter'].isin(open_promoters['promoter']), ['promoter_open']] = True

    promoters_df.to_csv(par['promoter_df'], index=False)

    promoters_df.to_csv(par['promoter_df'], index=False)

class TF2gene:
    def __init__(self, motif_scores, regions2promoters, motifs_info, n_jobs=20):
        self.tf2gene = self.run_tfs2genes(motif_scores, regions2promoters, motifs_info, n_jobs)

    @staticmethod
    def process_motif(motif, motif_scores_m, regions2promoters):
        # Filter based on scores
        scores = motif_scores_m.values.flatten()
        mask = scores > np.quantile(scores, 0.95)  # TODO: use a more stringent threshold
        motif_scores_m = motif_scores_m.loc[:, mask]

        # Reshape and reformat
        motif_scores_m = motif_scores_m.T
        motif_scores_m.columns = ['score']
        motif_scores_m['region'] = motif_scores_m.index
        motif_scores_m.reset_index(drop=True, inplace=True)
        motif_scores_m['motif'] = motif

        # Merge with promoter regions
        return motif_scores_m.merge(regions2promoters, on='region', how='inner')
    @staticmethod
    def run_tfs2genes(motif_scores, regions2promoters, motifs_info, n_jobs=20):
        unique_motifs = motif_scores.index.unique()
        
        # Parallel processing
        motif2genes_store = Parallel(n_jobs=n_jobs)(
            delayed(TF2gene.process_motif)(motif, motif_scores.loc[[motif]], regions2promoters) for motif in tqdm.tqdm(unique_motifs, desc='Motifs to genes')
        )

        # Combine results
        motif2genes = pd.concat(motif2genes_store, ignore_index=True)[['motif', 'region', 'score', 'promoter', 'gene_name', 'promoter_open']]
        tf2gene = motif2genes.merge(motifs_info, on='motif', how='left')[['tf', 'motif', 'region', 'score', 'gene_name', 'promoter_open']]
        
        return tf2gene
    
def run_tf2gene(par):
    """
    Run TF to gene mapping.
    """
    print('Running tf2gene...')
    # - read regions2promotors
    regions2promoters = pd.read_csv(par['regions2promoters'])
    
    # - read motif info (tf to motif mapping)
    motifs_info = pd.read_csv(par['motifs_info'], sep='\t')
    motifs_info = motifs_info[['#motif_id', 'gene_name']].rename(columns={'#motif_id': 'motif', 'gene_name': 'tf'})


    # - read motif scores
    motif_scores = pd.read_feather(par['motif_scores_db'])
    motif_scores = motif_scores.set_index('motifs')

    # - subset #TODO: needs to go
    tf = 'TGIF1'
    motifs = motifs_info[motifs_info['tf'] == tf]['motif'] 
    motif_scores = motif_scores[motif_scores.index.isin(motifs)]

    # - run tf2gene
    regions2promoters = regions2promoters[regions2promoters['promoter_open'] == True] #TODO: this is a temporary filter

    tf2gene_obj = TF2gene(motif_scores, regions2promoters, motifs_info, par['n_jobs'])
    tf2genes = tf2gene_obj.tf2gene
    print(f"Number of links in tf2gene: {tf2genes.shape[0]}")
    print(f"Number of open regions: {tf2genes['region'].nunique()} for {tf2genes['motif'].nunique()} motifs")
    tf2genes.to_csv(par['tf2gene'], index=False)