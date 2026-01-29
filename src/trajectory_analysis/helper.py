
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import scanpy as sc
import warnings
import anndata as ad
import os
from scipy.sparse import issparse
from scipy.stats import spearmanr, pearsonr
import decoupler as dc
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import tqdm
import logging
import argparse

warnings.filterwarnings('ignore')
logging.getLogger('anndata').setLevel(logging.ERROR)
pd.set_option('display.max_columns', 100)
plt.rcParams["font.family"] = "Arial"

# Local imports
from hiara import retrieve_feature_data, retrieve_sig_stats
from hiara import OUTPUT_DIR, PRIOR_DIR, PLOTS_DIR, CLOCKS_DIR, CELL_TYPES, surrogate_names, colors_blind, palette_cell_types, palette_datasets, palette_datasets_pretty, palette_genders, AGING_COHORTS
from hiara import retrieve_adata, retrieve_net, retrieve_net_consensus
from hiara import get_config
from grn_benchmark.src.helper import load_env
from task_grn_inference import normalize_func, bulkify_func


def write_traj_stats(results, dataset, cell_type, mode='dpt'):
    output_path = f"{OUTPUT_DIR}/tf_age_{mode}_interaction_{dataset}_{cell_type}.csv"
    results.to_csv(output_path, index=False)
def retrieve_traj_stats(dataset, cell_type, mode='dpt'):
    output_path = f"{OUTPUT_DIR}/tf_age_{mode}_interaction_{dataset}_{cell_type}.csv"
    traj_df = pd.read_csv(output_path)
    return traj_df 

def cluster(adata, dataset, resolution=1):
    """Cluster cells within each group."""
    pseudobulk_group = get_config(dataset).pseudobulk_group
    # print(f"Pseudobulk groups: {pseudobulk_group}")
    
    if 'leiden' in adata.obs:
        del adata.obs['leiden']
    
    adata.obs['group_id'] = adata.obs[pseudobulk_group].astype(str).agg('_'.join, axis=1)
    unique_groups = adata.obs['group_id'].unique()
    # print(f"Number of unique groups: {len(unique_groups)}")
    
    adata.obs['leiden'] = 'unassigned'
    
    for group_id in unique_groups:
        group_mask = adata.obs['group_id'] == group_id
        adata_group = adata[group_mask].copy()
        adata_group = normalize_func(adata_group)
        
        assert adata_group.n_obs > 100, f"Not enough cells in group {group_id} to perform clustering."
        
        sc.pp.neighbors(adata_group)
        sc.tl.leiden(adata_group, resolution=resolution)
        cluster_labels = [f"{group_id}_cluster_{c}" for c in adata_group.obs['leiden']]
        adata.obs.loc[group_mask, 'leiden'] = cluster_labels
    
    assert (adata.obs['leiden'] == 'unassigned').sum() == 0, "Some cells were not assigned a leiden cluster."
    adata.obs['leiden'] = pd.Categorical(adata.obs['leiden'])
    
    adata_cluster = bulkify_func(adata, covariates=['leiden'], cell_count_t=10)
    adata_cluster = normalize_func(adata_cluster)
    adata_cluster.obs['leiden'] = pd.Categorical(adata_cluster.obs['leiden'])
    
    return adata_cluster


def prepare_single_cell(adata, dataset):
    """Prepare single-cell data for trajectory analysis."""
    pseudobulk_group = get_config(dataset).pseudobulk_group
    
    # Create group_id for organizing cells
    adata.obs['group_id'] = adata.obs[pseudobulk_group].astype(str).agg('_'.join, axis=1)
    
    # Normalize if needed
    adata = normalize_func(adata)
    
    return adata


def annotate(adata):
    """Annotate cells with marker scores."""
    markers = {
        'naive': ["CD8B", "S100B", "CCR7", "RGS10", "NOSIP", "LINC02446", "LEF1", "CRTAM", "CD8A", "OXNAD1"],
        'proliferating': ["MKI67", "CD8B", "TYMS", "TRAC", "PCLAF", "CD3D", "CLSPN", "CD3G", "TK1", "RRM2"],
        'central_memory': ["CD8B", "ANXA1", "CD8A", "KRT1", "LINC02446", "YBX3", "IL7R", "TRAC", "NELL2", "LDHB"],
        'effector_memory': ["CCL5", "GZMH", "CD8A", "TRAC", "KLRD1", "NKG7", "GZMK", "CST7", "CD8B", "TRGC2"],
    }
    for subtype, genes in markers.items():
        # Filter to only genes present in the data
        available_genes = [g for g in genes if g in adata.var_names]
        assert len(available_genes) > 0, f"No genes from marker list found in data for subtype {subtype}"
        sc.tl.score_genes(adata, gene_list=available_genes, score_name=f'{subtype}_score')

    # print(f"Available genes per subtype:")
    # for subtype, genes in markers.items():
    #     available = [g for g in genes if g in adata.var_names]
    #     print(f"  {subtype}: {len(available)}/{len(genes)} genes")


def dpt(adata):
    """DPT without root - uses first diffusion component as pseudotime."""
    unique_groups = adata.obs['group_id'].unique()
    adata.obs['dpt'] = np.nan
    
    for group in unique_groups:
        adata_sub = adata[adata.obs['group_id'] == group].copy()
        sc.pp.neighbors(adata_sub)
        sc.tl.diffmap(adata_sub)
        sc.tl.dpt(adata_sub)
        adata.obs.loc[adata_sub.obs_names, 'dpt'] = adata_sub.obs['dpt_pseudotime']
    
    assert not adata.obs['dpt'].isna().any(), "Some cells were not assigned DPT pseudotime."


def plot_association_results(adata_combined, traj_stats, dataset, cell_type, top_n=10, age_cutoff=50):
    """
    Visualize top TFs showing altered association between TF activity and pseudotime with aging.
    Shows scatter plots with LOWESS trend lines for young vs old donors.
    """
    print(f"\n=== Plotting Association Results ===")
    
    # Get top TFs by interaction p-value
    top_tfs = traj_stats.nsmallest(top_n, 'interaction_pval')['TF'].tolist()
    print(f"Plotting top {top_n} TFs: {', '.join(top_tfs)}")
    
    # Categorize donors by age
    adata_combined.obs['age_group'] = adata_combined.obs['age'].apply(
        lambda x: 'Young' if x < age_cutoff else 'Old'
    )
    
    # Create plots
    n_rows = (top_n + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
    axes = axes.flatten()
    
    for idx, tf in enumerate(top_tfs):
        ax = axes[idx]
        
        # Get TF activity
        tf_activity = adata_combined.obsm['score_ulm'][tf].values
        pseudotime = adata_combined.obs['paga_pseudotime'].values
        age_group = adata_combined.obs['age_group'].values
        donor_ids = adata_combined.obs['donor_age'].values
        age = adata_combined.obs['age'].values
        
        # Plot scatter points with very low alpha
        for group, color in [('Young', 'blue'), ('Old', 'red')]:
            mask = age_group == group
            ax.scatter(pseudotime[mask], tf_activity[mask], 
                      alpha=0.01, s=1, c=color, label=f'{group} cells')
        
        # Fit LOWESS for each donor and plot
        unique_donors = np.unique(donor_ids)
        for donor_id in unique_donors:
            donor_mask = donor_ids == donor_id
            donor_age_group = age_group[donor_mask][0]
            color = 'blue' if donor_age_group == 'Young' else 'red'
            
            # Get donor data
            donor_pseudotime = pseudotime[donor_mask]
            donor_tf_activity = tf_activity[donor_mask]
            
            # Sort for plotting
            sort_idx = np.argsort(donor_pseudotime)
            donor_pseudotime_sorted = donor_pseudotime[sort_idx]
            donor_tf_activity_sorted = donor_tf_activity[sort_idx]
            
            # Fit LOWESS
            if len(donor_pseudotime) > 10:
                lowess_result = lowess(donor_tf_activity_sorted, donor_pseudotime_sorted, 
                                      frac=0.3, return_sorted=True)
                ax.plot(lowess_result[:, 0], lowess_result[:, 1], 
                       color=color, alpha=0.5, linewidth=1.5)
        
        # Get statistics for this TF
        tf_stats = traj_stats[traj_stats['TF'] == tf].iloc[0]
        interaction_coef = tf_stats['interaction_coef']
        interaction_pval = tf_stats['interaction_pval']
        interaction_qval = tf_stats['interaction_qval']
        
        ax.set_xlabel('DPT Pseudotime', fontsize=10)
        ax.set_ylabel(f'{tf} Activity', fontsize=10)
        ax.set_title(f'{tf}\nβ={interaction_coef:.3f}, p={interaction_pval:.2e}, q={interaction_qval:.2e}', 
                    fontsize=10)
        ax.legend(fontsize=8)
    
    # Remove empty subplots
    for idx in range(top_n, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plot_output = f"{PLOTS_DIR}/top_tf_association_{dataset}_{cell_type}.pdf"
    plt.savefig(plot_output, dpi=300, bbox_inches='tight')
    print(f"Association plot saved to: {plot_output}")
    plt.close()
    
    return top_tfs
    """
    Run association analysis: TF_activity ~ paga_pseudotime + age + paga_pseudotime:age
    Test interaction term to find TFs whose pseudotime association changes with age.
    """
    print("\n=== Running Association Analysis ===")
    
    # Get all TFs from ULM scores
    tfs = adata_combined.obsm['score_ulm'].columns.tolist()
    print(f"Testing {len(tfs)} TFs")
    
    results = []
    
    for tf in tfs:
        # Prepare data for regression
        df = pd.DataFrame({
            'tf_activity': adata_combined.obsm['score_ulm'][tf].values,
            'paga_pseudotime': adata_combined.obs['paga_pseudotime'].values, # might need normalization?!
            'age': adata_combined.obs['age'].values,
            'donor_id': adata_combined.obs['donor_id'].values
        })
        
        # Remove any missing values
        assert df.isna().sum().sum() == 0, "Missing values found in regression data."
        
        if len(df) < 10:
            print(f"Skipping {tf}: insufficient data (n={len(df)})")
            continue
        
        # Fit linear model with interaction
        model = ols('tf_activity ~ paga_pseudotime + age + paga_pseudotime:age', data=df).fit()
        
        # Extract interaction term statistics
        interaction_coef = model.params['paga_pseudotime:age']
        interaction_pval = model.pvalues['paga_pseudotime:age']
        interaction_stderr = model.bse['paga_pseudotime:age']
        
        # Also get main effects
        pseudotime_coef = model.params['paga_pseudotime']
        pseudotime_pval = model.pvalues['paga_pseudotime']
        age_coef = model.params['age']
        age_pval = model.pvalues['age']
        
        results.append({
            'TF': tf,
            'interaction_coef': interaction_coef,
            'interaction_pval': interaction_pval,
            'interaction_stderr': interaction_stderr,
            'pseudotime_coef': pseudotime_coef,
            'pseudotime_pval': pseudotime_pval,
            'age_coef': age_coef,
            'age_pval': age_pval,
            'r_squared': model.rsquared,
            'n_obs': len(df)
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Apply FDR correction on interaction p-values
    results_df['interaction_qval'] = multipletests(results_df['interaction_pval'], method='fdr_bh')[1]
    
    # Sort by interaction p-value
    results_df = results_df.sort_values('interaction_pval')
    
    # Save to CSV
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    print(f"Significant TFs (FDR < 0.05): {(results_df['interaction_qval'] < 0.05).sum()}")
    print(f"Significant TFs (p < 0.05): {(results_df['interaction_pval'] < 0.05).sum()}")
    
    # Display top 10 results
    print("\nTop 10 TFs by interaction p-value:")
    print(results_df[['TF', 'interaction_coef', 'interaction_pval', 'interaction_qval']].head(10))
    
    return results_df

def paga(adata):
    """Perform PAGA trajectory analysis on single-cell data with root selection based on naive markers."""
    unique_groups = adata.obs['group_id'].unique()
    
    # Annotate with marker scores for root selection
    annotate(adata)
    
    adata.obs['paga_pseudotime'] = np.nan
    adata.obs['leiden'] = 'unassigned'
    
    for group in unique_groups:
        adata_sub = adata[adata.obs['group_id'] == group].copy()
        
        # Cluster for PAGA
        sc.pp.neighbors(adata_sub)
        sc.tl.leiden(adata_sub, resolution=1, key_added='leiden')
        
        # Run PAGA
        sc.tl.paga(adata_sub, groups='leiden')
        
        # Select root based on naive markers
        cluster_naive_scores = adata_sub.obs.groupby('leiden')['naive_score'].mean()
        root_cluster = cluster_naive_scores.idxmax()
        print(f"Group {group}: Selected root cluster {root_cluster} (naive_score={cluster_naive_scores[root_cluster]:.3f})")
        
        root_cells = adata_sub.obs['leiden'] == root_cluster
        assert root_cells.sum() > 0, f"No cells found in root cluster {root_cluster}"
        
        root_idx = np.where(adata_sub.obs['leiden'] == root_cluster)[0][0]
        adata_sub.uns['iroot'] = root_idx
        
        # Compute diffusion map and pseudotime
        sc.tl.diffmap(adata_sub)
        sc.tl.dpt(adata_sub, n_dcs=10)
        
        # Store results back in original adata
        adata.obs.loc[adata_sub.obs_names, 'paga_pseudotime'] = adata_sub.obs['dpt_pseudotime']
        adata.obs.loc[adata_sub.obs_names, 'leiden'] = adata_sub.obs['leiden']
    
    assert not adata.obs['paga_pseudotime'].isna().any(), "Some cells were not assigned pseudotime."
    
    return adata




def association_analysis_dpt(adata_combined):
    """
    Run association analysis: TF_activity ~ paga_pseudotime + age + paga_pseudotime:age
    Test interaction term to find TFs whose pseudotime association changes with age.
    """
    print("\n=== Running Association Analysis ===")
    
    # Get all TFs from ULM scores
    tfs = adata_combined.obsm['score_ulm'].columns.tolist()
    print(f"Testing {len(tfs)} TFs")
    
    results = []
    
    for tf in tqdm(tfs, desc="Testing TFs"):
        # Prepare data for regression
        df = pd.DataFrame({
            'tf_activity': adata_combined.obsm['score_ulm'][tf].values,
            'paga_pseudotime': adata_combined.obs['paga_pseudotime'].values,
            'age': adata_combined.obs['age'].values,
            'donor_id': adata_combined.obs['donor_id'].values
        })
        
        # Remove any missing values
        assert df.isna().sum().sum() == 0, "Missing values found in regression data."
        
        if len(df) < 10:
            print(f"Skipping {tf}: insufficient data (n={len(df)})")
            continue
        
        # Fit linear model with interaction
        model = ols('tf_activity ~ paga_pseudotime + age + paga_pseudotime:age', data=df).fit()
        
        # Extract interaction term statistics
        interaction_coef = model.params['paga_pseudotime:age']
        interaction_pval = model.pvalues['paga_pseudotime:age']
        interaction_stderr = model.bse['paga_pseudotime:age']
        
        # Also get main effects
        pseudotime_coef = model.params['paga_pseudotime']
        pseudotime_pval = model.pvalues['paga_pseudotime']
        age_coef = model.params['age']
        age_pval = model.pvalues['age']
        
        results.append({
            'TF': tf,
            'interaction_coef': interaction_coef,
            'interaction_pval': interaction_pval,
            'interaction_stderr': interaction_stderr,
            'pseudotime_coef': pseudotime_coef,
            'pseudotime_pval': pseudotime_pval,
            'age_coef': age_coef,
            'age_pval': age_pval,
            'r_squared': model.rsquared,
            'n_obs': len(df)
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Apply FDR correction on interaction p-values
    results_df['interaction_qval'] = multipletests(results_df['interaction_pval'], method='fdr_bh')[1]
    
    # Sort by interaction p-value
    results_df = results_df.sort_values('interaction_pval')
    
    print(f"\nResults saved to: {output_path}")
    print(f"Significant TFs (FDR < 0.05): {(results_df['interaction_qval'] < 0.05).sum()}")
    print(f"Significant TFs (p < 0.05): {(results_df['interaction_pval'] < 0.05).sum()}")
    
    # Display top 10 results
    print("\nTop 10 TFs by interaction p-value:")
    print(results_df[['TF', 'interaction_coef', 'interaction_pval', 'interaction_qval']].head(10))
    
    return results_df


def process_donors(dataset, cell_type, min_cells_threshold=100, test_mode=False):
    """Process all donors and return combined AnnData object."""
    print(f"Loading data for dataset={dataset}, cell_type={cell_type}")
    net = retrieve_net_consensus(cell_type=cell_type)
    print('Loading data...', flush=True)
    adata_o = retrieve_adata(dataset=dataset, data_type='sc', cell_type=cell_type)
    
    print(f"Total cells: {adata_o.n_obs}")
    print(f"Unique donor_age samples: {adata_o.obs['donor_age'].nunique()}")
    
    # Check cell counts per donor_age
    print(f"\n=== Checking donor_age samples (min threshold: {min_cells_threshold} cells) ===")
    donor_age_counts = adata_o.obs['donor_age'].value_counts()
    print(f"Total donor_age samples: {len(donor_age_counts)}")
    
    if test_mode:
        # Select only 10 donors for testing
        test_donors = donor_age_counts.head(10).index
        donor_age_counts = donor_age_counts[donor_age_counts.index.isin(test_donors)]
        print("TEST MODE: Using only 10 donors")
    
    # Filter donors that pass the threshold
    passing_donors = donor_age_counts[donor_age_counts >= min_cells_threshold].index.tolist()
    failing_donors = donor_age_counts[donor_age_counts < min_cells_threshold].index.tolist()
    
    print(f"Passing samples: {len(passing_donors)} (>= {min_cells_threshold} cells)")
    print(f"Failing samples: {len(failing_donors)} (< {min_cells_threshold} cells)")

    if len(failing_donors) > 0:
        print(f"\nSkipping {len(failing_donors)} donor_age samples with insufficient cells:")
        for donor_id in failing_donors[:5]:  # Show first 5
            print(f"  - {donor_id}: {donor_age_counts[donor_id]} cells")
        if len(failing_donors) > 5:
            print(f"  ... and {len(failing_donors) - 5} more")
    
    # Filter data to only passing donors
    adata_o = adata_o[adata_o.obs['donor_age'].isin(passing_donors)].copy()
    print(f"\nFiltered data: {adata_o.n_obs} cells from {len(passing_donors)} donor_age samples")
    
    # Process each passing donor separately
    print(f"\n=== Processing {len(passing_donors)} passing donor_age samples ===")
    
    adata_list = []
    
    for donor_id in tqdm(passing_donors, desc="Processing donor_age samples"):
        adata_donor = adata_o[adata_o.obs['donor_age'] == donor_id].copy()
        
        age = adata_donor.obs['age'].iloc[0]
        
        # Prepare single-cell data (no pseudobulking)
        adata_sc = prepare_single_cell(adata_donor, dataset=dataset)
        
        # Run trajectory analysis on single cells
        paga(adata_sc)
        
        # Compute TF activities on single cells
        dc.mt.ulm(adata_sc, net=net, tmin=5)
        
        # Add age information
        adata_sc.obs['age'] = age
        adata_sc.obs['donor_age'] = donor_id
        
        adata_list.append(adata_sc)
    
    # Check if we have enough data for association analysis
    if len(adata_list) < 10:
        print(f"\nWARNING: Only {len(adata_list)} donor_age samples available.")
        print("Results may be unreliable with < 10 samples.")
    
    # Combine all donors
    print(f"\n=== Combining data from {len(adata_list)} donor_age samples ===")
    adata_combined = ad.concat(adata_list, join='outer')
    print(f"Combined data: {adata_combined.n_obs} observations")
    print(f"Age range: {adata_combined.obs['age'].min():.1f} - {adata_combined.obs['age'].max():.1f}")
    
    return adata_combined


def association_analysis_marker(adata_bulk, output_path):
    """
    Run association analysis: TF_activity ~ marker_score + age + marker_score:age
    Tests interaction term for both naive and effector markers to find TFs whose 
    association with differentiation markers changes with age.
    
    This is similar to DPT analysis but uses biological markers instead of pseudotime.
    """
    print("\n=== Running Marker-based Association Analysis ===")
    
    # Get all TFs from ULM scores
    tfs = adata_bulk.obsm['score_ulm'].columns.tolist()
    print(f"Testing {len(tfs)} TFs")
    
    results_naive = []
    results_effector = []
    
    for tf in tqdm(tfs, desc="Testing TFs"):
        # Test with naive marker
        df_naive = pd.DataFrame({
            'tf_activity': adata_bulk.obsm['score_ulm'][tf].values,
            'marker_score': adata_bulk.obs['naive_score'].values,
            'age': adata_bulk.obs['age'].values,
            'donor_age': adata_bulk.obs['donor_age'].values
        })
        
        if not df_naive.isna().any().any() and len(df_naive) >= 10:
            model = ols('tf_activity ~ marker_score + age + marker_score:age', data=df_naive).fit()
            
            results_naive.append({
                'TF': tf,
                'marker': 'naive',
                'interaction_coef': model.params['marker_score:age'],
                'interaction_pval': model.pvalues['marker_score:age'],
                'interaction_stderr': model.bse['marker_score:age'],
                'marker_coef': model.params['marker_score'],
                'marker_pval': model.pvalues['marker_score'],
                'age_coef': model.params['age'],
                'age_pval': model.pvalues['age'],
                'r_squared': model.rsquared,
                'n_obs': len(df_naive)
            })
        
        # # Test with effector marker
        # df_effector = pd.DataFrame({
        #     'tf_activity': adata_bulk.obsm['score_ulm'][tf].values,
        #     'marker_score': adata_bulk.obs['effector_memory_score'].values,
        #     'age': adata_bulk.obs['age'].values,
        #     'donor_age': adata_bulk.obs['donor_age'].values
        # })
        
        # if not df_effector.isna().any().any() and len(df_effector) >= 10:
        #     model = ols('tf_activity ~ marker_score + age + marker_score:age', data=df_effector).fit()
            
        #     results_effector.append({
        #         'TF': tf,
        #         'marker': 'effector_memory',
        #         'interaction_coef': model.params['marker_score:age'],
        #         'interaction_pval': model.pvalues['marker_score:age'],
        #         'interaction_stderr': model.bse['marker_score:age'],
        #         'marker_coef': model.params['marker_score'],
        #         'marker_pval': model.pvalues['marker_score'],
        #         'age_coef': model.params['age'],
        #         'age_pval': model.pvalues['age'],
        #         'r_squared': model.rsquared,
        #         'n_obs': len(df_effector)
        #     })
    
    # Combine results
    results_df = pd.DataFrame(results_naive + results_effector)
    
    # Apply FDR correction on interaction p-values
    results_df['interaction_qval'] = multipletests(results_df['interaction_pval'], method='fdr_bh')[1]
    
    # Sort by interaction p-value
    results_df = results_df.sort_values('interaction_pval')
    
    # Save to CSV
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    print(f"Significant TFs (FDR < 0.05): {(results_df['interaction_qval'] < 0.05).sum()}")
    print(f"Significant TFs (p < 0.05): {(results_df['interaction_pval'] < 0.05).sum()}")
    
    # Display top 10 results for each marker
    print("\nTop 10 TFs by interaction p-value (Naive marker):")
    print(results_df[results_df['marker'] == 'naive'][['TF', 'interaction_coef', 'interaction_pval', 'interaction_qval']].head(10))
    
    # print("\nTop 10 TFs by interaction p-value (Effector marker):")
    # print(results_df[results_df['marker'] == 'effector_memory'][['TF', 'interaction_coef', 'interaction_pval', 'interaction_qval']].head(10))
    
    return results_df
