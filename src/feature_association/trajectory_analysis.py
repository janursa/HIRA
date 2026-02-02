
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
from statsmodels.formula.api import ols, mixedlm
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


def write_traj_stats(results, dataset, cell_type):
    output_path = f"{OUTPUT_DIR}/tfa_dpt_stats_{dataset}_{cell_type}.csv"
    results.to_csv(output_path, index=False)
def retrieve_traj_stats(dataset, cell_type):
    output_path = f"{OUTPUT_DIR}/tfa_dpt_stats_{dataset}_{cell_type}.csv"
    traj_df = pd.read_csv(output_path)
    return traj_df 



def load_sc_data(dataset, cell_type, test_mode=False, min_cells_threshold=100):
    print(f"Loading data for dataset={dataset}, cell_type={cell_type}")
    print('Loading data...', flush=True)
    adata = retrieve_adata(dataset=dataset, data_type='sc', cell_type=cell_type, only_net_genes=True)
    if test_mode: # Select only 10 donors for testing
        print('TEST MODE: Selecting only 10 donors for testing...')
        adata = adata[adata.obs['donor_age'].isin(adata.obs['donor_age'].unique()[:10])].copy()
    # Check cell counts per donor_age
    print(f"\n=== Checking donor_age samples (min threshold: {min_cells_threshold} cells) ===")
    donor_age_counts = adata.obs['donor_age'].value_counts()
    print(f"Total donor_age samples: {len(donor_age_counts)}")
    # filter based on sc counts
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
    adata = adata[adata.obs['donor_age'].isin(passing_donors)].copy()
    pseudobulk_group = get_config(dataset).pseudobulk_group
    adata.obs['group_id'] = adata.obs[pseudobulk_group].astype(str).agg('_'.join, axis=1)
    return adata

def compute_dpt(adata, leiden_resolution=10):
    if False:
        print('Assiginig all cells to one group for testing DPT...')
        adata.obs['group_id'] = 'one_group'
    unique_groups = adata.obs['group_id'].unique()
    adata.obs['dpt'] = np.nan
    adata.obs['leiden'] = 'unassigned'
    
    for group in unique_groups:
        adata_sub = adata[adata.obs['group_id'] == group].copy()
        sc.pp.neighbors(adata_sub)
        sc.tl.leiden(adata_sub, resolution=leiden_resolution, key_added='leiden')
        sc.tl.paga(adata_sub, groups='leiden')
        if False:
            raise NotImplementedError("Root selection based on naive score not implemented yet.")
            cluster_naive_scores = adata_sub.obs.groupby('leiden')['naive_score'].mean()
            root_cluster = cluster_naive_scores.idxmax()
            print(f"Group {group}: number of clusters = {adata_sub.obs['leiden'].nunique()}, root cluster = {root_cluster} (mean naive score = {cluster_naive_scores[root_cluster]:.3f})")
            root_cells = adata_sub.obs['leiden'] == root_cluster
            assert root_cells.sum() > 0, f"No cells found in root cluster {root_cluster}"
            root_idx = np.where(adata_sub.obs['leiden'] == root_cluster)[0][0]
            adata_sub.uns['iroot'] = root_idx
        else: # select based on Sub_CT population 
            sub_cts = adata_sub.obs['Sub_CT'].unique()
            assert 'Tcm_Naive_CD8' in sub_cts, f"No 'naive' cells found in group {group} for root selection."
            naive_cells = adata_sub.obs['Sub_CT'] == 'Tcm_Naive_CD8'
            assert naive_cells.sum() > 0, f"No cells found in 'Tcm_Naive_CD8' Sub_CT for group {group}"
            root_idx = np.where(naive_cells)[0][0]
            adata_sub.uns['iroot'] = root_idx
        sc.tl.diffmap(adata_sub)
        sc.tl.dpt(adata_sub, n_dcs=10)
        adata.obs.loc[adata_sub.obs_names, 'dpt'] = adata_sub.obs['dpt_pseudotime']
        adata.obs.loc[adata_sub.obs_names, 'leiden'] = adata_sub.obs['leiden']
    assert not adata.obs['dpt'].isna().any(), "Some cells were not assigned pseudotime."
    return adata

def compute_tf_act(adata):
    print('Computing TF activities using ULM...', flush=True)
    cell_type = adata.obs['cell_type'].unique()
    assert len(cell_type) == 1, "Multiple cell types found in adata."
    cell_type = cell_type[0]
    net = retrieve_net_consensus(cell_type=cell_type)
    dc.mt.ulm(adata, net=net, tmin=5)
def annotate(adata):
    """Annotate cells with marker scores and assign ct_minor based on winning marker per cluster."""
    # Use more specific markers with less overlap
    markers = {
        'naive': ["CCR7", "LEF1", "SELL", "TCF7", "IL7R", "CD27"],
        'proliferating': ["MKI67", "TYMS", "PCLAF", "CLSPN", "TK1", "RRM2"],
        'central_memory': ["IL7R", "CD27", "CD28", "EOMES", "GZMK"],
        'effector_memory': ["CCL5", "GZMH", "GZMB", "PRF1", "KLRD1", "NKG7", "GNLY"],
    }
    
    adata.obs['ct_minor'] = 'unassigned'
    
    for group_id in adata.obs['group_id'].unique():
        adata_sub = adata[adata.obs['group_id'] == group_id].copy()
        
        # Calculate scores with available genes
        for subtype, genes in markers.items():
            available_genes = [g for g in genes if g in adata_sub.var_names]
            assert len(available_genes) > 0, f"No genes from marker list found in data for subtype {subtype}"
            sc.tl.score_genes(adata_sub, gene_list=available_genes, score_name=f'{subtype}_score')
        
        # Copy scores back to main adata
        adata.obs.loc[adata_sub.obs_names, [f'{subtype}_score' for subtype in markers.keys()]] = adata_sub.obs[[f'{subtype}_score' for subtype in markers.keys()]]
        
        # Reduce clustering resolution to avoid over-fragmentation
        sc.pp.neighbors(adata_sub)
        sc.tl.leiden(adata_sub, resolution=2, key_added='leiden_minor')
        
        for cluster in adata_sub.obs['leiden_minor'].unique():
            cluster_cells = adata_sub.obs['leiden_minor'] == cluster
            cluster_mask = adata_sub.obs_names[cluster_cells]
            
            # Calculate mean scores per cluster
            marker_mean_scores = {}
            for subtype in markers.keys():
                marker_mean_scores[subtype] = adata_sub.obs.loc[cluster_cells, f'{subtype}_score'].mean()
            
            # Select winning marker
            winning_marker = max(marker_mean_scores, key=marker_mean_scores.get)
            
            adata.obs.loc[cluster_mask, 'ct_minor'] = winning_marker

def dpt_marker_correlation(adata):
    """
    DPT pseudotime correlates with marker scores.
    """
    assert 'dpt' in adata.obs, "Pseudotime 'dpt' not found in adata.obs. Run compute_dpt() first."
    assert 'naive_score' in adata.obs, "Naive score not found in adata.obs. Run annotate() first."
    donor_correlations = []
    group_ids = adata.obs['group_id'].unique()
    
    for group_id in group_ids:
        adata_group = adata[adata.obs['group_id'] == group_id]
        donor_id = adata_group.obs['donor_id'].unique()
        assert len(donor_id) == 1, "Multiple donor_ids found for the same group_id."
        donor_id = donor_id[0]
        condition = adata_group.obs['condition'].unique()
        assert len(condition) == 1, "Multiple conditions found for the same group_id."
        condition = condition[0]
        
        # Correlate pseudotime with naive score (should be negative)
        corr_naive, pval_naive = spearmanr(adata_group.obs['dpt'], 
                                        adata_group.obs['naive_score'])
        
        # Correlate pseudotime with effector score (should be positive)
        corr_effector, pval_effector = spearmanr(adata_group.obs['dpt'], 
                                                adata_group.obs['effector_memory_score'])
        
        age = adata_group.obs['age'].unique()
        if len(age) > 1:
            print("Multiple ages found for the same group_id. Using the first one.")
        age = age[0]
        
        donor_correlations.append({
            'group_id': group_id,
            'donor_id': donor_id,
            'age': age,
            'n_cells': adata_group.n_obs,
            'naive_corr': corr_naive,
            'naive_pval': pval_naive,
            'effector_corr': corr_effector,
            'effector_pval': pval_effector,
            'condition': condition
        })
    
    corr_df = pd.DataFrame(donor_correlations)
    corr_df['condition'] = corr_df['condition'].astype(str)
    
    
    # Print summary
    print(f"\nCorrelation Summary:")
    print(f"Naive score - Mean corr: {corr_df['naive_corr'].mean():.3f} (Expected: negative)")
    print(f"Effector score - Mean corr: {corr_df['effector_corr'].mean():.3f} (Expected: positive)")
    print(f"Significant donors (p<0.05) for naive: {(corr_df['naive_pval'] < 0.05).sum()}/{len(corr_df)}")
    print(f"Significant donors (p<0.05) for effector: {(corr_df['effector_pval'] < 0.05).sum()}/{len(corr_df)}")
    return corr_df


def _run_association_model(adata_subset, tfs, factor_name, formula):
    """
    Helper function to run association model for a subset of data.
    Uses mixed-effects model with donor_id as random effect.
    
    """
    results = []
    
    for tf in tqdm(tfs, desc="Testing TFs", leave=False):
        # Prepare data for regression
        df = pd.DataFrame({
            'tfa': adata_subset.obsm['score_ulm'][tf].values,
            'dpt': adata_subset.obs['dpt'].values,
            'donor_id': adata_subset.obs['donor_id'].values,
            'group_id': adata_subset.obs['group_id'].values
        })
        
        # Add factor variable
        df[factor_name] = adata_subset.obs[factor_name].values
        
        # Skip if missing values or insufficient data
        if df.isna().sum().sum() > 0:
            raise ValueError("Missing values found in regression data.")
        if len(df) < 10:
            raise ValueError("Insufficient data for regression.")
        
        # Check if we have multiple donors
        n_donors = df['donor_id'].nunique()
        if n_donors < 2:
            raise ValueError(f"Need at least 2 donors for mixed-effects model, found {n_donors}")
        
        # Fit mixed-effects model with donor_id as random effect
        # Random intercept model: accounts for baseline differences between donors
        model = mixedlm(formula, data=df, groups=df['donor_id']).fit()
        
        # Extract statistics
        interaction_coef = model.params[f'dpt:{factor_name}']
        interaction_pval = model.pvalues[f'dpt:{factor_name}']
        interaction_stderr = model.bse[f'dpt:{factor_name}']
        pseudotime_coef = model.params['dpt']
        pseudotime_pval = model.pvalues['dpt']
        factor_coef = model.params[factor_name]
        factor_pval = model.pvalues[factor_name]
        
        results.append({
            'TF': tf,
            'interaction_coef': interaction_coef,
            'interaction_pval': interaction_pval,
            'interaction_stderr': interaction_stderr,
            'pseudotime_coef': pseudotime_coef,
            'pseudotime_pval': pseudotime_pval,
            f'{factor_name}_coef': factor_coef,
            f'{factor_name}_pval': factor_pval,
            'n_obs': len(df),
            'n_donors': n_donors
        })
    
    return results


def tfa_dpt_association(adata, association='continuous'):
    """
    Run association analysis to find TFs whose pseudotime association changes with a factor.
    
   
    """
    assert association in ['continuous', 'condition'], "association must be 'continuous' or 'condition'"
    
    print(f"\n=== Running Association Analysis (type: {association}) ===")
    
    # Get all TFs from ULM scores
    assert 'score_ulm' in adata.obsm, "ULM scores not found in adata.obsm['score_ulm']. Call compute_tf_act() first."
    tfs = adata.obsm['score_ulm'].columns.tolist()
    print(f"Testing {len(tfs)} TFs")
    
    all_results = []
    
    if association == 'continuous':
        # Single call for age association
        formula = 'tfa ~ dpt + age + dpt:age'
        factor_name = 'age'
        
        results = _run_association_model(adata, tfs, factor_name, formula)
        all_results.extend(results)
        
    else:  # condition
        # Get control mapping from config
        assert 'condition' in adata.obs.columns, "condition column not found in adata.obs"
        assert 'dataset' in adata.obs.columns, "dataset column not found in adata.obs"
        dataset = adata.obs['dataset'].unique()
        assert len(dataset) == 1, "Multiple datasets found in adata"
        dataset = dataset[0]
        
        config = get_config(dataset)
        control_mapping = config.control_mapping
        assert control_mapping is not None, f"control_mapping not defined for dataset {dataset}"
        
        # Build condition pairs list
        conditions_in_data = adata.obs['condition'].unique()
        condition_pairs = []
        
        if isinstance(control_mapping, str):
            # Single control for all
            control_cond = control_mapping
            for cond in conditions_in_data:
                if cond != control_cond:
                    condition_pairs.append((cond, control_cond))
        else:
            # Dict mapping - each condition has its control
            for cond in conditions_in_data:
                control = control_mapping[cond]
                if cond != control:
                    condition_pairs.append((cond, control))
        
        print(f"Found {len(condition_pairs)} condition-control pairs:")
        for treat, ctrl in condition_pairs:
            print(f"  - {treat} vs {ctrl}")
        
        # Loop through each condition-control pair
        for treatment_cond, control_cond in condition_pairs:
            print(f"\nAnalyzing: {treatment_cond} vs {control_cond}")
            
            # Filter data to only include this treatment and its control
            mask = adata.obs['condition'].isin([treatment_cond, control_cond])
            adata_subset = adata[mask].copy()
            
            # Create binary variable for this specific comparison
            adata_subset.obs['treatment'] = (adata_subset.obs['condition'] == treatment_cond).astype(int)
            
            print(f"  Samples: {adata_subset.n_obs} cells ({(adata_subset.obs['treatment']==1).sum()} treatment, {(adata_subset.obs['treatment']==0).sum()} control)")
            
            # Call the association function
            formula = 'tfa ~ dpt + treatment + dpt:treatment'
            factor_name = 'treatment'
            
            results = _run_association_model(adata_subset, tfs, factor_name, formula)
            
            # Add treatment and control names to results
            for result in results:
                result['treatment'] = treatment_cond
                result['control'] = control_cond
                result['comparison'] = f"{treatment_cond} vs {control_cond}"
            
            all_results.extend(results)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    if len(results_df) == 0:
        print("Warning: No results generated")
        return results_df
    
    # Apply FDR correction on interaction p-values
    results_df['interaction_qval'] = multipletests(results_df['interaction_pval'], method='fdr_bh')[1]
    
    # Sort by interaction p-value
    results_df = results_df.sort_values('interaction_pval')
    
    return results_df

def compute_tfa_dpt_association(adata):
    """
    Calculate Spearman correlation between TF activity and DPT per group.
    Stores results in adata.varm['tfa_dpt_corr'] and adata.varm['tfa_dpt_pval'].
    """
    from scipy.stats import spearmanr
    
    # Get TFs
    tfs = adata.obsm['score_ulm'].columns.tolist()
    
    # Get unique groups
    groups = adata.obs['group_id'].unique()
    
    # Initialize matrices: rows=TFs, columns=groups
    corr_matrix = pd.DataFrame(index=tfs, columns=groups, dtype=float)
    pval_matrix = pd.DataFrame(index=tfs, columns=groups, dtype=float)
    
    print(f"Computing TF-DPT correlations for {len(tfs)} TFs across {len(groups)} groups...")
    
    for group in groups:
        # Subset to this group
        adata_group = adata[adata.obs['group_id'] == group]
        
        if adata_group.n_obs < 10:
            print(f"Skipping {group}: insufficient cells ({adata_group.n_obs})")
            continue
        
        dpt = adata_group.obs['dpt'].values
        
        for tf in tfs:
            tf_activity = adata_group.obsm['score_ulm'][tf].values
            
            # Calculate Spearman correlation
            corr, pval = spearmanr(dpt, tf_activity)
            
            corr_matrix.loc[tf, group] = corr
            pval_matrix.loc[tf, group] = pval
    
    # Store in adata.varm (variable/feature metadata matrices)
    # adata.uns['tfa_dpt_corr'] = corr_matrix
    # adata.uns['tfa_dpt_pval'] = pval_matrix.values
    
    # print(f"Stored correlations in adata.uns['tfa_dpt_corr'] (shape: {corr_matrix.shape})")
    # print(f"Stored p-values in adata.uns['tfa_dpt_pval']")

    # we create a new AnnData to hold the correlation matrix
    corr_adata = ad.AnnData(X=corr_matrix.values.T, var=pd.DataFrame(index=corr_matrix.index), obs=pd.DataFrame(index=corr_matrix.columns))
    
    # Get unique group metadata (one row per group_id)
    group_metadata = adata.obs.drop_duplicates(subset='group_id').set_index('group_id')
    
    # Merge .obs to include group metadata
    corr_adata.obs = corr_adata.obs.merge(group_metadata, left_index=True, right_index=True, how='left')
        
    return corr_adata