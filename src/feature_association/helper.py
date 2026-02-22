import subprocess
import numpy as np
from scipy.stats import linregress, spearmanr
import pandas as pd
import os
import scipy
from concurrent.futures import ThreadPoolExecutor

import scanpy as sc
import anndata as ad
from statsmodels.stats.multitest import multipletests
from hiara.src.config import get_config_fa, FEATURES_DIR, MAJOR_CTS, get_config, surrogate_names, DISCOVERY_COHORTS, HIARA_DIR
from tqdm import tqdm
from hiara.src.config import OUTPUT_DIR, FEATURES_DIR, CORR_THRESHOLD, TF_MIN_TARGET, FEATURE_TYPES, SUB_CT_LABEL, MAJOR_CT_LABEL
from scipy.sparse import issparse
from hiara.src.utils.util import retrieve_adata, retrieve_net_consensus, retrieve_net
import warnings
from scipy.sparse import issparse
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_rel
warnings.filterwarnings("ignore")


def write_features_stats(stats, analysis_name, multi_cohort=True, dataset=None, suffix=''):

    os.makedirs(f'{FEATURES_DIR}/{analysis_name}/stats', exist_ok=True)
    if multi_cohort:
        file_name = f'{FEATURES_DIR}/{analysis_name}/stats/stats_multi_cohort{suffix}.csv'
    else:
        file_name = f'{FEATURES_DIR}/{analysis_name}/stats/stats_{dataset}{suffix}.csv'
    print(f"✓ Condition stats saved: {file_name}")
    stats.to_csv(file_name, index=False)
        

def retrieve_stats(analysis_name, cell_type=None, dataset=None, multi_cohort=None, features_dir=None, suffix=''):
    # determine whether this is multi-cohort
    if dataset is None:
        multi_cohort = True
    else:
        multi_cohort = False
    # load stats
    if features_dir is None:
        features_dir = FEATURES_DIR
    if multi_cohort:
        stats = pd.read_csv(
            f"{features_dir}/{analysis_name}/stats/stats_multi_cohort{suffix}.csv"
        )
    else:
        stats = pd.read_csv(
            f"{features_dir}/{analysis_name}/stats/stats_{dataset}{suffix}.csv"
        )
    # optional cell-type filtering
    if cell_type is not None:
        if cell_type not in stats["cell_type"].unique():
            raise ValueError(
                f'Given cell type "{cell_type}" not in {stats["cell_type"].unique()}'
            )
        stats = stats[stats["cell_type"] == cell_type]


    # significance logic
    if multi_cohort:
        stats["comparison"] = 'aging'
        p_val_col = "meta_p_adj"

        mask_trend = stats["trend"] != "Inconsistent"

        # genes where all datasets have |slope| > threshold
        def has_consistent_slope(group):
            return (group["slope"].abs() > CORR_THRESHOLD).all()

        valid_groups = (
            stats.groupby(["gene", "cell_type"])
            .filter(has_consistent_slope)
        )
        mask_slope = stats.index.isin(valid_groups.index)

    else:
        config = get_config(dataset)
        stats["comparison"] = stats["comparison"].map(
            lambda name: config.name_mapping.get(name, name)
        )
        p_val_col = "p_value_adj"
        
        stats['p_value_adj'] = stats['p_value_adj'].apply(lambda x: 1E-50 if x < 1E-50 else x)
        mask_trend = pd.Series(True, index=stats.index)
        mask_slope = stats["slope"].abs() > CORR_THRESHOLD

        if dataset == 'perez_sle':
            condition = 'disease'
        elif dataset in ['parsebioscience', 'op', 'CXCL9']:
            condition = 'treatment'
        elif dataset == 'soundlife':
            condition = 'aging'
        else:
            raise ValueError(f'Unknown dataset {dataset} for trend determination')
        if condition == 'treatment':
            stats['trend'] = [
                f'Increase after {condition}' if x > 0 else f'Decrease after {condition}' 
                for x in stats['slope']
            ]
        else:
            stats['trend'] = [
                f'Increase in {condition}' if x > 0 else f'Decrease in {condition}' 
                for x in stats['slope']
            ]

    mask_pvalue = stats[p_val_col] < 0.05

    stats["is_significant"] = mask_pvalue & mask_trend & mask_slope

    return stats
def retrieve_sig_stats_agg(**kwargs):
    stats = retrieve_sig_stats(**kwargs)
    stats_agg = stats.groupby(['cell_type', 'gene']).agg({
        'slope': 'mean',
        'meta_p_adj': 'first',
        'comparison': 'first'
    }).reset_index()
    return stats_agg
def retrieve_sig_stats(**kwargs):
    stats = retrieve_stats(**kwargs)
    stats_sig = stats[stats['is_significant']]
    return stats_sig

def retrieve_feature_data(
                          dataset, 
                          cell_type, 
                          analysis_name, 
                          condition=None, 
                          suffix=''
                          ):
    # Validate analysis_name exists
    try:
        _ = get_config_fa(analysis_name)
    except ValueError as e:
        raise ValueError(f'Analysis name {analysis_name} not found in CONFIG_FA') from e

    if analysis_name in ['ge_major_b', 'ge_sub_b']:
        granularity = get_config_fa(analysis_name)['granularity']
        data_type = get_config_fa(analysis_name)['data_type']
        adata = retrieve_adata(dataset=dataset, 
                               data_type=data_type, 
                               cell_type=cell_type, 
                               only_net_genes=True, 
                               condition=condition,
                               granularity=granularity)
        
    else:
        file_path = f'{FEATURES_DIR}/{analysis_name}/{dataset}_{cell_type}{suffix}.h5ad'
        if os.path.exists(file_path) == False:
            raise ValueError(f'File {file_path} does not exist')
        adata = ad.read_h5ad(file_path)
    # Filter by condition
    if condition is not None and 'condition' in adata.obs.columns:
        if condition not in adata.obs['condition'].unique():
            raise ValueError(f'Error in retrieving feature data: given condition "{condition}" not in {adata.obs["condition"].unique()}')
        
        
        adata = adata[adata.obs['condition'] == condition].copy()
        print(adata.obs['condition'].value_counts())
    
    # Filter by cell type using the specified granularity column
    # Skip filtering if cell_type is 'all' (used for cross-cell-type analyses like ccc_sub_b)
    if cell_type is not None and cell_type != 'all':
        granularity = get_config_fa(analysis_name)['granularity']
        if granularity not in adata.obs.columns:
            raise ValueError(f'Error in retrieving feature data: granularity column "{granularity}" not in adata.obs.columns')
        if cell_type not in adata.obs[granularity].unique():
            raise ValueError(f'Error in retrieving feature data: given cell type "{cell_type}" not in {adata.obs[granularity].unique()}')
        adata = adata[adata.obs[granularity] == cell_type]

    return adata

def write_feature_data(adata, dataset, cell_type, analysis_name, suffix=''):
    output_dir = f'{FEATURES_DIR}/{analysis_name}'
    os.makedirs(output_dir, exist_ok=True)
    adata.write_h5ad(f'{output_dir}/{dataset}_{cell_type}{suffix}.h5ad')

def bin_feature_values(adata):
    # - bin 
    expr = adata.to_df()
    expr = expr.merge(adata.obs[['age']], left_index=True, right_index=True, how='left').set_index('age')
    expr.sort_index(inplace=True)
    expr['age_bin'] = (expr.index.astype(int) // 5) * 5
    expr_mean = expr.groupby('age_bin').mean().T
    # Normalize expression
    min_vals = expr_mean.min(axis=1)
    max_vals = expr_mean.max(axis=1)
    expr_mean = (expr_mean.sub(min_vals, axis=0)).div(max_vals - min_vals, axis=0)
    return expr_mean

def determine_stats_condition(adata, ctr_group='normal', condition_col='condition', test_type='unpaired', conditions=None, config=None):
    
    if conditions is None:
        conditions = adata.obs[condition_col].unique()
    dataset = adata.obs['dataset'].unique()[0]
    name_mapping = {} if config is None else config.name_mapping if hasattr(config, 'name_mapping') else {}
    stats_all = []
    def stats_condition_vs_ctr(adata, condition):  
        mask_ctr = adata.obs[condition_col] == ctr_group
        assert mask_ctr.sum() > 0, f'No control found in {ctr_group} {dataset}'
        mask_condition = adata.obs[condition_col] == condition
        if mask_condition.sum() == 0:
            raise ValueError(f'No condition found in {condition} {dataset} condition_col: {condition_col}, available conditions: {adata.obs[condition_col].unique()}')
        
        control_group = adata.X[mask_ctr.values, :]
        case_group = adata.X[mask_condition.values, :]
        if (np.sum(mask_condition) < 3) or (np.sum(mask_ctr) < 3):
            print('Not enough samples for', condition, ' vs ', ctr_group)
            return None
        
        def process_gene(i_gene):
            i, gene = i_gene
            values_case = case_group[:, i]
            values_control = control_group[:, i]

            if np.sum(values_case) == 0 and np.sum(values_control) == 0:
                return None

            if issparse(values_case):
                values_case = values_case.todense().A.flatten()
            if issparse(values_control):
                values_control = values_control.todense().A.flatten()
            
            if test_type == 'unpaired':
                from scipy.stats import ttest_ind
                stat, pval = mannwhitneyu(values_case, values_control, alternative="two-sided")
                # stat, pval = ttest_ind(values_case, values_control, equal_var=False)
                coef = np.median(values_case) - np.median(values_control)
                
                return {
                    'gene': gene,
                    "p_value": pval,
                    "slope": coef,
                    'ctrl': ctr_group,
                    'condition': condition
                }
            elif test_type == 'paired':
                # Create dataframes with donor_id to ensure proper pairing
                obs_ctr = adata.obs.loc[mask_ctr, :].copy().reset_index(drop=True)
                obs_ctr['feature_values'] = values_control
                
                obs_case = adata.obs.loc[mask_condition, :].copy().reset_index(drop=True)
                obs_case['feature_values'] = values_case
                
                # Merge on donor_id to ensure samples are properly paired
                df_paired = obs_ctr[['donor_id', 'feature_values']].merge(
                    obs_case[['donor_id', 'feature_values']], 
                    on='donor_id', 
                    suffixes=('_ctr', '_case')
                )
                
                # Check if we have valid pairs
                if len(df_paired) < 3:
                    return None
                
                stat, pval = ttest_rel(df_paired['feature_values_case'], 
                                      df_paired['feature_values_ctr'])
                coef = np.median(df_paired['feature_values_case'] - df_paired['feature_values_ctr'])
                
                return {
                    'gene': gene,
                    "p_value": pval,
                    "slope": coef,
                    'ctrl': ctr_group,
                    'condition': condition
                }
            elif test_type == 'mixed-effect':
                from hiara.src.utils.util import test_mixed_effects
                
                obs_ctr = adata.obs.loc[mask_ctr, :].copy()
                obs_ctr['feature_values'] = values_control
                obs_ctr['condition'] = ctr_group
                obs_case = adata.obs.loc[mask_condition, :].copy()
                obs_case['condition'] = condition
                obs_case['feature_values'] = values_case
                
                df = pd.concat([obs_ctr, obs_case], ignore_index=True)
                
                pval, coef = test_mixed_effects(df, ctr_group, condition, 
                                                target_variable='feature_values', config=config)
                if np.isnan(pval):
                    return None
                # if gene=='TCF7':
                #     print({
                #     'gene': gene,
                #     "p_value": pval,
                #     "slope":  coef ,
                #     'ctrl': ctr_group,
                #     'condition': name_mapping.get(condition, condition)
                # })
                return {
                    'gene': gene,
                    "p_value": pval,
                    "slope":  coef ,
                    'ctrl': ctr_group,
                    'condition': name_mapping.get(condition, condition)
                }
               
   
            else:
                raise ValueError('Unknown test type')
        
        # Parallelize gene processing
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(process_gene, enumerate(adata.var_names)))
        
        # Filter out None results and flatten if needed (interaction models return lists)
        flattened_results = []
        for r in results:
            if r is None:
                continue
            if isinstance(r, list):
                # Interaction model - list of coefficient dicts
                flattened_results.extend(r)
            else:
                # Regular model - single dict
                flattened_results.append(r)
        
        results = pd.DataFrame(flattened_results)
        if len(results) == 0:
            raise ValueError(f'No valid results for condition {condition} vs {ctr_group} in dataset {dataset}')
        return results
    
    # Determine age stratification based on dataset and config
    if 'sle' in dataset:
        # SLE: Run for each age subset
        age_masks = {
            'Both age groups': adata.obs.index.notnull(),  # All samples
            'Younger than 50': adata.obs['age'] < 50,
            'Older than 50': adata.obs['age'] >= 50
        }
    else:
        # Default: no age stratification
        age_masks = {
            'All age groups': adata.obs.index.notnull()
        }
    
    for age_subset, mask in age_masks.items():
        adata_sub = adata[mask, :].copy()
        assert adata_sub.shape[0]!=0, f'shouldnt be empty'
        for condition in conditions:
            if condition == ctr_group:
                continue
            stats_df = stats_condition_vs_ctr(adata_sub, condition)


            if stats_df is None:
                print(f'Skipping {condition} vs {ctr_group} in dataset {dataset} because the stats df is empty (likely due to insufficient samples)')
                continue

            # For interaction models, apply FDR per coefficient type
            if 'coefficient' in stats_df.columns:
                # Interaction model - apply FDR per coefficient
                for coef_type in stats_df['coefficient'].unique():
                    mask = stats_df['coefficient'] == coef_type
                    stats_df.loc[mask, 'p_value_adj'] = multipletests(
                        stats_df.loc[mask, 'p_value'], 
                        method='fdr_bh'
                    )[1]
            else:
                # Regular model - apply FDR globally
                stats_df['p_value_adj'] = multipletests(stats_df["p_value"], method="fdr_bh")[1]
            
            assert np.any(np.isnan(stats_df['p_value_adj']) == False), f'NaN p-values in {stats_df}'
            
            stats_df['age_group'] = age_subset
            stats_all.append(stats_df)
    
    # Combine all stats
    if len(stats_all) == 0:
        print(f'No valid statistics for dataset {dataset} - all conditions had insufficient samples')
        return None
    
    stats_df = pd.concat(stats_all, ignore_index=True)
    stats_df['dataset'] = dataset

    return stats_df

def run_meta_analysis(stats_all, meta_association_type='max', min_degree=2, temp_dir='../output/tf_activity/'):
    os.makedirs(temp_dir, exist_ok=True)
    # ---------- prepare
    assert stats_all.shape[0]> 0, 'No stats for meta analysis'
    print('Meta analysis...')
    stats_all_c = stats_all.copy()
    stats_all_c.rename(columns={'p_value': 'pvalue'}, inplace=True)
    
    # -------- actual run
    if min_degree is not None:
        stats_all_c = stats_all_c.groupby(['gene', 'cell_type']).filter(lambda group: group['dataset'].nunique() >= min_degree)
    
    cell_types = stats_all_c['cell_type'].unique()
    df_meta_store = []
    for cell_type in cell_types:
        df = stats_all_c[stats_all_c['cell_type'] == cell_type]
        df['pvalue'] = df['pvalue'] + 1E-20 # to avoid 0 p value
        
        file_path = f'{temp_dir}/stats_{cell_type}.csv'
        df.to_csv(file_path, index=False)
        out_path = f'{temp_dir}/stats_{cell_type}_meta.csv'

        Rscript_file = f'{HIARA_DIR}/src/feature_association//meta_analysis/script.R'
        # Run the R script with the provided file paths
        try:
            subprocess.run(
                ["Rscript", Rscript_file, file_path, out_path, meta_association_type],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            print(f"Error while running R script: {Rscript_file}")
            print("STDOUT:", e.stdout.decode())  # Standard Output
            print("STDERR:", e.stderr.decode())  # Error Output from R
            raise
        df_meta = pd.read_csv(out_path)

        assert df_meta.isna().sum().sum() == 0, f'NaN values found in meta analysis results for {cell_type}'
        df_meta['cell_type'] = cell_type
        print(cell_type, df_meta.shape)
        df_meta_store.append(df_meta)
    if df_meta_store:
        df_meta_all = pd.concat(df_meta_store)
    else:
        df_meta_all = pd.DataFrame() 
    
    if df_meta_all.shape[0]==0:
        print(f"No meta analysis results for {stats_all['cell_type'].unique()}")
        return None
    stats_all = stats_all.merge(df_meta_all, on=['gene', 'cell_type'], how='left')
    return stats_all

def wrapper_meta_analysis(analysis_name, stats_features, par):
    feature_col = 'gene'
    min_degree = par['META_MIN_COHORT']
    def run_func(min_degree, meta_association_type):
        stats_store = []
        for cell_type in stats_features['cell_type'].unique():
            stats = stats_features[(stats_features['cell_type'] == cell_type) & (stats_features['condition']=='healthy')]
            if len(stats) == 0:
                print('No stats for', cell_type, ' skipping it')
                continue
            if stats.groupby(feature_col).size().max()<min_degree:
                print('Not enough mutual TFs for ', cell_type, ' skipping it')
                continue
            # - keep only min_degree info that is consistent
            assert isinstance(stats['p_value'].iloc[0], (float, int)), f'p_value should be numeric, found {stats["p_value"].iloc[0]} like values in {cell_type}'
            nan_sim = stats['p_value'].isna().sum()
            if nan_sim>0:
                raise ValueError(f'NaN p-values found in stats in {cell_type}: {nan_sim} NaNs')
            meta_stats = run_meta_analysis(stats, temp_dir=par['temp_dir'], meta_association_type=meta_association_type, min_degree=min_degree)
            pval_col = 'meta_p_adj'
            meta_stats = compute_trend(analysis_name, meta_stats, pval_col=pval_col, slope_col='slope', col=feature_col)
            stats_store.append(meta_stats)
        if len(stats_store) > 0:
            stats_discovery = pd.concat(stats_store)
            stats_discovery['condition'] = 'healthy'
            return stats_discovery
        else:
            return pd.DataFrame()
    
    meta_association_type='fisher'
    print(f'Running meta analysis for all datasets, min degree  {min_degree}, meta_association_type {meta_association_type}')
    stats = run_func(min_degree, meta_association_type)

    #- save
    return stats

def wrapper_association_with_age_condition(analysis_name, 
                                par, 
                                association_type, 
                                test_type=None, 
                                condition='healthy', 
                                features=None,
                                config=None):
    """
    Wrapper function to compute association with age and condition.
    
    Now uses configuration objects to eliminate dataset-specific if-else chains.
    
    Parameters
    ----------
    par : dict
        Parameters containing datasets, feature_type, type, cell_types, etc.
    condition : str
        Condition filter for loading data
    config : ConditionConfig, optional
        Configuration object (will auto-load if None)
    """
    
    datasets = par['datasets']
    cell_types = par['cell_types']
    promotor_only = par.get('promotor_only', False)
    suffix = '_promotor' if promotor_only else ''

    analys_cfg = get_config_fa(analysis_name)
    granularity = analys_cfg['granularity']
    
    stats_store = []
    for cell_type in tqdm(cell_types, desc='cell types'):
        for dataset in datasets:        
            adata = retrieve_feature_data(
                dataset=dataset, 
                cell_type=cell_type, 
                analysis_name=analysis_name, 
                condition=condition,
                suffix=suffix,
            )
            # - sanity check
            cell_types_in_data = adata.obs[granularity].unique()
            assert len(cell_types_in_data) == 1 and cell_types_in_data[0] == cell_type, f'Cell type mismatch in {dataset}, {cell_type}'
            adata = adata[:, adata.var_names.isin(features)] if features is not None else adata
            print(adata.shape, dataset, cell_type)

            # Filter by cell type
            if adata.shape[0] < 3:
                raise ValueError(f'Not enough samples for {cell_type} in {dataset}, only {adata.shape[0]} samples')
            
            if issparse(adata.X):
                adata.X = adata.X.toarray()
            
            # Determine statistics based on configuration
            if association_type == 'continous':
                print('Aging analysis for', cell_type, 'in', dataset)
                stats = association_with_age(adata, association_type='spearman')
                stats['condition'] = condition
                stats['comparison'] = 'aging'
            elif association_type == 'grouped':
                # Condition analysis using config
                print('Condition analysis for', cell_type, 'in', dataset)
                stats = associate_with_condition(
                    adata, 
                    config, 
                    test_type=test_type
                )
                
            else:
                raise ValueError(f'Unknown analysis type: {association_type}')
            if stats is None or len(stats) == 0:
                raise ValueError(f'No stats calculated for {cell_type} in {dataset}, something went wrong')

            stats['dataset'] = dataset
            stats['cell_type'] = cell_type
            stats_store.append(stats)
    
    assert len(stats_store) > 0, 'No stats calculated, something went wrong'
    
    if len(stats_store) == 1:
        stats_all = stats_store[0]
    else:
        stats_all = pd.concat(stats_store)
    
    print(stats_all['cell_type'].unique())


    return stats_all


def associate_with_condition(adata, config, test_type=None):
    """
    Compute condition statistics using configuration object.
    
    """
    # Auto-detect condition column for datasets with variants
    condition_col = config.condition_column
    if test_type is None:
        test_type = config.test_type
        assert test_type is not None, 'test type should be given either in config or passed to the function'

    # Get treatment groups
    if config.treatment_groups == 'all':
        treatment_groups = adata.obs[condition_col].unique()
    else:
        treatment_groups = config.treatment_groups

    control_mapping = config.control_mapping
    assert control_mapping is not None, 'control_mapping should be defined.'
    if isinstance(control_mapping, str):
        # Single control for all treatments
        config.control_mapping = {treatment: control_mapping for treatment in treatment_groups}
    
    stats_list = []
    for treatment in treatment_groups:
        
        control = config.control_mapping[treatment]
        if treatment == control:
            continue
        stats = determine_stats_condition(
            adata,
            ctr_group=control,
            condition_col=condition_col,
            test_type=test_type,
            conditions=[treatment],
            config=config
        )
        
        if stats is None:
            print(f'Skipping comparison {treatment} vs {control} - no valid statistics')
            continue
            
        stats['comparison'] = f'{treatment} vs {control}'
        stats['comparison'] = stats['comparison'].map(lambda name: config.name_mapping.get(name, name))
        
        stats_list.append(stats)
    
    if len(stats_list) == 0:
        raise ValueError(f'No valid comparisons found - all had insufficient samples')
        
    stats = pd.concat(stats_list)

    return stats


def wrapper_sub_celltype_markers(analysis_name, par):
    """
    Identify TF activity markers for each sub cell type within a major cell type.
    Performs DE analysis between each sub cell type and all other sub cell types
    within the same major cell type, across multiple datasets.
    
    """
    from hiara.src.config import mapping_major_2_minor
    
    datasets = par['datasets']
    major_cell_types = par['cell_types']
    promotor_only = par.get('promotor_only', False)
    suffix = '_promotor' if promotor_only else ''
    
    analys_cfg = get_config_fa(analysis_name)
    granularity = analys_cfg['granularity']
    
    print(f'Major cell types: {major_cell_types}')
    
    stats_store = []
    
    for major_ct in tqdm(major_cell_types, desc='Major cell types'):
        sub_cell_types = mapping_major_2_minor[major_ct]
        print(f"\n{major_ct}: analyzing sub types {sub_cell_types}")
        
        for dataset in datasets:
            print(f"  Dataset: {dataset}")
            
            # Load TF activity data for all sub cell types of this major type
            adata_list = []
            for ct in sub_cell_types:
                adata = retrieve_feature_data(
                        dataset=dataset,
                        cell_type=ct,
                        analysis_name='tfa_sub_b',  # Use existing TF activity data
                        condition='healthy',
                        suffix=suffix
                        )
                adata_list.append(adata)
            adata = ad.concat(adata_list)
            adata = adata[adata.obs[granularity].isin(sub_cell_types)].copy()     

            if issparse(adata.X):
                adata.X = adata.X.toarray()
            
            # For each sub cell type, perform DE analysis vs all others
            for target_sub_ct in sub_cell_types:
                    
                # Create binary labels: target vs rest
                mask_target = adata.obs[granularity] == target_sub_ct
                mask_others = ~mask_target
                
                n_target = mask_target.sum()
                n_others = mask_others.sum()
                
                
                # Perform DE analysis for each TF
                from scipy.stats import mannwhitneyu
                
                results = []
                for gene in adata.var_names:
                    values_target = adata[mask_target, gene].X
                    values_others = adata[mask_others, gene].X
                    # Mann-Whitney U test
                    stat, pval = mannwhitneyu(values_target.flatten(), values_others.flatten(), alternative="two-sided")
                    
                    # Effect size (median difference)
                    coef = np.median(values_target) - np.median(values_others)
                    
                    results.append({
                        'gene': gene,
                        'p_value': pval,
                        'slope': coef,
                        'sub_cell_type': target_sub_ct,
                        'major_cell_type': major_ct,
                        'n_target': n_target,
                        'n_others': n_others
                    })
                
                stats_df = pd.DataFrame(results)
                # print(stats_df)
                # aaa
                
                # FDR correction per sub cell type
                
                stats_df['dataset'] = dataset
                stats_df['cell_type'] = target_sub_ct  # Use sub cell type as main cell_type for compatibility
                stats_df['comparison'] = f'{target_sub_ct} vs others'
                
                stats_store.append(stats_df)
                print(f"    {target_sub_ct}: {len(stats_df)} TFs analyzed")
    
    if len(stats_store) == 0:
        raise ValueError("No marker statistics calculated. Check data availability.")
    
    stats_all = pd.concat(stats_store, ignore_index=True)
    stats_all['condition'] = 'healthy'  # Add condition column for consistency
    
    return stats_all

def wrapper_tf_activity(analysis_name, par):
    print('Loading data...')
    cell_types = par['cell_types']
    datasets = par['datasets']
    condition = par.get('condition', None)
    promotor_only = par['promotor_only']
    
    print('Calculating TF activity...')
    print(f'  - Promotor-based only: {promotor_only}')

    config = get_config_fa(analysis_name)
    data_type = config['data_type']
    granularity = config['granularity']
    for dataset in datasets:
        adata = retrieve_adata(dataset=dataset, data_type=data_type, condition=condition)

        for cell_type in tqdm(cell_types, desc='cell types'):
            print(dataset, data_type)
            adata_t = adata[adata.obs[granularity] == cell_type].copy()
            if len(adata_t) == 0:
                print(f'No samples for {cell_type} in {dataset}, skipping TF activity calculation')
                continue
            if par['use_consensus_net']:
                net = retrieve_net_consensus(cell_type=cell_type, promotor_only=promotor_only)
            else:
                net = retrieve_net(dataset=dataset, cell_type=cell_type, promotor_only=promotor_only)
            if adata_t.shape[0] < 10:
                continue
            tf_acts = calculate_tf_activity(adata_t, net)
            tf_acts.obs['dataset'] = dataset
            tf_acts.uns['dataset'] = dataset
            tf_acts = tf_acts[tf_acts.obs['age'].isna()==False] # there is a bug in the code that causes age to be NaN
            write_feature_data(tf_acts, dataset=dataset, cell_type=cell_type, analysis_name=analysis_name, suffix='_promotor' if promotor_only else '')

def wrapper_genesets_scores(par):
    from hiara.src.utils.util import get_genesets
    # --------- load data
    cell_types = par['cell_types']
    datasets = par['datasets']
    analysis_name = par['analysis_name']
    
    config = get_config_fa(analysis_name)
    data_type = config['data_type']
    granularity = config['granularity']

    # --------- load data
    print('Loading data...')
    print(f'  - Granularity: {granularity}')
    adata_dict = {dataset: retrieve_adata(dataset, data_type, granularity=granularity) for dataset in datasets}

    pathways = get_genesets()
    
    print('Calculating gene scores...')
    for cell_type in tqdm(cell_types, desc='cell types'):
        # ----------- calculate tf activity for all datasets
        net = retrieve_net_consensus(cell_type=cell_type) #TOGO 
        net_genes = net['target'].unique()
        for dataset in datasets:
            adata = adata_dict[dataset][adata_dict[dataset].obs[granularity]==cell_type]
            
            # Ensure genes are in adata.var_names
            gene_scores = {}
            n_matching_genes = {}
            for pathway, genes in pathways.items():
                if par['gene_coverage'] == 'all_genes':
                    pass
                elif par['gene_coverage'] == 'target_genes':
                    genes = [gene for gene in genes if gene in net_genes]  # TOGO: Filter genes based on the net
                else:
                    raise ValueError(f'Unknown gene coverage {par["gene_coverage"]}, should be one of all_genes, target_genes')
                genes_present = list(set(genes).intersection(adata.var_names))
                if len(genes_present) == 0:
                    continue
                if False:
                    score = np.array(adata[:, genes_present].X.mean(axis=1)).flatten()
                else:
                    sc.tl.score_genes(adata, gene_list=genes_present, score_name=pathway, use_raw=False)
                    score = adata.obs[pathway].values
                
                gene_scores[pathway] = score
                n_matching_genes[pathway] = len(genes_present)

            if not gene_scores:
                print(f"No matching genes found for {cell_type} in {dataset}. Skipping.")
                continue

            # Create a new AnnData object to store scores
            X_df = pd.DataFrame(gene_scores, index=adata.obs_names)

            # Make sure index (cell names) is properly set as obs
            obs = adata.obs.loc[X_df.index].copy()

            # Each column in X_df is a gene score for a pathway
            var = pd.DataFrame(index=X_df.columns)
            var['n_matching_genes'] = pd.Series(n_matching_genes)

            adata_scores = sc.AnnData(X=X_df.values, obs=obs, var=var)
            write_feature_data(adata_scores, dataset, cell_type, analysis_name=analysis_name, suffix='')
def df_2_adata(df, obs):
    df_adata = ad.AnnData(X=df.values, var=pd.DataFrame(index=df.columns), obs=pd.DataFrame(index=df.index))
    
    # Use donor_age as the primary grouping key
    group_metadata = obs.drop_duplicates(subset='donor_age').set_index('donor_age')
    cell_counts = obs.groupby('donor_age').size().rename('cell_count')
    
    df_adata.obs = df_adata.obs.merge(group_metadata, left_index=True, right_index=True, how='left')
    df_adata.obs = df_adata.obs.merge(cell_counts, left_index=True, right_index=True, how='left')
    return df_adata

def wrapper_ct_freq(par):
    from hiara.src.feature_association.trajectory_analysis import load_sc_data
    # --------- load data
    datasets = par['datasets']
    analysis_name = par['analysis_name']
    # test_mode = par.get('test_mode', False)
    # min_cells_threshold=100
    data_type = get_config_fa(analysis_name)['data_type']
    granularity = get_config_fa(analysis_name)['granularity']
    cell_types = par['cell_types']
    
    # Handle condition parameter - convert to list if needed
    conditions = par['condition']
    if isinstance(conditions, str):
        conditions = [conditions]
    
    for dataset in datasets:
        for cell_type in tqdm(cell_types, desc='cell types'):
            # Process each condition separately and concatenate
            freq_adata_list = []
            
            for condition in conditions:
                obs = retrieve_adata(dataset=dataset, 
                                    data_type=data_type, 
                                    only_obs=True, 
                                    condition=condition, 
                                    cell_type=cell_type
                                    )
                
                # Skip if no data for this condition
                if len(obs) == 0:
                    print(f'No data for {dataset}, {cell_type}, {condition}')
                    continue
                
                freq_df = obs.groupby(['donor_age', granularity]).size().unstack(fill_value=0)
                freq_df = freq_df.div(freq_df.sum(axis=1), axis=0)
                freq_adata = df_2_adata(freq_df, obs)
                freq_adata.obs[granularity] = cell_type
                freq_adata.obs['condition'] = condition
                freq_adata_list.append(freq_adata)
            
            # Concatenate all conditions
            if len(freq_adata_list) == 0:
                print(f'No frequency data calculated for {dataset}, {cell_type}')
                raise ValueError(f'No frequency data calculated for {dataset}, {cell_type}')
            elif len(freq_adata_list) == 1:
                freq_adata_combined = freq_adata_list[0]
            else:
                freq_adata_combined = ad.concat(freq_adata_list, join='outer', fill_value=0)
            
            write_feature_data(freq_adata_combined, dataset, cell_type, analysis_name=analysis_name)
        

def wrapper_tfa_peg(par):
    from hiara.src.feature_association.trajectory_analysis import load_sc_data, compute_tfa_peg_association
    # --------- load data
    cell_types = par['cell_types']
    datasets = par['datasets']
    analysis_name = par['analysis_name']
    test_mode = False
    leiden_resolution=10
    min_cells_threshold=100

    print(f'  - Analysis: {analysis_name}')
    
    for cell_type in tqdm(cell_types, desc='cell types'):
        for dataset in datasets:
            adata = load_sc_data(dataset=dataset, cell_type=cell_type, test_mode=test_mode, min_cells_threshold=min_cells_threshold)
            # annotate(adata)
            # compute_dpt(adata, leiden_resolution=leiden_resolution) # save adata for visualization and downstream analysis
            calculate_tf_activity(adata, net=retrieve_net_consensus(cell_type=cell_type))
            corr_matrix = compute_tfa_peg_association(adata, target=SUB_CT_LABEL)
            corr_adata = df_2_adata(corr_matrix.T, adata)
            write_feature_data(corr_adata, dataset, cell_type, analysis_name=analysis_name, suffix='')

def wrapper_aging_hallmarks(par):
    from hiara.src.config import PRIOR_DIR
    # --------- load data
    cell_types = par['cell_types']
    datasets = par['datasets']
    analysis_name = par['analysis_name']
    
    config = get_config_fa(analysis_name)
    data_type = config['data_type']
    granularity = config['granularity']
    
    print('Loading data...')
    print(f'  - Granularity: {granularity}')
    
    # Load aging hallmark genes
    aging_hallmark_genes_path = f'{PRIOR_DIR}/aging_hallmark_genes.csv'
    if not os.path.exists(aging_hallmark_genes_path):
        raise FileNotFoundError(f'Aging hallmark genes file not found at {aging_hallmark_genes_path}')
    
    aging_hallmark_genes_df = pd.read_csv(aging_hallmark_genes_path)
    aging_hallmark_genes = aging_hallmark_genes_df['gene'].tolist()
    
    print(f'Loaded {len(aging_hallmark_genes)} aging hallmark genes')

    adata_dict = {dataset: retrieve_adata(dataset=dataset, data_type=data_type, granularity=granularity) for dataset in datasets}

    print('Calculating aging hallmark gene expression...')
    
    for cell_type in tqdm(cell_types, desc='cell types'):
        for dataset in datasets:
            adata = adata_dict[dataset][adata_dict[dataset].obs[granularity]==cell_type]
            # Filter to only aging hallmark genes
            available_genes = [g for g in aging_hallmark_genes if g in adata.var_names]
            if len(available_genes) == 0:
                raise ValueError(f'Warning: No aging hallmark genes found in dataset {dataset}, cell type {cell_type}')
            
            adata = adata[:, available_genes].copy()
            
            write_feature_data(adata, dataset, cell_type, analysis_name=analysis_name)

def wrapper_ct_pol_dist(par):
    """
    Calculate mean absolute TF activity difference between naive and effector subtypes.
    For CD8T: Tcm_Naive_CD8 vs Tem_Temra_CD8
    For CD4T: Tcm_Naive_CD4 vs Tem_Effector_CD4
    """
    from hiara.src.config import mapping_major_2_minor
    
    cell_types = par['cell_types']
    datasets = par['datasets']
    analysis_name = par['analysis_name']
    
    # Define naive-effector pairs
    pol_pairs = {
        'CD8T': ('Tcm_Naive_CD8', 'Tem_Temra_CD8'),
        'CD4T': ('Tcm_Naive_CD4', 'Tem_Effector_CD4')
    }
    
    print('Calculating cell type polarization distance using mean absolute TF activity difference...')
    
    for cell_type in tqdm(cell_types, desc='cell types'):
        if cell_type not in pol_pairs:
            print(f'Skipping {cell_type} - no polarization pair defined')
            continue
            
        naive_ct, effector_ct = pol_pairs[cell_type]
        
        for dataset in datasets:
            # Load TF activity for both subtypes
            adata_naive = retrieve_feature_data(
                analysis_name='tfa_sub_b',
                dataset=dataset,
                cell_type=naive_ct,
                condition='healthy'
            )
            
            adata_effector = retrieve_feature_data(
                analysis_name='tfa_sub_b',
                dataset=dataset,
                cell_type=effector_ct,
                condition='healthy'
            )
            
            if len(adata_naive) != len(adata_effector):
                groups_a = adata_naive.obs['bulk_group'].unique()
                groups_b = adata_effector.obs['bulk_group'].unique()
                # keep only common donors
                common_groups = np.intersect1d(groups_a, groups_b)
                adata_naive = adata_naive[adata_naive.obs['bulk_group'].isin(common_groups)]
                adata_effector = adata_effector[adata_effector.obs['bulk_group'].isin(common_groups)]
                assert len(adata_naive) == len(adata_effector), f'After filtering to common donors, still mismatch in number of samples for {dataset}, {cell_type}'
            # Sort by donor_id
            adata_naive = adata_naive[adata_naive.obs['bulk_group'].argsort()]
            adata_effector = adata_effector[adata_effector.obs['bulk_group'].argsort()]
            
            # Get common TFs
            common_tfs = np.intersect1d(adata_naive.var_names, adata_effector.var_names)
            adata_naive = adata_naive[:, common_tfs]
            adata_effector = adata_effector[:, common_tfs]
            
            # Get dense matrices
            X_naive = adata_naive.X.toarray() if issparse(adata_naive.X) else adata_naive.X
            X_effector = adata_effector.X.toarray() if issparse(adata_effector.X) else adata_effector.X
            
            # Calculate mean absolute difference across all TFs per donor
            pol_dist = np.mean(np.abs(X_effector - X_naive), axis=1)
            
            # Create new AnnData with single polarization distance feature
            pol_adata = ad.AnnData(
                X=pol_dist.reshape(-1, 1),
                obs=adata_naive.obs.copy(),
                var=pd.DataFrame(index=['pol_dist'])
            )
            
            write_feature_data(pol_adata, dataset, cell_type, analysis_name=analysis_name)

def _wrapper_cell_cell_communication(analysis_name, par):
    """
    Calculate cell-cell communication scores using ligand-receptor pairs from CellPhoneDB.
    Analyzes communication across ALL cell type subtypes (e.g., CD8T subtypes → CD4T subtypes).
    Features are named: {sender_subtype}_{ligand}_{receiver_subtype}_{receptor}
    This creates a donor-level feature matrix (not subtype-level).
    
    Note: Since communication is across all subtypes, cell_type parameter is used to create
    a single combined output per dataset (we use 'all' as the cell_type key).
    """
    from omnipath.interactions import import_intercell_network
    
    datasets = par['datasets']
    condition = par.get('condition', 'healthy')
    
    print('Loading CellPhoneDB ligand-receptor database...')
    lr_pairs = import_intercell_network(
        interactions_params={'datasets': 'cellphonedb'},
        transmitter_params={'categories': 'ligand'},
        receiver_params={'categories': 'receptor'}
    )
    print(f'  - Loaded {len(lr_pairs)} ligand-receptor pairs')
    
    config = get_config_fa(analysis_name)
    data_type = config['data_type']
    
    print('Calculating cell-cell communication scores ACROSS all cell types...')
    print(f'  - Data type: {data_type}')
    print(f'  - Feature naming: {{sender_subtype}}_{{ligand}}_{{receiver_subtype}}_{{receptor}}')
    
    for dataset in datasets:
        print(f'\nProcessing {dataset}...')
        
        # Load ALL data at once (all cell types, all subtypes)
        adata_all = retrieve_adata(
            dataset=dataset,
            data_type=data_type,
            condition=condition,
            granularity=SUB_CT_LABEL
        )
        
        # Get all unique subtypes
        all_subtypes = adata_all.obs[SUB_CT_LABEL].unique()
        print(f'  Found {len(all_subtypes)} subtypes: {list(all_subtypes)}')
        
        # Get all unique donors from the full dataset
        all_donors = sorted(adata_all.obs['bulk_group'].unique())
        
        if len(all_donors) < 10:
            print(f'  Skipping {dataset} - only {len(all_donors)} donors')
            continue
        
        print(f'  Found {len(all_donors)} donors')
        
        # Split by subtype for easier processing
        subtype_adatas = {
            subtype: adata_all[adata_all.obs[SUB_CT_LABEL] == subtype].copy()
            for subtype in all_subtypes
        }
        
        # Calculate communication scores for ALL subtype pairs (across cell types)
        comm_scores_dict = {}
        
        for sender_subtype, adata_sender in tqdm(subtype_adatas.items(), desc=f'{dataset} - sender subtypes'):
            
            for receiver_subtype, adata_receiver in subtype_adatas.items():
                
                # Find common donors between THIS sender-receiver pair
                sender_donors = set(adata_sender.obs['bulk_group'].unique())
                receiver_donors = set(adata_receiver.obs['bulk_group'].unique())
                pair_common_donors = sorted(sender_donors & receiver_donors)
                
                if len(pair_common_donors) == 0:
                    continue
                
                # For each L-R pair, calculate communication score
                for _, row in lr_pairs.iterrows():
                    # OmniPath intercell network uses 'source_genesymbol' and 'target_genesymbol'
                    ligand = row.get('source_genesymbol', row.get('genesymbol_intercell_source', ''))
                    receptor = row.get('target_genesymbol', row.get('genesymbol_intercell_target', ''))
                    
                    if not ligand or not receptor:
                        continue
                    
                    # Check if ligand and receptor are expressed
                    if ligand not in adata_sender.var_names or receptor not in adata_receiver.var_names:
                        continue
                    
                    # Get expression per donor (mean across cells in each donor) for common donors only
                    ligand_expr_per_donor = adata_sender[:, ligand].to_df().groupby(adata_sender.obs['bulk_group']).mean()[ligand]
                    receptor_expr_per_donor = adata_receiver[:, receptor].to_df().groupby(adata_receiver.obs['bulk_group']).mean()[receptor]
                    
                    # Filter to common donors and ensure same order
                    ligand_expr_per_donor = ligand_expr_per_donor.loc[pair_common_donors]
                    receptor_expr_per_donor = receptor_expr_per_donor.loc[pair_common_donors]

                    # Communication score = ligand_expression * receptor_expression
                    pair_comm_score = ligand_expr_per_donor.values * receptor_expr_per_donor.values
                    
                    # Feature name
                    feature_name = f'{sender_subtype}_{ligand}_{receiver_subtype}_{receptor}'
                    
                    # Store scores for all donors (fill with NaN for donors not in this pair)
                    if feature_name not in comm_scores_dict:
                        comm_scores_dict[feature_name] = pd.Series(index=all_donors, dtype=float)
                    
                    comm_scores_dict[feature_name].loc[pair_common_donors] = pair_comm_score
        
        if not comm_scores_dict:
            print(f'  No communication scores calculated for {dataset}')
            continue
        
        # Create feature matrix (donors × L-R pairs) from dict of Series
        comm_df = pd.DataFrame(comm_scores_dict)
        
        # Remove donors with all NaN values (donors that don't have any of the subtypes)
        comm_df = comm_df.dropna(how='all')
        
        if comm_df.shape[0] < 10:
            print(f'  Skipping {dataset} - only {comm_df.shape[0]} donors with data after filtering')
            continue
        
        # Create AnnData object
        # Get metadata from the full dataset
        obs_metadata = adata_all.obs[adata_all.obs['bulk_group'].isin(comm_df.index)].copy()
        obs_metadata = obs_metadata.drop_duplicates(subset='bulk_group').set_index('bulk_group')
        obs_metadata = obs_metadata.loc[comm_df.index]  # Ensure same order
        
        comm_adata = ad.AnnData(
            X=comm_df.values,
            obs=obs_metadata,
            var=pd.DataFrame(index=comm_df.columns)
        )
        
        # Add dataset info
        comm_adata.obs['dataset'] = dataset
        comm_adata.uns['dataset'] = dataset
        
        # Write feature data - use 'all' as cell_type since it spans all cell types
        write_feature_data(comm_adata, dataset=dataset, cell_type='all', analysis_name=analysis_name)
        print(f'  ✓ {dataset}: {comm_adata.shape[0]} donors × {comm_adata.shape[1]} L-R pairs across all subtypes')



def wrapper_ccc(analysis_name, par, n_jobs=1): 
    import liana as li   
    datasets = par['datasets']
    condition = par['condition']

    config = get_config_fa(analysis_name)
    granularity = config['granularity']
    data_type = config['data_type']
    cell_types = config['cell_types']
    
    # Handle multiple conditions - convert to list if needed
    conditions = condition if isinstance(condition, list) else [condition]
    
    for dataset in datasets:
        print(f'\nProcessing {dataset}...')
        # Load ALL data at once (all cell types, all subtypes)
        for cell_type in cell_types:
            # Process each condition separately and concatenate
            condition_adatas = []
            
            for cond in conditions:
                print(f'  Condition: {cond}')
                adata = retrieve_adata(
                    dataset=dataset,
                    data_type=data_type,
                    condition=cond,
                    test_mode=par['test_mode'],
                    cell_type=cell_type,
                )
                
                if adata.shape[0] == 0:
                    print(f'    No samples for {cell_type} in {dataset} with condition {cond}')
                    continue
                    
                # Get all unique donors for this condition
                all_donors = sorted(adata.obs['donor_age'].unique())
                print(f'    Found {len(all_donors)} donors')
                
                # Dictionary to store communication scores per donor
                # Key: feature_name (source_ligand_target_receptor)
                # Value: Series indexed by donor_age
                comm_scores_dict = {}
                for donor_age in tqdm(all_donors, desc=f'{dataset} - {cond} - donors'):
                    adata_donor = adata[adata.obs['donor_age'] == donor_age].copy()
                    # Run LIANA rank_aggregate for this donor
                    lr_results = li.mt.rank_aggregate(
                        adata_donor,
                        groupby=SUB_CT_LABEL,   # Communication between subtypes
                        resource_name="consensus",  # Uses consensus LR database
                        n_jobs=n_jobs,
                        expr_prop=0.1,  # Min expression proportion
                        n_perms=None,  # Skip permutation test for speed
                        verbose=False,
                        use_raw=False,
                        inplace=False
                    )
                    
                    # Extract communication scores
                    # Use 'lrscore' as the main metric (LIANA's aggregate score)
                    for _, row in lr_results.iterrows():
                        source = row['source']
                        target = row['target']
                        ligand = row['ligand_complex']
                        receptor = row['receptor_complex']
                        score = row['lrscore']  # Use LIANA's aggregate score
                        assert np.isnan(score) == False, f'NaN score for {source} → {target} ({ligand} → {receptor}) in donor {donor_age}, skipping'
                        
                        # Create feature name: source_ligand_target_receptor
                        feature_name = f'{source}__{ligand}__{target}__{receptor}'
                        
                        # Initialize if not exists
                        if feature_name not in comm_scores_dict:
                            comm_scores_dict[feature_name] = pd.Series(index=all_donors, dtype=float)
                        
                        # Store score for this donor
                        comm_scores_dict[feature_name].loc[donor_age] = score
                
                if not comm_scores_dict:
                    print(f'    No communication scores calculated for {dataset}, {cond}')
                    continue
                
                # Create feature matrix (donors × L-R pairs)
                comm_df = pd.DataFrame(comm_scores_dict)
                
                # Remove donors with all NaN (shouldn't happen, but safety check)
                comm_df = comm_df.dropna(how='all')
                
                # Create AnnData object
                # Get metadata from the full dataset
                obs_metadata = adata.obs[adata.obs['donor_age'].isin(comm_df.index)].copy()
                obs_metadata = obs_metadata.drop_duplicates(subset='donor_age').set_index('donor_age')
                obs_metadata = obs_metadata.loc[comm_df.index]  # Ensure same order
                
                comm_adata = ad.AnnData(
                    X=comm_df.values,
                    obs=obs_metadata,
                    var=pd.DataFrame(index=comm_df.columns)
                )
                
                # Add dataset and condition info
                comm_adata.obs['dataset'] = dataset
                comm_adata.uns['dataset'] = dataset
                comm_adata.obs[granularity] = cell_type
                comm_adata.obs['condition'] = cond
                
                condition_adatas.append(comm_adata)
                print(f'    ✓ {dataset} - {cond}: {comm_adata.shape[0]} donors × {comm_adata.shape[1]} L-R pairs')
            
            # Concatenate all conditions
            if len(condition_adatas) == 0:
                raise ValueError(f'No communication scores calculated for {dataset}, {cell_type} across all conditions')
            elif len(condition_adatas) == 1:
                comm_adata_combined = condition_adatas[0]
            else:
                # Concatenate with fill_value=np.nan (not 0) to preserve missing data structure
                # This is important for proper variance calculation
                comm_adata_combined = ad.concat(condition_adatas, join='outer', fill_value=np.nan)
                # Make observation names unique by combining with condition
                comm_adata_combined.obs_names = [
                    f"{idx}__{row['condition']}" 
                    for idx, row in comm_adata_combined.obs.iterrows()
                ]
                comm_adata_combined.obs_names_make_unique()
            
            # Write combined feature data
            write_feature_data(comm_adata_combined, dataset=dataset, cell_type=cell_type, analysis_name=analysis_name)
            print(f'  ✓ {dataset}: Combined {comm_adata_combined.shape[0]} donors × {comm_adata_combined.shape[1]} L-R pairs')


def association_with_age(adata, association_type, gene_col='gene'):
    '''
    Calculate p-values for the linear regression or Spearman correlation
    of the top tfs across datasets with ageing, and apply FDR correction
    (Benjamini-Hochberg). Includes safeguards and prints diagnostics when
    values are invalid.
    '''
    def process_gene(gene):
        mask_gene = adata.var_names == gene
        adata_sub = adata[:, mask_gene]

        df = adata_sub.to_df()
        df = df.reset_index(drop=True)
        df['age'] = adata_sub.obs['age'].values
        
        # Remove NaN values (donors missing this feature/subtype)
        valid_mask = ~df[gene].isna()
        df = df[valid_mask]
        
        if len(df) < 10:
            # Not enough non-NaN samples for reliable statistics
            return {
                gene_col: gene,
                'p_value': 1.0,
                'slope': 0.0,
                'n_samples': len(df)
            }
        
        df.sort_values('age', inplace=True)

        ages = df['age'].values
        expression = df[gene].values

        # Safeguard: need variance in both age and expression
        if len(np.unique(ages)) <= 1:
            print(f"[SKIP] {gene}: only one unique age → assigning p=1.0, slope=0.0")
            p_value, slope = 1.0, 0.0
        elif np.std(expression) == 0:
            print(f"[SKIP] {gene}: expression constant → assigning p=1.0, slope=0.0")
            p_value, slope = 1.0, 0.0
        else:
            if association_type == 'linear':
                slope, intercept, r_value, p_value, _ = linregress(ages, expression)
            elif association_type == 'spearman':
                slope, p_value = spearmanr(ages, expression)
            else:
                raise ValueError("association_type must be 'linear' or 'spearman'")

            # Handle invalid results
            if p_value is None or np.isnan(p_value) or p_value < 0 or p_value > 1:
                print(f"[WARN] {gene}: invalid p-value {p_value}")
                raise ValueError('NaN p-value encountered')
            if slope is None or np.isnan(slope):
                print(f"[WARN] {gene}: invalid slope {slope}")
                raise ValueError('NaN slope encountered')
            if abs(slope) > 1:
                print(f"[CHECK] {gene}: unusually high slope = {slope:.3f} ({association_type})")

        return {
            gene_col: gene,
            'p_value': float(p_value),
            'slope': float(slope),
            'n_samples': len(df)
        }

    # Parallelize gene processing
    with ThreadPoolExecutor(max_workers=20) as executor:
        p_value_store = list(executor.map(process_gene, adata.var_names))

    stats_df = pd.DataFrame(p_value_store)

    # Apply FDR correction
    if not stats_df.empty:
        adj_p = multipletests(stats_df['p_value'], method='fdr_bh')[1]
        if np.any(np.isnan(adj_p)):
            print("[WARN] NaN values detected in adjusted p-values")
            raise ValueError('NaN adjusted p-values encountered')
            
        stats_df['p_value_adj'] = adj_p
    else:
        stats_df['p_value_adj'] = []
    
    return stats_df

def compute_trend(analysis_name, df, pval_col='meta_p_adj', slope_col='slope', col='gene'):
    # Compute -log10(p_value_adj) for dot size
    df["neg_log10_adj_pval"] = -np.log10(df[pval_col])
    if 'trend' in df.columns:
        df.drop('trend', inplace=True, axis=1)
    increase_trend = get_config_fa(analysis_name)['trend_labels'][0] if 'trend_labels' in get_config_fa(analysis_name) else 'Increase in aging'
    decrease_trend = get_config_fa(analysis_name)['trend_labels'][1] if 'trend_labels' in get_config_fa(analysis_name) else 'Decrease in aging'
    # Function to assign trend based on slope sign and min_degree
    def determine_trend(x):
        pos = (x > 0).sum()
        neg = (x < 0).sum()
        total = len(x)
        threshold = total
        
        # print(pos, neg, total, threshold)
        if pos >= (threshold):
            return increase_trend
        elif neg >= (threshold):
            return decrease_trend
        else:
            return 'Inconsistent'

    # Apply the function group-wise
    trend = df.groupby([col, 'cell_type'])[slope_col].apply(determine_trend).reset_index(name='trend')
    # trend = trend.dropna()

    df = df.merge(trend, on=[col, 'cell_type'], how='left')
    df["trend"] = pd.Categorical(
        df["trend"], 
        categories=['Increase in aging', 'Decrease in aging', "Inconsistent"], 
        ordered=True
    )
    return df

def tf_activity_local(net, adata, tf_all=None):
    net = net.pivot(index='source', columns='target', values='weight').fillna(0)
    net = net[[g for g in adata.var_names if g in net.columns]]
    if tf_all is not None:
        tfs_present = np.intersect1d(net.index, tf_all)
    else:
        tfs_present = net.index
    net = net[net.index.isin(tfs_present)]
    # print('ratio of porosity: ', (net==0).sum().sum()/net.size)
    # - subset the adata
    adata = adata[:, adata.var_names.isin(net.columns)]
    # - enrich tfs 
    X = adata.X
    X = X.toarray() if scipy.sparse.issparse(X) else X
    mat = X.T
    
    tf_acts = np.dot(net, mat)
    # - format
    tf_acts = pd.DataFrame(tf_acts, index=net.index, columns=adata.obs.index)
    tf_acts = tf_acts.reset_index().melt(id_vars='source', var_name='sample', value_name='activity')

    # cols = [c for c in adata.obs.columns if c not in ['sample', 'cell_type', 'cell_count']]
    cols = adata.obs.columns

    tf_acts = tf_acts.set_index('sample').merge(adata.obs[cols], left_index=True, right_index=True).reset_index(drop=False)
    # print(f"net: {net.shape}, mat: {mat.shape}, source: {tf_acts['source'].nunique()}")
    
    if 'sample' not in tf_acts.columns:
        tf_acts['sample'] = tf_acts['index']
    
    # Calculate ranks within each sample
    # tf_acts['rank'] = tf_acts.groupby('sample')['activity'].transform(lambda x: x.abs().rank(method='dense', ascending=False)) 

    return tf_acts

def calculate_tf_activity(adata, net, tf_all=None, tmin=TF_MIN_TARGET):    
    # - TFs
    if tf_all is not None:
        net = net[net['source'].isin(tf_all)]
    if True: # run decoupler
        import decoupler as dc

        dc.mt.ulm(adata, net, tmin=tmin)
        tf_acts_X = adata.obsm['score_ulm']
        var = pd.DataFrame({'source': tf_acts_X.columns})
        var.index = var['source']
        
        tf_acts_adata = ad.AnnData(X=tf_acts_X.values, obs=adata.obs, var=var)

    else: # run my implementation
        n_targets_t = tmin
        if True:
            tf_size = net.groupby('source').size()
            tfs = tf_size[tf_size > n_targets_t].index
            
            net = net[net['source'].isin(tfs)]
        tf_acts = tf_activity_local(net, adata, tf_all)
    
        if 'index' in tf_acts.columns:
            tf_acts = tf_acts.drop('index', axis=1)
        
        assert tf_acts.shape[0] != 0, 'Empty'
        assert 'sample' in tf_acts.columns, 'sample not in tf_acts' 

        tf_acts['age'] = pd.to_numeric(tf_acts['age'], errors='coerce')

        # - create anndata
        X_df = tf_acts.pivot(index='sample', columns='source', values='activity')
        obs_df = tf_acts.drop_duplicates(subset='sample').set_index('sample')[[c for c in tf_acts.columns if c not in ['source', 'activity', 'sample']]]
        obs_df = obs_df.loc[X_df.index]
        var_df = pd.DataFrame(index=X_df.columns)
        var_df['source'] = var_df.index
        tf_acts_adata = ad.AnnData(X=X_df.values, obs=obs_df, var=var_df)
        # print(tf_acts_adata)
        # aaa
    return tf_acts_adata


def filter_for_consistent_trends(df):
    """
    Filter dataframe to keep only TFs with consistent trend direction across all cell types.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: cell_type, tf, slope
    
    Returns
    -------
    pd.DataFrame
        Filtered dataframe with only consistent trends
    """
    consistent_groups = (
        df.groupby(['cell_type', 'gene'])['slope']
        .apply(lambda x: np.sign(x).nunique() == 1)
    )
    valid_tuples = consistent_groups[consistent_groups].index
    filtered_df = df.set_index(['cell_type', 'gene']).loc[valid_tuples].reset_index()
    return filtered_df
