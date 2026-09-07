import numpy as np
from scipy.stats import linregress, spearmanr, norm, rankdata, pearsonr, combine_pvalues
import pandas as pd
import os
import scipy
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import gc

import scanpy as sc
import anndata as ad
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from hira.src.config import GRNS_DIR, get_config_fa, FEATURES_DIR, MAJOR_CTS, get_config, surrogate_names, DISCOVERY_COHORTS, NET_SKELETON
from tqdm import tqdm
from hira.src.config import OUTPUT_DIR, FEATURES_DIR, FEATURE_DATA_DIR, CORR_THRESHOLD, TF_MIN_TARGET, FEATURE_TYPES, SUB_CT_LABEL, MAJOR_CT_LABEL
from hira.src.config import CONFOUND_COVARIATES, COARSENED_COVARIATES
from scipy.sparse import issparse
from hira.src.utils.util import retrieve_adata, retrieve_net_consensus, retrieve_net, coarsen
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
        

def retrieve_stats(analysis_name, cell_type=None, dataset=None, multi_cohort=None, features_dir=None, suffix='', feature_props=[]):
    feature_type = get_config_fa(analysis_name)['feature_type']
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
    stats['cell_type'] = stats['cell_type'].astype(str)
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

        mask_slope = stats["pooled_rho"].abs() > CORR_THRESHOLD
        mask_trend = stats["sign_consistent"].astype(bool)

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

    if 'centrality' in feature_props:
        for ct in stats['cell_type'].unique():
            stats_ct = stats['cell_type'] == ct
            grn = retrieve_net_consensus(cell_type=ct, grns_dir=GRNS_DIR)
            assert grn.groupby(['source', 'target']).size().max() == 1, f"GRN for cell type {ct} has duplicate edges, cannot determine centrality. Please check the GRN data."
            if feature_type == 'tf_activity':
                feature_col = 'source' 
            elif feature_type in 'gene_expression':
                feature_col = 'target'
            else:
                raise ValueError(f'Unknown feature type {feature_type} for centrality calculation')
            c = grn.groupby(feature_col).size().reset_index(name='centrality')
            c.rename(columns={feature_col: 'gene'}, inplace=True)
            
            # Normalize centrality
            c['centrality'] = c['centrality'] / c['centrality'].max()
            
            # Merge centrality into stats for this cell type
            stats.loc[stats_ct, 'centrality'] = stats.loc[stats_ct, 'gene'].map(
                c.set_index('gene')['centrality']
            )


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

# analysis_name -> cached tf_activity feature dir to read from, when it differs from
# analysis_name itself (i.e. the analysis only changes the association step, not the features)
FA_ANALYSIS_FEATURE_SOURCE = {'tfa_major_b_ctNaiveToEffector': 'tfa_major_b'}
NAIVE_EFFECTOR_COLS = {
    'CD8T': ('Tcm_Naive_CD8_count', ['Tem_Trm_CD8_count', 'Tem_Temra_CD8_count']),  # MAIT excluded
    'CD4T': ('Tcm_Naive_CD4_count', ['Tem_Effector_CD4_count']),
    'MONO': ('Classic_MONO_count', ['NonClassic_MONO_count']),
}


def add_naive_ratio(adata, cell_type):
    """naive_ratio = naive / (naive + effector) cells per donor, from the
    {minor_ct}_count columns baked into the bulk pseudobulk (see bulkify/script.py)."""
    naive_col, effector_cols = NAIVE_EFFECTOR_COLS[cell_type]
    naive = adata.obs[naive_col].astype(float)
    effector = adata.obs[effector_cols].astype(float).sum(axis=1)
    total = naive + effector
    adata.obs['naive_ratio'] = np.where(total > 0, naive / total, np.nan)
    return adata


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

    if get_config_fa(analysis_name)['feature_type'] == 'gene_expression':
        granularity = get_config_fa(analysis_name)['granularity']
        data_type = get_config_fa(analysis_name)['data_type']
        adata = retrieve_adata(dataset=dataset, 
                               data_type=data_type, 
                               cell_type=cell_type, 
                               only_net_genes=True, 
                               condition=condition,
                               granularity=granularity)
        
    else:
        # tfa_major_b_ctNaiveToEffector reuses tfa_major_b's cached TF-activity features
        # (no recompute), it only adds a naive_ratio covariate for the association step.
        source_analysis = FA_ANALYSIS_FEATURE_SOURCE.get(analysis_name, analysis_name)
        file_path = f'{FEATURE_DATA_DIR}/{source_analysis}/{dataset}_{cell_type}{suffix}.h5ad'
        if os.path.exists(file_path) == False:
            raise ValueError(f'File {file_path} does not exist')
        adata = ad.read_h5ad(file_path)
        if analysis_name == 'tfa_major_b_ctNaiveToEffector':
            adata = add_naive_ratio(adata, cell_type)
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
    output_dir = f'{FEATURE_DATA_DIR}/{analysis_name}'
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
                from hira.src.utils.util import test_mixed_effects
                
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

def _fisher_meta(p_values, slopes):
    """Fisher's method on per-cohort p-values, plus mean rho and sign agreement.

    Returns (meta p, mean rho, all cohorts same sign). Fisher saturates at these cohort
    sizes on its own, so selection relies on the sign-agreement and |rho| > CORR_THRESHOLD
    filters applied downstream in retrieve_stats.
    """
    meta_p = combine_pvalues(np.clip(p_values, 1e-20, None), method='fisher')[1]
    sign = np.sign(slopes)
    return meta_p, float(np.mean(slopes)), bool((sign > 0).all() or (sign < 0).all())


def run_meta_analysis(stats_all, min_degree=2):
    """Meta-analysis across cohorts, per gene x cell type.

    Merges meta_p, meta_p_adj (BH within cell type), pooled_rho and sign_consistent
    into stats_all. Fisher alone calls ~all TFs significant at these cohort sizes; a
    feature counts as significant only once retrieve_stats also requires sign agreement
    across cohorts and |pooled_rho| > CORR_THRESHOLD.
    """
    assert stats_all.shape[0] > 0, 'No stats for meta analysis'
    print('Meta analysis (Fisher + sign agreement + |rho| threshold)...')
    df = stats_all
    if min_degree is not None:
        df = df.groupby(['gene', 'cell_type']).filter(lambda g: g['dataset'].nunique() >= min_degree)
    if df.empty:
        print(f"No meta analysis results for {stats_all['cell_type'].unique()}")
        return None

    meta = pd.DataFrame(
        [(gene, cell_type, *_fisher_meta(g['p_value'].values, g['slope'].values))
         for (gene, cell_type), g in df.groupby(['gene', 'cell_type'], sort=False)],
        columns=['gene', 'cell_type', 'meta_p', 'pooled_rho', 'sign_consistent'],
    )
    meta['meta_p_adj'] = meta.groupby('cell_type')['meta_p'].transform(
        lambda p: multipletests(p.values, method='fdr_bh')[1])
    assert meta.isna().sum().sum() == 0, 'NaN values found in meta analysis results'
    for cell_type, n in meta.groupby('cell_type').size().items():
        print(cell_type, n)

    return stats_all.merge(meta, on=['gene', 'cell_type'], how='left')


def wrapper_meta_analysis(analysis_name, stats_features, par):
    feature_col = 'gene'
    min_degree = par['META_MIN_COHORT']
    def run_func(min_degree):
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
            meta_stats = run_meta_analysis(stats, min_degree=min_degree)
            meta_stats = compute_trend(analysis_name, meta_stats, col=feature_col)
            stats_store.append(meta_stats)
        if len(stats_store) > 0:
            stats_discovery = pd.concat(stats_store)
            stats_discovery['condition'] = 'healthy'
            return stats_discovery
        else:
            return pd.DataFrame()
    
    print(f'Running meta analysis for all datasets, min degree {min_degree}')
    stats = run_func(min_degree)

    #- save
    return stats

def _association_for_cell_type(task):
    """Per-cell-type worker for wrapper_association_with_age_condition, run in a
    separate process (see ProcessPoolExecutor there) since each cell type's
    per-gene association test is independent and CPU-bound."""
    (cell_type, datasets, analysis_name, condition, suffix, features,
     granularity, association_type, config, test_type) = task

    stats_store = []
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

        if adata.shape[0] < 3:
            raise ValueError(f'Not enough samples for {cell_type} in {dataset}, only {adata.shape[0]} samples')

        if issparse(adata.X):
            adata.X = adata.X.toarray()

        if association_type == 'continous':
            print('Aging analysis for', cell_type, 'in', dataset)
            extra_covariates = ['naive_ratio'] if 'naive_ratio' in adata.obs.columns else None
            categorical_covariates = CONFOUND_COVARIATES.get(dataset, [])
            for c in categorical_covariates:
                if c not in adata.obs.columns and c in COARSENED_COVARIATES:
                    adata.obs[c] = coarsen(adata.obs[COARSENED_COVARIATES[c]])
            stats = association_with_age(adata, association_type='partial_spearman', extra_covariates=extra_covariates,
                                          categorical_covariates=categorical_covariates)
            stats['condition'] = condition
            stats['comparison'] = 'aging'
        elif association_type == 'grouped':
            print('Condition analysis for', cell_type, 'in', dataset)
            stats = associate_with_condition(
                adata,
                config,
                test_type=test_type
            )
        else:
            raise ValueError(f'Unknown association type: {association_type}')
        if stats is None or len(stats) == 0:
            raise ValueError(f'No stats calculated for {cell_type} in {dataset}, something went wrong')

        stats['dataset'] = dataset
        stats['cell_type'] = cell_type
        stats_store.append(stats)
    return stats_store


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
    tasks = [
        (cell_type, datasets, analysis_name, condition, suffix, features,
         granularity, association_type, config, test_type)
        for cell_type in cell_types
    ]
    stats_store = []
    if len(tasks) == 1:
        stats_store.extend(_association_for_cell_type(tasks[0]))
    else:
        with ProcessPoolExecutor(max_workers=min(len(tasks), 5)) as executor:
            for result in tqdm(executor.map(_association_for_cell_type, tasks), total=len(tasks), desc='cell types'):
                stats_store.extend(result)

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


def subsample_cells_per_group(adata, bulk_group, max_cells, seed=0):
    # ponytail: bounds memory/runtime for per-cell ULM; a median over max_cells is already stable, full completeness isn't needed
    group_cols = [c for c in bulk_group if c != 'cell_type']
    group_key = adata.obs[group_cols].astype(str).agg('_'.join, axis=1)
    pos = pd.Series(np.arange(adata.n_obs))
    keep = pos.groupby(group_key.values, group_keys=False).apply(
        lambda s: s.sample(n=min(len(s), max_cells), random_state=seed)
    )
    return adata[keep.values].copy()

def aggregate_tf_activity_per_donor(tf_acts_cell, bulk_group):
    # per-cell/per-metacell ULM scores -> per-donor mean. Each unit is depth-normalised
    # independently, so this avoids the pseudobulk cell_count/dropout confound while still
    # tracking subpopulation composition; the median discards it (e.g. GATA3 in CD8T).
    group_cols = [c for c in bulk_group if c != 'cell_type']
    obs = tf_acts_cell.obs
    group_key = obs[group_cols].astype(str).agg('_'.join, axis=1)

    df = tf_acts_cell.to_df()
    donor_agg = df.groupby(group_key.values).mean()

    cell_count = group_key.value_counts().rename('cell_count')
    keep_obs_cols = list(dict.fromkeys(group_cols + ['age', 'condition', 'dataset']))
    donor_obs = obs[keep_obs_cols].copy()
    donor_obs['group'] = group_key.values
    donor_obs = donor_obs.drop_duplicates('group').set_index('group').join(cell_count)
    donor_obs = donor_obs.loc[donor_agg.index]

    var = pd.DataFrame({'source': donor_agg.columns}, index=donor_agg.columns)
    return ad.AnnData(X=donor_agg.values, obs=donor_obs, var=var)

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
    per_donor_agg = analysis_name in ['tfa_major_sc']
    # mc has multiple metacells per donor; must still collapse to one sample/donor to avoid pseudoreplication
    aggregate_per_donor = per_donor_agg or analysis_name == 'tfa_major_mc'
    load_data_type = 'sc' if per_donor_agg else data_type
    max_cells_per_group = 300
    for dataset in datasets:
        bulk_group = get_config(dataset).bulk_group
        if not per_donor_agg:
            adata = retrieve_adata(dataset=dataset, data_type=load_data_type, condition=condition)

        for cell_type in tqdm(cell_types, desc='cell types'):
            print(dataset, load_data_type)
            if per_donor_agg:
                # load only this cell type's cells (not the whole dataset) to bound memory
                adata_t = retrieve_adata(dataset=dataset, data_type=load_data_type, condition=condition,
                                          cell_type=cell_type, granularity=granularity)
                if len(adata_t) > 0:
                    adata_t = subsample_cells_per_group(adata_t, bulk_group, max_cells_per_group)
            else:
                adata_t = adata[adata.obs[granularity] == cell_type].copy()
            if len(adata_t) == 0:
                print(f'No samples for {cell_type} in {dataset}, skipping TF activity calculation')
                continue
            if par['use_consensus_net']:
                net = retrieve_net_consensus(cell_type=cell_type, skeleton='promotor' if promotor_only else NET_SKELETON)
            else:
                net = retrieve_net(dataset=dataset, cell_type=cell_type, skeleton='promotor' if promotor_only else NET_SKELETON)
            if adata_t.obs['condition'].value_counts().min() < 3:
                continue
            tf_acts = calculate_tf_activity(adata_t, net)
            if aggregate_per_donor:
                tf_acts = aggregate_tf_activity_per_donor(tf_acts, bulk_group)
                tf_acts.obs[granularity] = cell_type
            tf_acts.obs['dataset'] = dataset
            tf_acts.uns['dataset'] = dataset
            tf_acts = tf_acts[tf_acts.obs['age'].isna()==False] # there is a bug in the code that causes age to be NaN
            write_feature_data(tf_acts, dataset=dataset, cell_type=cell_type, analysis_name=analysis_name, suffix='_promotor' if promotor_only else '')

def wrapper_genesets_scores(par):
    from hira.src.utils.util import get_genesets
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
    from hira.src.feature_association.trajectory_analysis import load_sc_data
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
    from hira.src.feature_association.trajectory_analysis import load_sc_data, compute_tfa_peg_association
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
    from hira.src.config import PRIOR_DIR
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
    from hira.src.config import mapping_major_2_minor
    cell_types = par['cell_types']
    datasets = par['datasets']
    condition = par['condition']
    if not isinstance(condition, list):
        condition = [condition]
    analysis_name = par['analysis_name']
    pol_pairs = {
        'CD8T': ('Tcm_Naive_CD8', 'Tem_Temra_CD8'),
        'CD4T': ('Tcm_Naive_CD4', 'Tem_Effector_CD4')
    }
    for cell_type in tqdm(cell_types, desc='cell types'):
        if cell_type not in pol_pairs:
            print(f'Skipping {cell_type} - no polarization pair defined')
            continue
        naive_ct, effector_ct = pol_pairs[cell_type]
        for dataset in datasets:
            pol_adata_list = []
            for c in condition:
                adata_naive = retrieve_feature_data(
                    analysis_name='tfa_sub_b',
                    dataset=dataset,
                    cell_type=naive_ct,
                    condition=c
                )
                adata_effector = retrieve_feature_data(
                    analysis_name='tfa_sub_b',
                    dataset=dataset,
                    cell_type=effector_ct,
                    condition=c
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
                pol_adata.obs['condition'] = c
                pol_adata_list.append(pol_adata)
            
            pol_adata = ad.concat(pol_adata_list)

            
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
def wrapper_ct_tf_markers(analysis_name, par):
    print('Loading data...')
    cell_types = par['cell_types']
    datasets = par['datasets']
    condition = par.get('condition', None)
    promotor_only = par['promotor_only']
    
    print('Calculating TF activity...')
    print(f'  - Promotor-based only: {promotor_only}')

    config = get_config_fa(analysis_name)
    data_type = config['data_type']
    for dataset in datasets:
        adata = retrieve_adata(dataset=dataset, data_type=data_type, condition=condition)
        for cell_type in tqdm(cell_types, desc='cell types'):
            print(dataset, data_type)
            adata_t = adata[adata.obs[MAJOR_CT_LABEL] == cell_type].copy()
            if len(adata_t) == 0:
                print(f'No samples for {cell_type} in {dataset}, skipping TF activity calculation')
                continue
            if par['use_consensus_net']:
                net = retrieve_net_consensus(cell_type=cell_type, skeleton='promotor' if promotor_only else NET_SKELETON)
            else:
                net = retrieve_net(dataset=dataset, cell_type=cell_type, skeleton='promotor' if promotor_only else NET_SKELETON)
            if adata_t.shape[0] < 10:
                continue
            tf_acts = calculate_tf_activity(adata_t, net)
            tf_acts.obs['dataset'] = dataset
            tf_acts.uns['dataset'] = dataset
            tf_acts = tf_acts[tf_acts.obs['age'].isna()==False] # there is a bug in the code that causes age to be NaN
            write_feature_data(tf_acts, dataset=dataset, cell_type=cell_type, analysis_name=analysis_name, suffix='_promotor' if promotor_only else '')
 

def wrapper_ccc(analysis_name, par, n_jobs=1): 
    import liana as li   
    datasets = par['datasets']
    condition = par['condition']

    config = get_config_fa(analysis_name)
    granularity = config['granularity']
    data_type = config['data_type']
    cell_type = 'all'  # We will calculate communication across all subtypes, so we use 'all' as the cell_type key for output
    
    # Handle multiple conditions - convert to list if needed
    conditions = condition if isinstance(condition, list) else [condition]
    
    for dataset in datasets:
        print(f'\nProcessing {dataset}...')
        # Load ALL data at once (all cell types, all subtypes)
    
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
                
                # Check if we have at least 2 cell types with cells
                cell_type_counts = adata_donor.obs[granularity].value_counts()
                if len(cell_type_counts) < 2:
                    print(f'    Skipping donor {donor_age}: only {len(cell_type_counts)} cell type(s) present')
                    continue
                
                # Check if any cell type has 0 cells (shouldn't happen but safety check)
                if (cell_type_counts == 0).any():
                    print(f'    Skipping donor {donor_age}: some cell types have 0 cells')
                    continue
                
                # Run LIANA rank_aggregate for this donor
                lr_results = li.mt.rank_aggregate(
                    adata_donor,
                    groupby=granularity,   # Communication between subtypes
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
        
        # Convert categorical/object columns to string to avoid h5ad write errors
        for col in comm_adata_combined.obs.columns:
            if comm_adata_combined.obs[col].dtype == 'object' or isinstance(comm_adata_combined.obs[col].dtype, pd.CategoricalDtype):
                comm_adata_combined.obs[col] = comm_adata_combined.obs[col].astype(str)
        
        # Write combined feature data
        write_feature_data(comm_adata_combined, dataset=dataset, cell_type=cell_type, analysis_name=analysis_name)
        print(f'  ✓ {dataset}: Combined {comm_adata_combined.shape[0]} donors × {comm_adata_combined.shape[1]} L-R pairs')


def shrink_residualize(y, codes, k):
    """Empirical-Bayes partial pooling: residualize y on a categorical grouping
    (codes: 0..k-1) by shrinking each group's mean deviation toward the grand mean
    by its reliability var_between/(var_between + var_within/n_g). Degenerates to a
    full fixed effect for large, even groups and to ~no adjustment for many small
    groups, so unlike raw dummy OLS it can't overfit incidental small-group params
    (see src/exp_analysis/confounders.py and temp/confounder_adjustment for validation)."""
    y = np.asarray(y, dtype=float)
    N = len(y)
    grand_mean = y.mean()
    sums = np.zeros(k)
    np.add.at(sums, codes, y)
    n_g = np.bincount(codes, minlength=k).astype(float)
    group_mean = sums / n_g
    ss_between = (n_g * (group_mean - grand_mean) ** 2).sum()
    ss_total = ((y - grand_mean) ** 2).sum()
    ss_within = max(ss_total - ss_between, 0)
    msb = ss_between / max(k - 1, 1)
    msw = ss_within / max(N - k, 1)
    n0 = (N - (n_g ** 2).sum() / N) / max(k - 1, 1)
    var_between = max((msb - msw) / n0, 0) if n0 > 0 else 0
    shrink = var_between / (var_between + msw / n_g + 1e-300)
    shrunk_group_mean = grand_mean + shrink * (group_mean - grand_mean)
    return y - shrunk_group_mean[codes] + grand_mean


def association_with_age(adata, association_type, gene_col='gene', extra_covariates=None, categorical_covariates=None):
    '''
    Calculate p-values for the linear regression or Spearman correlation
    of the top tfs across datasets with ageing, and apply FDR correction
    (Benjamini-Hochberg). Includes safeguards and prints diagnostics when
    values are invalid.

    extra_covariates: optional list of adata.obs columns to additionally
    control for in the partial_spearman residualization, alongside cell_count
    (e.g. a cell-type ratio, to rule out a composition-shift confound).

    categorical_covariates: optional list of adata.obs columns (e.g. a batch/site
    id) to control for in the partial_spearman residualization via shrink_residualize,
    applied after the numeric covariates. Unlike extra_covariates these are never
    rankdata()'d -- that would silently treat category labels as ordinal.
    '''
    covariate_cols = ['cell_count'] + list(extra_covariates or [])
    categorical_covariates = list(categorical_covariates or [])

    def process_gene(gene):
        mask_gene = adata.var_names == gene
        adata_sub = adata[:, mask_gene]

        df = adata_sub.to_df()
        df = df.reset_index(drop=True)
        df['age'] = adata_sub.obs['age'].values
        for c in covariate_cols:
            df[c] = adata_sub.obs[c].values
        for c in categorical_covariates:
            df[c] = adata_sub.obs[c].values

        # Remove NaN values (donors missing this feature/subtype or a covariate)
        valid_mask = ~df[[gene] + covariate_cols + categorical_covariates].isna().any(axis=1)
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
            # if association_type == 'linear':
            #     X = sm.add_constant(df['age'])
            #     fit = sm.OLS(df[gene], X).fit()
            #     slope, p_value = fit.params['age'], fit.pvalues['age']
            
            if association_type == 'linear':
                X = sm.add_constant(df[['age', 'cell_count']])
                fit = sm.OLS(df[gene], X).fit()
                slope, p_value = fit.params['age'], fit.pvalues['age']
            elif association_type == 'spearman':
                slope, p_value = spearmanr(ages, expression)
            elif association_type == 'partial_spearman':
                # Spearman of age vs feature, adjusted for pseudobulk depth (and any
                # extra_covariates, e.g. cell-type ratio): rank-transform everything,
                # residualize age-ranks and feature-ranks on the covariate ranks via OLS,
                # then correlate the residuals. rho stays in [-1, 1] so the Fisher-z
                # meta-analysis downstream stays valid (one df lost, see _random_effects).
                ra, rx = rankdata(ages), rankdata(expression)
                rc = np.column_stack([rankdata(df[c].values) for c in covariate_cols])
                if not np.all(np.std(rc, axis=0) == 0):  # skip if covariates all constant
                    X = sm.add_constant(rc)
                    resid = lambda y: sm.OLS(y, X).fit().resid
                    ra, rx = resid(ra), resid(rx)
                for c in categorical_covariates:
                    codes, uniques = pd.factorize(df[c].astype(str).values)
                    if len(uniques) > 1:
                        ra = shrink_residualize(ra, codes, len(uniques))
                        rx = shrink_residualize(rx, codes, len(uniques))
                if np.std(ra) == 0 or np.std(rx) == 0:
                    print(f"[SKIP] {gene}: no variance left after covariate adjustment")
                    slope, p_value = 0.0, 1.0
                else:
                    slope, p_value = pearsonr(ra, rx)
            else:
                raise ValueError("association_type must be 'linear', 'spearman' or 'partial_spearman'")

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

def compute_trend(analysis_name, df, pval_col='meta_p_adj', slope_col='pooled_rho', col='gene'):
    """Label each gene x cell type by the direction of its pooled effect."""
    df["neg_log10_adj_pval"] = -np.log10(df[pval_col])
    labels = get_config_fa(analysis_name).get('trend_labels', ['Increase in aging', 'Decrease in aging'])
    df["trend"] = pd.Categorical(
        np.where(df[slope_col] > 0, labels[0], labels[1]),
        categories=labels, ordered=True
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
