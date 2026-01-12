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
from hiara.src.config import FEATURES_DIR, get_config,  surrogate_names, AGING_COHORTS, HIARA_DIR
from tqdm import tqdm
from hiara.src.config import OUTPUT_DIR, FEATURES_DIR
from scipy.sparse import issparse
from hiara.src.utils.util import retrieve_adata, retrieve_net_consensus, retrieve_net
import warnings
warnings.filterwarnings("ignore")


def write_features_stats(stats, data_type, feature_type, multi_cohort=True, dataset=None, suffix=''):    
    os.makedirs(f'{FEATURES_DIR}/{feature_type}/stats', exist_ok=True)
    if multi_cohort:
        file_name = f'{FEATURES_DIR}/{feature_type}/stats/stats_multi_cohort_{data_type}{suffix}.csv'
    else:
        file_name = f'{FEATURES_DIR}/{feature_type}/stats/stats_{dataset}_{data_type}{suffix}.csv'
    print(f"✓ Condition stats saved: {file_name}")
    stats.to_csv(file_name, index=False)
        

def retrieve_features_stats(data_type, feature_type, cell_type=None, multi_cohort=True, dataset=None):
    if multi_cohort:
        stats = pd.read_csv(f'{FEATURES_DIR}/{feature_type}/stats/stats_multi_cohort_{data_type}.csv')
    else:
        stats = pd.read_csv(f'{FEATURES_DIR}/{feature_type}/stats/stats_{dataset}_{data_type}.csv')
        
    # print(stats)
    if cell_type is not None: 
        if cell_type not in stats['cell_type'].unique():
            raise ValueError(f'Given cell type "{cell_type}" not in {stats["cell_type"].unique()}')
        stats = stats[stats['cell_type'] == cell_type]
    if 'comparison' in stats.columns:
        config = get_config(dataset)
        stats['comparison'] = stats['comparison'].map(lambda name: config.name_mapping.get(name, name))
    return stats

def retrieve_sig_stats(data_type='bulk', feature_type='tf_activity', filter_inconsistent=True, cell_type=None, multi_cohort=True, dataset=None):
    stats = retrieve_features_stats(data_type=data_type, feature_type=feature_type, multi_cohort=multi_cohort, dataset=dataset)
    slope_t = 0.1
    if multi_cohort:  
        p_val_col = 'meta_p_adj'
        stats = stats[stats[p_val_col] < 0.05]
        if filter_inconsistent:
            stats = stats[stats['trend'] != 'Inconsistent']
        
    else:
        p_val_col = 'p_value_adj'
        stats = stats[stats[p_val_col] < 0.05]
    if cell_type is not None:
        if cell_type not in stats['cell_type'].unique():
            raise ValueError(f'Given cell type "{cell_type}" not in {stats["cell_type"].unique()}')
        stats = stats[stats['cell_type'] == cell_type]
        
    slope_col = 'slope'
    
    # Slope-based filtering: Keep only genes where ALL datasets have abs(slope) > slope_t
    def has_consistent_slope(group):
        return (group[slope_col].abs() > slope_t).all()
    stats = stats.groupby(['gene', 'cell_type']).filter(has_consistent_slope)
    
    return stats

def retrieve_feature_data(dataset, cell_type=None, data_type='bulk', feature_type='tf_activity', condition='healthy', suffix=''):
    if feature_type == 'gene_expression':
        adata = retrieve_adata(dataset=dataset, cell_type=cell_type, data_type=data_type)
        return adata
    file_path = f'{FEATURES_DIR}/{feature_type}/{data_type}/{dataset}_{cell_type}{suffix}.h5ad'
    if os.path.exists(file_path) == False:
        raise ValueError(f'File {file_path} does not exist')

    adata = ad.read_h5ad(file_path)

    # Filter by condition
    if condition is not None and 'condition' in adata.obs.columns:
        if condition not in adata.obs['condition'].unique():
            raise ValueError(f'Error in retrieving feature data: given condition "{condition}" not in {adata.obs["condition"].unique()}')
        adata = adata[adata.obs['condition'] == condition].copy()
    
    if cell_type is not None:
        if cell_type not in adata.obs['cell_type'].unique():
            raise ValueError(f'Error in retrieving feature data: given cell type "{cell_type}" not in {adata.obs["cell_type"].unique()}')
        adata = adata[adata.obs['cell_type'] == cell_type]

    return adata

def write_feature_data(adata, dataset, cell_type, data_type, feature_type='tf_activity', suffix=''):
    import os
    output_dir = f'{FEATURES_DIR}/{feature_type}/{data_type}'
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
    from scipy.sparse import issparse
    from scipy.stats import mannwhitneyu
    from scipy.stats import ttest_rel
    if conditions is None:
        conditions = adata.obs[condition_col].unique()
    dataset = adata.obs['dataset'].unique()[0]
    name_mapping = {} if config is None else config.name_mapping if hasattr(config, 'name_mapping') else {}
    stats_all = []
    def stats_condition_vs_ctr(adata, condition):  

        mask_ctr = adata.obs[condition_col] == ctr_group
        mask_condition = adata.obs[condition_col] == condition
        
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
                stat, pval = ttest_rel(values_case, values_control)
                coef = np.median(values_case) - np.median(values_control)
                
                return {
                    'gene': gene,
                    "p_value": pval,
                    "slope": coef,
                    'ctrl': ctr_group,
                    'condition': condition
                }
            elif test_type == 'mixed-effect':
                from hiara.src.utils.util import test_mixed_effects
                
                obs_ctr = adata.obs.loc[mask_ctr, :]
                obs_ctr['feature_values'] = values_control
                obs_ctr['condition'] = ctr_group
                obs_case = adata.obs.loc[mask_condition, :]
                obs_case['condition'] = condition
                obs_case['feature_values'] = values_case
                
                df = pd.concat([obs_ctr, obs_case])
                
                # Original behavior for non-interaction models
                pval, coef = test_mixed_effects(df, ctr_group, condition, 
                                                target_variable='feature_values', config=config)
    
                if np.isnan(pval):
                    return None
                
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
            'Both age groups': adata.obs.index.notnull()
        }
    
    for age_subset, mask in age_masks.items():
        adata_sub = adata[mask, :].copy()
        assert adata_sub.shape[0]!=0, f'shouldnt be empty'
        for condition in conditions:
            if condition == ctr_group:
                continue
            stats_df = stats_condition_vs_ctr(adata_sub, condition)

            if stats_df is None:
                raise ValueError(f'No stats returned for condition {condition} vs {ctr_group} in dataset {dataset}')

            # For interaction models, apply FDR per coefficient type
            # For regular models, apply FDR globally
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

        assert df_meta.isna().sum().sum() == 0
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

def wrapper_meta_analysis(stats_features, par):
    feature_col = 'gene'
    min_degree = par['meta_analysis_min_cohorts']
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
            nan_sim = stats['p_value_adj'].isna().sum()
            if nan_sim>0:
                raise ValueError(f'NaN p-values found in stats in {cell_type}: {nan_sim} NaNs')
            meta_stats = run_meta_analysis(stats, temp_dir=par['temp_dir'], meta_association_type=meta_association_type, min_degree=min_degree)
            pval_col = 'meta_p_adj'
            meta_stats = compute_trend(meta_stats, pval_col=pval_col, slope_col='slope', col=feature_col)
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

def wrapper_association_with_age_condition(par, association_type, features=None, test_type=None, condition='healthy', config=None):
    """
    Wrapper function to compute association with age and condition.
    
    Now uses configuration objects to eliminate dataset-specific if-else chains.
    
    Parameters
    ----------
    par : dict
        Parameters containing datasets, feature_type, type, cell_types, etc.
    features : list, optional
        List of features to analyze
    condition : str
        Condition filter for loading data
    config : ConditionConfig, optional
        Configuration object (will auto-load if None)
    """
    
    datasets = par['datasets']
    feature_type = par['feature_type']
    data_type = par['data_type']
    cell_types = par['cell_types']
    only_promotor_based = par.get('only_promotor_based', False)
    suffix = '_promotor' if only_promotor_based else ''

    print(f'Association {feature_type} with condition...')
    if 'minor' in data_type:
        cell_types_l = minor_cell_types
    else:
        cell_types_l = cell_types
    
    stats_store = []
    for cell_type in tqdm(cell_types_l, desc='cell types'):
        for dataset in datasets:            
            # Load data
            adata = retrieve_feature_data(
                dataset=dataset, 
                cell_type=cell_type, 
                data_type=data_type, 
                feature_type=feature_type, 
                condition=condition,
                suffix=suffix
            )
            adata = adata[:, adata.var_names.isin(features)] if features is not None else adata
            
            # Filter by cell type
            adata_sub = adata[adata.obs['cell_type']==cell_type]
            if adata_sub.shape[0] < 3:
                raise ValueError(f'Not enough samples for {cell_type} in {dataset}, only {adata_sub.shape[0]} samples')
            
            # Subset features
            genes = adata_sub.var_names
            adata_sub = adata_sub[:, adata_sub.var_names.isin(genes)]

            if issparse(adata_sub.X):
                adata_sub.X = adata_sub.X.toarray()
            
            # Determine statistics based on configuration
            if association_type == 'continous':
                print('Aging analysis for', cell_type, 'in', dataset)
                stats = association_with_age(adata_sub, association_type='spearman')
                stats['condition'] = condition
            elif association_type == 'grouped':
                # Condition analysis using config
                print('Condition analysis for', cell_type, 'in', dataset)
                stats = _compute_condition_stats_from_config(
                    adata_sub, 
                    config, 
                    test_type=test_type or config.test_type
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
    
    if feature_type == 'gene_expression':
        stats_all.rename(columns={'gene': 'target'}, inplace=True)

    return stats_all


def _compute_condition_stats_from_config(adata, config, test_type):
    """
    Compute condition statistics using configuration object.
    
    """
    
    # Auto-detect condition column for datasets with variants
    condition_col = config.condition_column

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
        stats['comparison'] = f'{treatment} vs {control}'
        stats['comparison'] = stats['comparison'].map(lambda name: config.name_mapping.get(name, name))

        # print('\n Control used for', treatment, 'is', control)
        # print('\n', 'Total', stats.shape, ' sig:',stats[stats['p_value']<0.05].shape)
        # aaa
        
        stats_list.append(stats)
    stats = pd.concat(stats_list)

    return stats

def wrapper_tf_activity(par):
    print('Loading data...')
    data_type = par['data_type']
    cell_types = par['cell_types']
    datasets = par['datasets']
    condition = par.get('condition', None)
    only_promotor_based = par.get('only_promotor_based', False)
    print('Calculating TF activity...')
    print(f'  - Promotor-based only: {only_promotor_based}')
    for dataset in datasets:
        print(dataset, data_type)
        adata = retrieve_adata(dataset=dataset, data_type=data_type, condition=condition)

        for cell_type in tqdm(cell_types, desc='cell types'):
            adata_t = adata[adata.obs['cell_type']==cell_type]
            if par['use_consensus_net']:
                net = retrieve_net_consensus(cell_type=cell_type, only_promotor_based=only_promotor_based)
            else:
                net = retrieve_net(dataset=dataset, cell_type=cell_type, only_promotor_based=only_promotor_based)
            if adata_t.shape[0] < 10:
                continue
            tf_acts = calculate_tf_activity(adata_t, net)
            tf_acts.obs['dataset'] = dataset
            tf_acts.uns['dataset'] = dataset
            tf_acts = tf_acts[tf_acts.obs['age'].isna()==False] # there is a bug in the code that causes age to be NaN
            write_feature_data(tf_acts, dataset, cell_type, data_type, suffix='_promotor' if only_promotor_based else '')

def wrapper_gene_score(par):
    from hiara.src.utils.util import get_genesets
    # --------- load data
    cell_types = par['cell_types']
    type = par['type']
    datasets = par['datasets']
    feature_type = par['feature_type']

    # --------- load data
    print('Loading data...')
    adata_dict = {dataset: retrieve_adata(dataset, type) for dataset in datasets}

    pathways = get_genesets()
    
    print('Calculating gene scores...')
    stats_store = []
    for cell_type in tqdm(cell_types, desc='cell types'):
        # ----------- calculate tf activity for all datasets
        net = retrieve_net_consensus(cell_type=cell_type) #TOGO 
        net_genes = net['target'].unique()
        for dataset in datasets:
            adata = adata_dict[dataset][adata_dict[dataset].obs['cell_type']==cell_type]
            
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
            write_feature_data(adata_scores, dataset, cell_type, type, feature_type=feature_type)

# def wrapper_gene_expression(par):
#     # --------- load data
#     cell_types = par['cell_types']
#     type = par['type']
#     datasets = par['datasets']
#     print('Loading data...')
#     adata_dict = {dataset: retrieve_adata(dataset, type) for dataset in datasets}

#     print('Calculating gene expression...')
#     stats_store = []
#     for cell_type in tqdm(cell_types, desc='cell types'):
#         for dataset in datasets:
#             adata = adata_dict[dataset][adata_dict[dataset].obs['cell_type']==cell_type]

#             if type == 'sc':
#                 sc.pp.normalize_total(adata)
#                 sc.pp.log1p(adata)
#             write_feature_data(adata, dataset, cell_type, type, feature_type='gene_expression')

def wrapper_aging_hallmarks(par):
    from hiara.src.config import PRIOR_DIR
    # --------- load data
    cell_types = par['cell_types']
    type = par['type']
    datasets = par['datasets']
    print('Loading data...')
    # Load aging hallmark genes
    aging_hallmark_genes_path = f'{PRIOR_DIR}/aging_hallmark_genes.csv'
    if not os.path.exists(aging_hallmark_genes_path):
        raise FileNotFoundError(f'Aging hallmark genes file not found at {aging_hallmark_genes_path}')
    
    aging_hallmark_genes_df = pd.read_csv(aging_hallmark_genes_path)
    aging_hallmark_genes = aging_hallmark_genes_df['gene'].tolist()
    
    print(f'Loaded {len(aging_hallmark_genes)} aging hallmark genes')

    adata_dict = {dataset: retrieve_adata(dataset, type) for dataset in datasets}

    print('Calculating aging hallmark gene expression...')
    
    for cell_type in tqdm(cell_types, desc='cell types'):
        for dataset in datasets:
            adata = adata_dict[dataset][adata_dict[dataset].obs['cell_type']==cell_type]
            # Filter to only aging hallmark genes
            available_genes = [g for g in aging_hallmark_genes if g in adata.var_names]
            if len(available_genes) == 0:
                raise ValueError(f'Warning: No aging hallmark genes found in dataset {dataset}, cell type {cell_type}')
            
            adata = adata[:, available_genes].copy()
            if type == 'sc':
                sc.pp.normalize_total(adata)
                sc.pp.log1p(adata)
            
            
            write_feature_data(adata, dataset, cell_type, type, feature_type='aging_hallmarks')

def determine_std(adata):
    # Ensure .X is dense
    if isinstance(adata.X, np.ndarray):
        X_dense = adata.X
    elif hasattr(adata.X, "todense"):
        X_dense = adata.X.todense().A
    else:
        raise TypeError("Unexpected type for adata.X: {}".format(data_type(adata.X)))

    # Create DataFrame
    df = pd.DataFrame(X_dense, index=adata.obs.index, columns=adata.var_names)
    genes = df.columns

    # Add metadata
    df["age"] = adata.obs["age"].astype(str)
    df["donor_id"] = adata.obs["donor_id"].astype(str)
    df["cell_type"] = adata.obs["cell_type"].astype(str)

    # Compute standard deviation
    std_df = df.groupby(['cell_type', "donor_id", 'age']).std()
    std_adata = sc.AnnData(std_df.values)

    # Reconstruct obs
    std_adata.obs = std_df.reset_index()[['cell_type', "donor_id", 'age']]

    # Prepare metadata from adata.obs with only unique mappings
    group_cols = ['cell_type', 'donor_id', 'age']
    unique_cols = [col for col in adata.obs.columns if col not in group_cols]
    meta_df = adata.obs.copy()
    meta_df[group_cols] = meta_df[group_cols].astype(str)  # Ensure dtype consistency
    meta_df = meta_df[group_cols + unique_cols].drop_duplicates(subset=group_cols)

    std_adata.obs[group_cols] = std_adata.obs[group_cols].astype(str)  # Also convert here
    std_adata.obs = std_adata.obs.merge(meta_df, on=group_cols, how='left')

    # Final cleanup
    std_adata.obs['age'] = pd.to_numeric(std_adata.obs['age'], errors='coerce')
    std_adata.var = pd.DataFrame(index=genes)
    std_adata.X = np.nan_to_num(std_adata.X, nan=0)

    return std_adata

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
            if p_value is None or np.isnan(p_value) or p_value <= 0 or p_value > 1:
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
            'slope': float(slope)
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

def compute_trend(df, pval_col='meta_p_adj', slope_col='slope', col='gene'):
    # Compute -log10(p_value_adj) for dot size
    df["neg_log10_adj_pval"] = -np.log10(df[pval_col])
    if 'trend' in df.columns:
        df.drop('trend', inplace=True, axis=1)
    
    # Function to assign trend based on slope sign and min_degree
    def determine_trend(x):
        pos = (x > 0).sum()
        neg = (x < 0).sum()
        total = len(x)
        threshold = total
        
        # print(pos, neg, total, threshold)
        if pos >= (threshold):
            return 'Increase in aging'
        elif neg >= (threshold):
            return 'Decrease in aging'
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

def calculate_tf_activity(adata, net, tf_all=None):    
    # - TFs
    if tf_all is not None:
        net = net[net['source'].isin(tf_all)]

    if True: # run decoupler
        import decoupler as dc

        dc.mt.ulm(adata, net, tmin=5)
        tf_acts_X = adata.obsm['score_ulm']
        var = pd.DataFrame({'source': tf_acts_X.columns})
        var.index = var['source']
        
        tf_acts_adata = ad.AnnData(X=tf_acts_X.values, obs=adata.obs, var=var)

    else: # run my implementation
        n_targets_t = 1
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
        print(tf_acts_adata)
        aaa
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
