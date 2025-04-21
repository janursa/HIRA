import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from scipy.stats import linregress, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import r2_score
import pandas as pd
import scipy
import shap
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
from statsmodels.stats.multitest import multipletests
from pandas.api.types import CategoricalDtype
from ciim.src.common import datasets_e, datasets_a, surrogate_names, datasets_all
from scipy.stats import mannwhitneyu
from tqdm import tqdm
from ciim.src.common import cell_types
from ciim.src.common import mapping_major_2_minor
from scipy.sparse import issparse

def retrieve_stats_features(type, feature_type, cell_type=None, datasets=None, condition=None):
    if feature_type == 'tf_activity':
        save_dir = f'/home/jnourisa/projs/ongoing/ciim/output/tf_activation/'
    elif feature_type == 'gene_expression':
        save_dir = f'/home/jnourisa/projs/ongoing/ciim/output/gene_expression/'
    else:
        raise ValueError('Unknown feature type')
    stats = pd.read_csv(f'{save_dir}/stats_features_{type}.csv')
    
    if cell_type is not None:
        stats = stats[stats['cell_type'] == cell_type]
    
    if datasets is not None:
        stats = stats[stats['dataset'].isin(datasets)]
    
    if condition is not None:
        assert condition in stats['condition'].unique(), f'Given condition "{condition}" not in {stats["condition"].unique()}'
        stats = stats[stats['condition'] == condition]
    return stats

def read_feature_data(dataset, cell_type, type, feature_type='tf_activity'):
    if feature_type == 'tf_activity':
        save_dir = f'/home/jnourisa/projs/ongoing/ciim/output/tf_activation/tf_acts'
    elif feature_type == 'gene_expression':
        save_dir = f'/home/jnourisa/projs/ongoing/ciim/output/gene_expression/gene_expression'
    else:
        raise ValueError('Unknown feature type')
    return ad.read_h5ad(f'{save_dir}/{dataset}_{cell_type}_{type}.h5ad')

def write_feature_data(adata, dataset, cell_type, type, feature_type='tf_activity'):
    if feature_type == 'tf_activity':
        save_dir = f'/home/jnourisa/projs/ongoing/ciim/output/tf_activation/tf_acts'
    elif feature_type == 'gene_expression':
        save_dir = f'/home/jnourisa/projs/ongoing/ciim/output/gene_expression/gene_expression'
    else:
        raise ValueError('Unknown feature type')
    adata.write_h5ad(f'{save_dir}/{dataset}_{cell_type}_{type}.h5ad')
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
def retrieve_valid_stats(type):
    stats_e = retrieve_sig_stats(type, race='european')
    stats_a = retrieve_sig_stats(type, race='asian')
    stats_valid = pd.concat([stats_e, stats_a]).drop_duplicates(subset=['cell_type', 'tf', 'analysis'])

    degree = stats_valid.groupby(['cell_type', 'tf']).size()
    
    tuple_index = degree[degree==2].index
    stats_valid = stats_valid.set_index(['cell_type', 'tf']).loc[tuple_index].reset_index()
    stats_valid = stats_valid.drop_duplicates(subset=['cell_type', 'tf'])[['tf', 'cell_type', 'trend']]
    return stats_valid

def retrieve_sig_stats(type, race='european', filter_inconsistent=True):
    stats_all = pd.read_csv(f'../output/tf_activation/stats_all_{type}.csv')
    
    mask = (stats_all['condition']=='healthy') & (stats_all['meta_p_adj'] < 0.05) 
    if race == 'both':
        pass
    else:
        mask &= (stats_all['analysis'] == race)
    # Filter valid rows
    stats_all = stats_all[
        mask
    ]
    if filter_inconsistent:
        stats_all = stats_all[stats_all['trend'] != 'Inconsistent']
    # stats_all = stats_all[~stats_all['trend'].isna()]
    return stats_all


def retrieve_net(dataset, cell_type):  
    net = pd.read_csv(f"/home/jnourisa/projs/ongoing/ciim/output/grns/{dataset}/net_{cell_type}_all_agegroups_all_batches.csv")
    gene_names = np.loadtxt(f'/vol/projects/jnourisa/prior/gene_names.txt', dtype=str)
    net = net[net['target'].isin(gene_names)]
    # tf_all = np.loadtxt(f"/vol/projects/jnourisa/prior/tf_all.csv", dtype=str)
    # net = net.loc[net['source'].isin(tf_all)]
    return net

def adata_lambda(dataset, type='bulk'): 
    base_path = "/vol/projects/jnourisa/datasets/"
    gene_names = np.loadtxt(f'/vol/projects/jnourisa/prior/gene_names.txt', dtype=str)

    # assert type in ['bulk', 'sc', 'bulk_minor', 'bulk_M', 'bulk_F']
    if (type == 'bulk_M') & (type == 'bulk_F'):
        gender = type.split('_')[1]
        type = type.split('_')[0]
        adata = ad.read_h5ad(f"{base_path}/{dataset}_{type}.h5ad")
        adata.obs['sex'] = adata.obs['sex'].apply(lambda name: {'Male': 'M', 'Female': 'F'}.get(name, name))
        adata = adata[tf_acts.obs['sex']==gender]
    else:
        if 'std' in type:
            type = type.split('_')[0]
        adata = ad.read_h5ad(f"{base_path}/{dataset}_{type}.h5ad")
    adata.obs['dataset'] = dataset
    adata = adata[:, adata.var_names.isin(gene_names)]
    return adata

def determine_consensus_nets(datasets, cell_types, min_degree=5):
    consensus_nets = {}
    for cell_type in cell_types:
        net_store = []
        for dataset in datasets:
            net = retrieve_net(dataset, cell_type)
            net['dataset'] = dataset
            net_store.append(net)
        nets = pd.concat(net_store)

        nets['link'] = nets['source'] + '_' + nets['target']
        degrees = nets.groupby(['link'])['dataset'].size()
        shared_links = degrees[degrees>=min_degree].index
        nets = nets[nets['link'].isin(shared_links)]

        net_mean = nets.groupby(['source', 'target', 'cell_type'])['weight'].mean().reset_index()
        consensus_nets[cell_type] = net_mean
    return consensus_nets
def determine_consensus_net(datasets, cell_type, min_degree=5):
    consensus_nets = {}

    net_store = []
    for dataset in datasets:
        net = retrieve_net(dataset, cell_type)
        net['dataset'] = dataset
        net_store.append(net)
    nets = pd.concat(net_store)

    nets['link'] = nets['source'] + '_' + nets['target']
    degrees = nets.groupby(['link'])['dataset'].size()
    shared_links = degrees[degrees>=min_degree].index
    nets = nets[nets['link'].isin(shared_links)]

    net_mean = nets.groupby(['source', 'target', 'cell_type'])['weight'].mean().reset_index()
    return net_mean

def wrapper_extract_sig_network(par):
    type = par['type']
    min_degree = par['min_degree_network']
    # - extract sig tfs for cell types
    stats_features_valid = retrieve_sig_stats(type).drop_duplicates(subset=['cell_type', 'tf']) 
    sig_tfs_dict = stats_features_valid.groupby(['cell_type'])['tf'].unique().to_dict()

    # - identify consensus networks across datasets with at least n datasets shared
    consensus_nets = determine_consensus_nets(datasets_all, cell_types=sig_tfs_dict.keys(), min_degree=min_degree)

    # - subset stats_target to only those targets that are connected to sig tfs
    stats_target_all = pd.read_csv(f'../output/tf_activation/stats_targets_{type}.csv')
    stats_target_store = []
    for cell_type, sig_tfs in sig_tfs_dict.items():
        stats_target = stats_target_all[(stats_target_all['cell_type'] == cell_type)]
        net = consensus_nets[cell_type]
        targets = net[net['source'].isin(sig_tfs)]['target'].unique()
        stats_target = stats_target[stats_target['target'].isin(targets)]
        stats_target_store.append(stats_target)
    stats_target_all = pd.concat(stats_target_store)
    print('Number of targets across datasets:')
    # print(stats_target_all.groupby(['cell_type', 'dataset']).size().reset_index(name='counts'))

    # - meta analysis to identify consistent targets across datasets
    from ciim.src.tf_activity.meta_analysis.helper import run_meta_analysis
    stats_all_c = stats_target_all.copy()
    stats_all_c.rename(columns={'p_value': 'pvalue', 'target':'gene'}, inplace=True)
    meta_analysis_type = 'fisher'
    df_meta_all = run_meta_analysis(stats_all_c, type=meta_analysis_type, min_degree=min_degree, temp_dir='../output/tmp')
    df_meta_all.rename(columns={'gene': 'target'}, inplace=True)
    stats_target_all = stats_target_all.merge(df_meta_all, on=['target', 'cell_type'], how='left')
    stats_target_all = compute_trend(stats_target_all, pval_col= 'meta_p_adj', slope_col='slope', col='target')
    stats_target_consistent = stats_target_all[stats_target_all['meta_p_adj'] < 0.05]
    stats_target_consistent = stats_target_consistent[stats_target_consistent['trend'] != 'Inconsistent'][['cell_type', 'target', 'meta_p_adj', 'trend']].drop_duplicates()

    print(stats_target_consistent.groupby(['cell_type', 'trend']).size().reset_index(name='counts'))
    sig_targets_dict = stats_target_consistent.groupby(['cell_type'])['target'].unique().to_dict()
    # - subset consensus networks to only those targets that are connected to sig tfs
    sig_nets = []
    for cell_type, sig_tfs in sig_tfs_dict.items():
        net = consensus_nets[cell_type]
        if cell_type not in sig_targets_dict:
            continue
        sig_targets = sig_targets_dict[cell_type]
        net = net[(net['source'].isin(sig_tfs)) & (net['target'].isin(sig_targets))].reset_index(drop=True)

        net = net.merge(stats_features_valid, left_on=['cell_type', 'source'], right_on=['cell_type', 'tf'], how='left')
        net = net.merge(stats_target_consistent, left_on=['cell_type', 'target'], right_on=['cell_type', 'target'], how='left', suffixes=('_tf', '_target'))
        sig_nets.append(net)
    sig_nets = pd.concat(sig_nets)

    sig_nets.to_csv(par['sig_nets'], index=False)

def determine_stats_disease(adata, age_cutoff=50, ctr_group='normal', disease_col='disease'):
    conditions = adata.obs['disease'].unique()
    dataset = adata.obs['dataset'].unique()[0]
    stats_all = []
    name_mapping = {'normal': 'healthy', 'systemic lupus erythematosus': 'SLE'}
    # case 1: association with age in healhty and disease samples
    for group in conditions:
        adata_sub = adata[adata.obs[disease_col] == group]
        stats_df = association_with_age(adata_sub, genes=adata_sub.var_names)
        stats_df['p_value_adj'] = multipletests(stats_df["p_value"], method="fdr_bh")[1]
        stats_df['condition'] = name_mapping[group]
        # stats_df['dataset'] = f"{dataset}_{group}"
        stats_all.append(stats_df)

    # case 2: disease vs healthy 
    def stats_disease_vs_healthy(adata, condition):  
        mask_ctr = adata.obs[disease_col] == ctr_group
        mask_condition = adata.obs[disease_col] == condition
        
        control_group = adata.X[mask_ctr, :]
        case_group = adata.X[mask_condition, :]

        if (np.sum(mask_condition) < 10) or (np.sum(mask_ctr) < 10):
            print('Not enough samples for', condition, ' vs ', ctr_group)
            return None
        results = []
        for i, gene in enumerate(adata.var_names):
            values_case = case_group[:, i]
            values_control = control_group[:, i]

            if np.sum(values_case) == 0 and np.sum(values_control) == 0:
                continue  

            if issparse(values_case):
                values_case = values_case.todense().A.flatten()
            if issparse(values_control):
                values_control = values_control.todense().A.flatten()

            stat, pval = mannwhitneyu(values_case, values_control, alternative="two-sided")
            # direction = "Increase in disease" if np.median(values_case) > np.median(values_control) else "Decrease in disease"

            results.append({
                "tf": gene,
                "p_value": pval,
                "slope_disease":  np.median(values_case) - np.median(values_control) ,
                "values_case": values_case,
                "values_control": values_control,
                'condition': name_mapping[condition]
            }) 
        results = pd.DataFrame(results)
        return results
    for condition in conditions:
        if condition == ctr_group:
            continue
        stats_df = stats_disease_vs_healthy(adata, condition)
        if stats_df is None:
            continue
        stats_df['p_value_adj'] = multipletests(stats_df["p_value"], method="fdr_bh")[1]
        # stats_df['dataset'] = f"{dataset}_{condition}"
        stats_all.append(stats_df)
    # Combine all stats
    stats_df = pd.concat(stats_all, ignore_index=True)
    stats_df['dataset'] = dataset

    return stats_df

def wrapper_run_meta_analysis(par):
    from ciim.src.tf_activity.meta_analysis.helper import run_meta_analysis
    stats_features = pd.read_csv(par['stats_features'])
    
    if 'tf' in stats_features.columns:
        feature_col = 'tf'
    elif 'target' in stats_features.columns:
        feature_col = 'target'
    else:
        print(stats_features)
        raise ValueError('Unknown feature column')
    cell_types = stats_features['cell_type'].unique()
    # - european 
    print('european ...')
    stats_store = []
    for cell_type in stats_features['cell_type'].unique():
        stats = stats_features[(stats_features['cell_type'] == cell_type) & (stats_features['dataset'].isin(datasets_e) & (stats_features['condition']=='healthy'))]
        if len(stats) == 0:
            print('No stats for', cell_type, ' skipping it')
            continue
        if stats.groupby(feature_col).size().max()<len(datasets_e):
            print('Not enough mutual TFs for ', cell_type, ' skipping it')
            continue
        min_degree = len(datasets_e) - 1
        meta_stats = run_meta_analysis(stats, temp_dir=par['temp_dir'], min_degree=min_degree, meta_analysis_type='fisher')
        pval_col = 'meta_p_adj'
        meta_stats = compute_trend(meta_stats, pval_col=pval_col, slope_col='slope', col=feature_col)
        stats_store.append(meta_stats)
    stats_discovery = pd.concat(stats_store)
    stats_discovery['condition'] = 'healthy'
    stats_discovery['analysis'] = 'european'
    # - validation
    print('Asian ...')
    stats_store = []
    for cell_type in stats_features['cell_type'].unique():
        stats = stats_features[(stats_features['cell_type'] == cell_type) & (stats_features['dataset'].isin(datasets_a) & (stats_features['condition']=='healthy'))]

        if len(stats) == 0:
            print('No stats for', cell_type, ' skipping it')
            continue
        if stats.groupby(feature_col).size().max()<len(datasets_a):
            print('Not enough mutual TFs for ', cell_type, ' skipping it')
            continue
        min_degree = len(datasets_a)
        meta_stats = run_meta_analysis(stats, temp_dir=par['temp_dir'], min_degree=min_degree, meta_analysis_type='fisher')
        pval_col = 'meta_p_adj'
        meta_stats = compute_trend(meta_stats, pval_col=pval_col, slope_col='slope', col=feature_col)
        
        stats_store.append(meta_stats)
    stats_validation = pd.concat(stats_store)
    stats_validation['condition'] = 'healthy'
    stats_validation['analysis'] = 'asian'

    # - combine
    stats_combined = pd.concat([stats_discovery, stats_validation])

    #- save
    print('Saving results to ', par['stats_all'])
    stats_combined.to_csv(par['stats_all'], index=False)

def wrapper_association_with_age(par, cell_types, datasets, feature_type='tf_activity'):
    # - calculate tf activity for all datasets
    print(f'Association {feature_type} with age/disease...')
    stats_store = []
    for cell_type in tqdm(cell_types, desc='cell types'):
        # ----------- calculate tf activity for all datasets
        for dataset in datasets:
            adata = read_feature_data(dataset, cell_type, par['type'], feature_type=feature_type)
            if 'disease' in adata.obs.columns:
                disease_flag = True
            else:
                disease_flag = False

            # - add which cell type resolution to run the analysis
            adata.obs['cell_type_resolution'] = adata.obs[par['cell_type_resolution']]
            
            for cell_type_resolution in adata.obs['cell_type_resolution'].unique():
                adata_sub = adata[adata.obs['cell_type_resolution']==cell_type_resolution]
                if adata_sub.shape[0] < 10:
                    print('Not enough samples for', cell_type, dataset, cell_type_resolution)
                    continue
                
                # - subset based on prior (only for target genes) -> add this to meta analysis
                if feature_type == 'gene_expression':
                    net = retrieve_net(dataset, cell_type)
                    targets = net['target'].unique()
                    features = targets
                else:
                    features = adata_sub.var_names

                if disease_flag:
                    if issparse(adata_sub.X):
                        adata_sub.X = adata_sub.X.toarray()
                    stats = determine_stats_disease(adata_sub)
                else:
                    stats = association_with_age(adata_sub, genes=features, association_type=par['association_type'])

                    stats['condition'] = 'healthy'
                    stats['dataset'] = dataset
                
                stats['cell_type'] = cell_type_resolution
                
                stats_store.append(stats)
        
    stats_all = pd.concat(stats_store)
    if feature_type == 'gene_expression':
        stats_all.rename(columns={'tf': 'target'}, inplace=True)
    return stats_all
# def wrapper_target_association_age(par):
#     stats_store = []
#     for dataset in tqdm(datasets_all, desc='datasets'):
#         adata = adata_lambda(dataset, type=par['type'])
#         if par['type'] == 'sc_std':
#             adata = determine_std(adata)
#             adata.write_h5ad(f"{par['std_dir']}/std_{dataset}.h5ad")
#         for cell_type in tqdm(cell_types, desc='cell types'):
#             net = retrieve_net(dataset, cell_type)
#             targets = net['target'].unique()
#             adata_t = adata[adata.obs['cell_type']==cell_type].copy()
            
#             stats = association_with_age(adata_t, targets, association_type=par['association_type'])
#             stats.rename(columns={'tf': 'target'}, inplace=True) 
#             stats['dataset'] = dataset
#             stats['cell_type'] = cell_type
#             stats_store.append(stats)
#     stats_target_all = pd.concat(stats_store)
#     stats_target_all.to_csv(par['stats_targets'], index=False)


def wrapper_tf_activity(cell_types, datasets, type='bulk'):
    # --------- load data
    print('Loading data...')
    adata_dict = {dataset: adata_lambda(dataset, type) for dataset in datasets}
    tf_all = np.loadtxt(f"/vol/projects/jnourisa/prior/tf_all.csv", dtype=str)

    # - calculate tf activity for all datasets
    print('Calculating TF activity...')
    stats_store = []
    for cell_type in tqdm(cell_types, desc='cell types'):
        # ----------- calculate tf activity for all datasets
        for dataset in datasets:
            adata = adata_dict[dataset][adata_dict[dataset].obs['cell_type']==cell_type]
            net = retrieve_net(dataset, cell_type)
            net = net[net['source'].isin(tf_all)]

            if adata.shape[0] < 10:
                continue
            tf_acts = calculate_tf_activity(adata, net)
            tf_acts.obs['dataset'] = dataset
            tf_acts.uns['dataset'] = dataset
            
            tf_acts = tf_acts[tf_acts.obs['age'].isna()==False] # there is a bug in the code that causes age to be NaN
            # print(cell_type, tf_acts.shape)
            if 'std' in type:
                save_type = type.split('_')[0]
            else:
                save_type = type
            write_feature_data(tf_acts, dataset, cell_type, save_type)

            if type == 'sc_std':
                tf_acts = determine_std(tf_acts)
                write_feature_data(tf_acts, dataset, cell_type, type)
def wrapper_gene_expression(cell_types, datasets, type='bulk'):
    # --------- load data
    print('Loading data...')
    adata_dict = {dataset: adata_lambda(dataset, type) for dataset in datasets}

    print('Calculating gene expression...')
    stats_store = []
    for cell_type in tqdm(cell_types, desc='cell types'):
        # ----------- calculate tf activity for all datasets
        for dataset in datasets:
            adata = adata_dict[dataset][adata_dict[dataset].obs['cell_type']==cell_type]
            
            if 'std' in type:
                save_type = type.split('_')[0]
            else:
                save_type = type
            write_feature_data(adata, dataset, cell_type, save_type, feature_type='gene_expression')

            if type == 'sc_std':
                tf_acts = determine_std(tf_acts, feature_type='gene_expression')
                write_feature_data(tf_acts, dataset, cell_type, type)
def determine_std(adata):
    # Ensure .X is dense
    if isinstance(adata.X, np.ndarray):
        X_dense = adata.X
    elif hasattr(adata.X, "todense"):
        X_dense = adata.X.todense().A
    else:
        raise TypeError("Unexpected type for adata.X: {}".format(type(adata.X)))

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

def association_with_age(adata, genes, gene_col='tf', association_type='linear'):
    '''
    Calculate p-values for the linear regression of the top tfs across datasets with ageing,
    and apply FDR correction (Benjamini-Hochberg).
    '''
    p_value_store = []

    for gene in genes:
        if gene not in adata.var_names:
            continue
        mask_gene = adata.var_names == gene
        adata_sub = adata[:, mask_gene]

        df = adata_sub.to_df()
        df = df.merge(adata_sub.obs[['age']], left_index=True, right_index=True)

        df.sort_values('age', inplace=True)

        ages = df['age'].values
        expression = df[gene].values

        # Fit linear regression
        if len(ages) > 1:
            if association_type == 'linear':
                # Perform linear regression
                slope, intercept, r_value, p_value, _ = linregress(ages, expression)
            elif association_type == 'spearman':
                slope, p_value = spearmanr(ages, expression)
            
            p_value_store.append({
                gene_col: gene, 
                'p_value': p_value,
                'slope': slope
            })

    stats_df = pd.DataFrame(p_value_store)

    # # Apply FDR correction to p-values
    if not stats_df.empty:
        stats_df['p_value_adj'] = multipletests(stats_df['p_value'], method='fdr_bh')[1]

    return stats_df

def compute_trend(df, pval_col='meta_p_adj', slope_col='slope', col='tf'):
    # Compute -log10(p_value_adj) for dot size
    df["neg_log10_adj_pval"] = -np.log10(df[pval_col])
    if 'trend' in df.columns:
        df.drop('trend', inplace=True, axis=1)
    # Determine color based on slope sign
    def determine_trend(x):
        if (x > 0).all():
            return 'Increase in aging'
        elif (x < 0).all():
            return 'Decrease in aging'
        else:
            return 'Inconsistent'

    # Apply the function group-wise
    trend = df.groupby([col, 'cell_type'])[slope_col].apply(determine_trend).reset_index(name='trend')
    trend = trend.dropna()
 
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

    if False: # run decoupler
        mat = pd.DataFrame(
            data=adata.X.todense(),  
            columns=adata.var_names,  
            index=adata.obs.index  
        )

        tf_acts, tf_pvals = decoupler.run_ulm(mat, net, source='source', target='target', weight='weight', use_raw=False)
        # - formatize
        tf_acts = tf_acts.reset_index().melt(id_vars='index', var_name='source', value_name='activity')
        # cols = [c for c in adata.obs.columns if c not in ['sample', 'cell_type', 'cell_count']]
     
        # obs = adata.obs[cols]
        obs = adata.obs.copy()

        obs = obs.reset_index()
        
        tf_acts['index'] = tf_acts['index'].astype(str)
        obs['index'] = obs['index'].astype(str)
        tf_acts = tf_acts.merge(obs, on='index', how='left').drop('index', axis=1)
        assert tf_acts.shape[0]==tf_acts.shape[0]
    else: # run my implementation
        tf_acts = tf_activity_local(net, adata, tf_all)
    
    if 'index' in tf_acts.columns:
        tf_acts = tf_acts.drop('index', axis=1)
    
    assert tf_acts.shape[0] != 0, 'Empty'
    assert 'sample' in tf_acts.columns, 'sample not in tf_acts' 
    if False:
        tf_acts['sample'] = tf_acts['sample'].astype(int)
    tf_acts['age'] = pd.to_numeric(tf_acts['age'], errors='coerce')

    # - create anndata
    X_df = tf_acts.pivot(index='sample', columns='source', values='activity')
    obs_df = tf_acts.drop_duplicates(subset='sample').set_index('sample')[[c for c in tf_acts.columns if c not in ['source', 'activity', 'sample']]]
    obs_df = obs_df.loc[X_df.index]
    var_df = pd.DataFrame(index=X_df.columns)
    var_df['source'] = var_df.index
    tf_acts_adata = ad.AnnData(X=X_df.values, obs=obs_df, var=var_df)
    return tf_acts_adata

def pathway_analysis_wrapper(df):
    import gseapy as gp
    from gseapy import barplot, dotplot

    res2d_store = []
    for cell_type in df['cell_type'].unique():
    # for cell_type in ['MONO']:
        for trend in df['trend'].unique():
        # for trend in ['Decrease in aging']:
            mask = (df['cell_type'] == cell_type) & (df['trend'] == trend)
            if mask.sum() == 0:
                continue
            stats_df = df[mask]
            # - prepare
            stats_df = stats_df[['tf', 'meta_p_adj']]
            stats_df = stats_df[~stats_df.duplicated()].reset_index(drop=True)
            # ranked_genes = stats_df.set_index('tf')['meta_p_adj'].sort_values(ascending=True)

            # - EA
            genes = stats_df['tf'].unique()
            # print(genes)
            # genes = np.random.choice(df['tf'].unique(), len(genes))

            print(f"cell_type: {cell_type}, trend: {trend}, n tfs: {len(genes)}")
            np.savetxt('../output/test.csv', genes, fmt='%s', delimiter=',')
            rr = gp.enrichr(gene_list='../output/test.csv',
                            gene_sets=['MSigDB_Hallmark_2020'], #, 'KEGG_2021_Human'
                            organism='human', 
                            outdir=None, 
                            # background=tf_all,
                            )
            res2d = rr.res2d
            res2d = res2d[res2d['Adjusted P-value']<0.05]
            print(res2d.shape)
            res2d['cell_type'] = cell_type
            res2d['trend'] = trend
            res2d_store.append(res2d)
    if len(res2d_store) == 0:
        return None
    
    res2d_all = pd.concat(res2d_store)
    return res2d_all
