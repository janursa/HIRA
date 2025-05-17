import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from scipy.stats import linregress, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import r2_score
import pandas as pd
import os
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
from ciim.src.common import mapping_major_2_minor, mapping_minor_2_major
from scipy.sparse import issparse

def retrieve_stats_features(type, feature_type, race=None, cell_type=None, datasets=None, condition=None):
    if feature_type == 'tf_activity':
        save_dir = f'/home/jnourisa/projs/ongoing/ciim/output/tf_activity/'
    elif feature_type == 'gene_expression':
        save_dir = f'/home/jnourisa/projs/ongoing/ciim/output/gene_expression/'
    else:
        raise ValueError('Unknown feature type')
    stats = pd.read_csv(f'{save_dir}/stats_features_{type}.csv')
    if cell_type is not None: 
        if cell_type not in stats['cell_type'].unique():
            raise ValueError(f'Given cell type "{cell_type}" not in {stats["cell_type"].unique()}')
        stats = stats[stats['cell_type'] == cell_type]
    
    if datasets is not None:
        stats = stats[stats['dataset'].isin(datasets)]
    
    if condition is not None:
        assert condition in stats['condition'].unique(), f'Given condition "{condition}" not in {stats["condition"].unique()}'
        stats = stats[stats['condition'] == condition]
    if race is not None:
        if race == 'european':
            datasets = datasets_e
        else:
            datasets = datasets_a
        stats = stats[stats['dataset'].isin(datasets)]
    return stats

def retrieve_feature_data(dataset, cell_type, type, feature_type='tf_activity'):
    if feature_type == 'tf_activity':
        save_dir = f'/home/jnourisa/projs/ongoing/ciim/output/tf_activity/tf_acts'
    elif feature_type == 'gene_expression':
        save_dir = f'/home/jnourisa/projs/ongoing/ciim/output/gene_expression/gene_expression'
    else:
        raise ValueError('Unknown feature type')
    cell_type_major = mapping_minor_2_major.get(cell_type, cell_type)
    file_path = f'{save_dir}/{dataset}_{cell_type_major}_{type}.h5ad'
    if os.path.exists(file_path) == False:
        raise ValueError(f'File {file_path} does not exist')
    adata = ad.read_h5ad(file_path)
    
    return adata

def write_feature_data(adata, dataset, cell_type, type, feature_type='tf_activity'):
    if feature_type == 'tf_activity':
        save_dir = f'/home/jnourisa/projs/ongoing/ciim/output/tf_activity/tf_acts'
    elif feature_type == 'gene_expression':
        save_dir = f'/home/jnourisa/projs/ongoing/ciim/output/gene_expression/gene_expression'
    else:
        raise ValueError('Unknown feature type')
    adata.write_h5ad(f'{save_dir}/{dataset}_{cell_type}_{type}.h5ad')

def retrieve_valid_stats(type):
    stats_e = retrieve_sig_stats(type, race='european')
    stats_a = retrieve_sig_stats(type, race='asian')
    stats_valid = pd.concat([stats_e, stats_a]).drop_duplicates(subset=['cell_type', 'tf', 'race'])

    degree = stats_valid.groupby(['cell_type', 'tf']).size()
    
    tuple_index = degree[degree==2].index
    stats_valid = stats_valid.set_index(['cell_type', 'tf']).loc[tuple_index].reset_index()
    stats_valid = stats_valid.drop_duplicates(subset=['cell_type', 'tf'])[['tf', 'cell_type', 'trend']]
    return stats_valid

def retrieve_sig_stats(type, feature_type='tf_activity',race='european', filter_inconsistent=True, cell_type=None):
    stats_all = pd.read_csv(f'../output/{feature_type}/stats_all_{type}.csv')
    
    mask = (stats_all['condition']=='healthy') & (stats_all['meta_p_adj'] < 0.05) 

    mask &= (stats_all['race'] == race)
    # Filter valid rows
    stats_all = stats_all[
        mask
    ]
    if filter_inconsistent:
        stats_all = stats_all[stats_all['trend'] != 'Inconsistent']
    # stats_all = stats_all[~stats_all['trend'].isna()]
    if cell_type is not None:
        if cell_type not in stats_all['cell_type'].unique():
            raise ValueError(f'Given cell type "{cell_type}" not in {stats_all["cell_type"].unique()}')
        stats_all = stats_all[stats_all['cell_type'] == cell_type]
    return stats_all

def retrieve_net(dataset, cell_type, only_promotor_based=False, c_t=5):  
    cell_type_major = mapping_minor_2_major.get(cell_type, cell_type)
    assert cell_type_major in ['CD4T', 'CD8T', 'NK', 'B', 'MONO'], f'Unknown cell type {cell_type_major}'
    if dataset == '!CXCL9':
        net = get_consensus_net(datasets_e, cell_type_major, min_degree=2)
    else:
        net = pd.read_csv(f"/home/jnourisa/projs/ongoing/ciim/output/grns/{dataset}/net_{cell_type_major}_all_agegroups_all_batches.csv")
    gene_names = np.loadtxt(f'/vol/projects/jnourisa/prior/gene_names.txt', dtype=str)
    net = net[net['target'].isin(gene_names)]
    # tf_all = np.loadtxt(f"/vol/projects/jnourisa/prior/tf_all.csv", dtype=str)
    if False:
        skeleton = pd.read_csv(f'/vol/projects/jnourisa/prior/skeleton_promotor.csv')
        net['edge'] = net['source'] + '_' + net['target']
        net = net[net['edge'].isin(skeleton['edge'])]
        net = net.drop('edge', axis=1)
    if only_promotor_based:
        net = net[net['promotor_based']]
    
    centrality_df = net.groupby(['source']).size()
    tfs = centrality_df[centrality_df>c_t].index
    net = net[net['source'].isin(tfs)]
    return net[['source', 'target', 'weight', 'cell_type']]

def retrieve_nets(datasets, cell_type, only_promotor_based=False):
    net_store = []
    for dataset in datasets:
        net = retrieve_net(dataset=dataset, cell_type=cell_type, only_promotor_based=only_promotor_based)
        net['dataset'] = dataset
        net_store.append(net)
    nets = pd.concat(net_store, ignore_index=True)
    return nets

def retrieve_sig_net(type, race, cell_type=None):
    df = pd.read_csv(f'/home/jnourisa/projs/ongoing/ciim/output/sig_nets/sig_nets_{type}_{race}.csv')
    if cell_type is not None:
        df = df[df['cell_type'] == cell_type]
    return df

def determine_sig_network(type, race, min_degree=3):
    os.makedirs(f'/home/jnourisa/projs/ongoing/ciim/output/sig_nets', exist_ok=True)
    from ciim.src.tf_activity.helper import retrieve_nets
    stats_tfs = retrieve_sig_stats(type, feature_type='tf_activity')
    stats_targets = retrieve_sig_stats(type, feature_type='gene_expression')

    if race == 'european':
        datasets = datasets_e
    elif race == 'asian':
        datasets = datasets_a
    elif race == 'both':
        datasets = datasets_all
    else:
        raise ValueError('')

    nets_stats_store = []
    for cell_type in cell_types:
        stats_tfs_t = stats_tfs[stats_tfs['cell_type'] == cell_type].drop_duplicates(subset=['cell_type', 'tf'])[['tf', 'meta_p_adj', 'slope', 'trend']]
        stats_targets_t = stats_targets[stats_targets['cell_type'] == cell_type].drop_duplicates(subset=['cell_type', 'target'])[['target', 'meta_p_adj', 'slope', 'trend']]
        
        if len(stats_tfs_t) == 0:
            print('No source for', cell_type, ' skipping it')
            continue
        if len(stats_targets_t) == 0:
            print('No target for', cell_type, ' skipping it')
            continue
        
        # - get the nets
        net = get_consensus_nets(datasets, [cell_type], min_degree=min_degree)[cell_type]
        sig_tfs = stats_tfs_t['tf'].unique()
        sig_targets = stats_targets_t['target'].unique()
        net = net[(net['source'].isin(sig_tfs)) & (net['target'].isin(sig_targets))]
        net = net.groupby(['source', 'target'])['weight'].mean().reset_index() # probably not necessary
        # - get the stats
        nets_stats = pd.merge(net, stats_tfs_t, left_on='source', right_on='tf', how='left')
        nets_stats = pd.merge(nets_stats, stats_targets_t, left_on='target', right_on='target', how='left', suffixes=('_source', '_target'))
        nets_stats = nets_stats[['source', 'target', 'weight', 'slope_source', 'slope_target', 'meta_p_adj_source', 'meta_p_adj_target', 'trend_source', 'trend_target']]
        nets_stats['cell_type'] = cell_type
        nets_stats['race'] = race
        nets_stats_store.append(nets_stats)
    nets_stats = pd.concat(nets_stats_store)
    nets_stats.to_csv(f'/home/jnourisa/projs/ongoing/ciim/output/sig_nets/sig_nets_{type}_{race}.csv')


def retrieve_adata_bulk(dataset, type='bulk', cell_type=None): 
    base_path = "/vol/projects/jnourisa/datasets/"
    gene_names = np.loadtxt(f'/vol/projects/jnourisa/prior/gene_names.txt', dtype=str)

    # assert type in ['bulk', 'sc', 'bulk_minor', 'bulk_M', 'bulk_F']
    if (type == 'bulk_M') & (type == 'bulk_F'):
        gender = type.split('_')[1]
        type = type.split('_')[0]
        adata = ad.read_h5ad(f"{base_path}/{dataset}_{type}.h5ad")
        adata.obs['sex'] = adata.obs['sex'].apply(lambda name: {'Male': 'M', 'Female': 'F'}.get(name, name))
        adata = adata[adata.obs['sex']==gender]
    else:
        if 'std' in type:
            type = type.split('_')[0]
        adata = ad.read_h5ad(f"{base_path}/{dataset}_{type}.h5ad")
    adata.obs['dataset'] = dataset
    adata = adata[:, adata.var_names.isin(gene_names)]

    if cell_type is not None:
        if cell_type not in adata.obs['cell_type'].unique():
            raise ValueError(f'Given cell type "{cell_type}" not in {adata.obs["cell_type"].unique()}')
        adata = adata[adata.obs['cell_type'] == cell_type]
    adata = adata[~adata.obs['age'].isna()].copy()
    adata.obs['age'] = adata.obs['age'].astype(int)
    adata.obs['sex'] = adata.obs['sex'].apply(lambda name: {'F': 'Female', 'M':'Male'}.get(name, name))

    return adata


    
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
def get_consensus_nets(datasets, cell_types, min_degree=5):
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
def get_consensus_net(datasets, cell_type, min_degree=5):
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
def determine_stats_condition(adata, ctr_group='normal', condition_col='disease', test_type='unpaired', conditions=None, name_mapping = {'normal': 'healthy', 'systemic lupus erythematosus': 'SLE'}):
    from scipy.stats import wilcoxon
    from scipy.sparse import issparse
    from scipy.stats import mannwhitneyu
    from scipy.stats import ttest_rel
    import statsmodels.formula.api as smf


    if conditions is None:
        conditions = adata.obs[condition_col].unique()
    dataset = adata.obs['dataset'].unique()[0]
    
    stats_all = []
    if 'SLE' in dataset:
        # case 1: association with age in healhty and disease samples
        for group in conditions:
            adata_sub = adata[adata.obs[condition_col] == group]
            stats_df = association_with_age(adata_sub)
            
            stats_df['p_value_adj'] = multipletests(stats_df["p_value"], method="fdr_bh")[1]
            stats_df['condition'] = name_mapping.get(group, group)
            # stats_df['dataset'] = f"{dataset}_{group}"
            stats_all.append(stats_df)
    
    # case 2: condition vs ctrl 
    def stats_condition_vs_ctr(adata, condition):  
        mask_ctr = adata.obs[condition_col] == ctr_group
        mask_condition = adata.obs[condition_col] == condition
        
        control_group = adata.X[mask_ctr, :]
        case_group = adata.X[mask_condition, :]

        if (np.sum(mask_condition) < 5) or (np.sum(mask_ctr) < 5):
            print('Not enough samples for', condition, ' vs ', ctr_group)
            return None
        results = []
        # adata = adata[:, adata.var_names.isin(['FOXO1', 'FOXO3', 'FOXO4', 'NFE2L2', 'TP53', 'SIRT1', 'HIF1A'])] #TODO: remove this
        for i, gene in enumerate(adata.var_names):
            values_case = case_group[:, i]
            values_control = control_group[:, i]

            if np.sum(values_case) == 0 and np.sum(values_control) == 0:
                continue  

            if issparse(values_case):
                values_case = values_case.todense().A.flatten()
            if issparse(values_control):
                values_control = values_control.todense().A.flatten()
            
            if test_type == 'unpaired':
                stat, pval = mannwhitneyu(values_case, values_control, alternative="two-sided")
                coef = np.median(values_case) - np.median(values_control)
            elif test_type == 'paired':
                stat, pval = ttest_rel(values_case, values_control)
                # stat, pval = wilcoxon(values_case, values_control, alternative="two-sided")
                coef = np.median(values_case) - np.median(values_control)
            elif test_type == 'mixed-effect':
                import warnings
                warnings.filterwarnings("ignore")
                donors_ctr = adata.obs.loc[mask_ctr, 'donor_id']
                donors_case = adata.obs.loc[mask_condition, 'donor_id']
                df = pd.DataFrame({
                    "G": np.concatenate([values_control, values_case]), 'donor_id': np.concatenate([donors_ctr, donors_case]), 'condition': [ctr_group]*len(donors_ctr) + [condition]*len(donors_case)
                    })
                df['condition'] = pd.Categorical(df['condition'], categories=[ctr_group, condition], ordered=True)
                model = smf.mixedlm("G ~ condition", df, groups=df["donor_id"])
                try:
                    result = model.fit()
                    # print(gene,result.summary())
                    coef = result.params[f'condition[T.{condition}]']
                    pval = result.pvalues[f'condition[T.{condition}]']
                except Exception as e:
                    coef = np.nan
                    pval = np.nan
                

                values_case = None
                values_control = None
            else:
                raise ValueError('Unknown test type')

            results.append({
                "tf": gene,
                "p_value": pval,
                "slope_condition":  coef ,
                'ctrl': ctr_group,
                'condition': name_mapping.get(condition, condition)
            }) 
        results = pd.DataFrame(results)
        return results
    for condition in conditions:
        if condition == ctr_group:
            continue
        stats_df = stats_condition_vs_ctr(adata, condition)
        if stats_df is None:
            continue
        stats_df['p_value_adj'] = multipletests(stats_df["p_value"], method="fdr_bh")[1]
        stats_all.append(stats_df)
    # Combine all stats
    stats_df = pd.concat(stats_all, ignore_index=True)
    stats_df['dataset'] = dataset

    return stats_df

def wrapper_meta_analysis(par):
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
    # - 
    def run_func(datasets, min_degree, meta_analysis_type):
        stats_store = []
        for cell_type in stats_features['cell_type'].unique():
            stats = stats_features[(stats_features['cell_type'] == cell_type) & (stats_features['dataset'].isin(datasets) & (stats_features['condition']=='healthy'))]
            if len(stats) == 0:
                print('No stats for', cell_type, ' skipping it')
                continue
            if stats.groupby(feature_col).size().max()<min_degree:
                print('Not enough mutual TFs for ', cell_type, ' skipping it')
                continue
            # - keep only min_degree info that is consistent
            meta_stats = run_meta_analysis(stats, temp_dir=par['temp_dir'], meta_analysis_type=meta_analysis_type, min_degree=min_degree)
            pval_col = 'meta_p_adj'
            meta_stats = compute_trend(meta_stats, pval_col=pval_col, slope_col='slope', col=feature_col, min_degree=min_degree)
            stats_store.append(meta_stats)
        if len(stats_store) > 0:
            stats_discovery = pd.concat(stats_store)
            stats_discovery['condition'] = 'healthy'
            return stats_discovery
        else:
            return pd.DataFrame()
    if ('_M' in par['type']) or ('_F' in par['type']): # one meta analysis for all datasets for gender specific analysis
        min_degree = 2
        meta_analysis_type='max'
        if ('_M' in par['type']):
            datasets = ['data1', 'data13_Japanese', 'data13_Korean']
        elif ('_F' in par['type']):
            datasets = ['data1', 'data13_Japanese', 'data13_Korean']
        else:
            raise ValueError('Unknown type')
            
        stats_all = run_func(datasets)
        stats_all['race'] = 'both'
    else: # seperate meta analysis for asian and european
        min_degree = 3
        meta_analysis_type='fisher'
        print(f'Running meta analysis for European datasets, min degree  {min_degree}, meta_analysis_type {meta_analysis_type}')
        stats_e = run_func(datasets_e, min_degree, meta_analysis_type)
        stats_e['race'] = 'european'

        min_degree = 2
        meta_analysis_type='max'
        print(f'Running meta analysis for Asian datasets, min degree  {min_degree}, meta_analysis_type {meta_analysis_type}')
        stats_a = run_func(datasets_a, min_degree, meta_analysis_type)
        stats_a['race'] = 'asian'

        
        meta_analysis_type='fisher'
        min_degree = 4
        print(f'Running meta analysis for all datasets, min degree  {min_degree}, meta_analysis_type {meta_analysis_type}')
        stats_both = run_func(datasets_all, min_degree, meta_analysis_type)
        # print(stats_both[stats_both['cell_type'] == 'CD8T'].groupby('tf').size().sort_values(ascending=False).head(20))
        # aaa
        stats_both['race'] = 'both'

        # - combine
        stats_all = pd.concat([stats_e, stats_a, stats_both])

    #- save
    print('Saving results to ', par['stats_all'])
    stats_all.to_csv(par['stats_all'], index=False)

def wrapper_association_with_age_condition(par, cell_types, datasets, feature_type='tf_activity', features=None, test_type='unpaired'):
    # - calculate tf activity for all datasets
    print(f'Association {feature_type} with age/disease...')
    stats_store = []
    for cell_type in tqdm(cell_types, desc='cell types'):
        # ----------- calculate tf activity for all datasets
        for dataset in datasets:
            try:
                adata = retrieve_feature_data(dataset, cell_type, par['type'], feature_type=feature_type)
            except ValueError as e:
                print(e)
                continue

            # - add which cell type resolution to run the analysis
            adata.obs['cell_type_resolution'] = adata.obs[par['cell_type_resolution']]
            
            for cell_type_resolution in adata.obs['cell_type_resolution'].unique():
                adata_sub = adata[adata.obs['cell_type_resolution']==cell_type_resolution]
                if adata_sub.shape[0] < 10:
                    print('Not enough samples for', cell_type, dataset, cell_type_resolution)
                    continue
                
                # - subset based on prior (only for target genes) -> add this to meta analysis
                if features is not None:
                    genes = features
                elif feature_type == 'gene_expression':
                    net = retrieve_net(dataset, cell_type)
                    targets = net['target'].unique()
                    genes = targets
                else:
                    genes = adata_sub.var_names
                
                adata_sub = adata_sub[:, adata_sub.var_names.isin(genes)]
                if issparse(adata_sub.X):
                    adata_sub.X = adata_sub.X.toarray()
                if 'SLE' in dataset:
                    stats = determine_stats_condition(adata_sub, test_type=test_type)
                elif dataset == 'CXCL9':
                    stats_1 = determine_stats_condition(adata_sub, ctr_group='24 h RPMI', condition_col='treatment', test_type=test_type)
                    stats_2 = determine_stats_condition(adata_sub, ctr_group='24 h LPS', condition_col='treatment', test_type=test_type,  conditions=['24 h LPS + metformin', '24 h LPS + metformin + ruxolitinib', '24 h LPS + ruxolitinib'])
                    stats = pd.concat([stats_1, stats_2])
                else:
                    stats = association_with_age(adata_sub, association_type=par['association_type'])
                    stats['condition'] = 'healthy'
                    stats['dataset'] = dataset
                stats['cell_type'] = cell_type_resolution
                
                stats_store.append(stats)
        
    stats_all = pd.concat(stats_store)
    if feature_type == 'gene_expression':
        stats_all.rename(columns={'tf': 'target'}, inplace=True)
    return stats_all

def wrapper_tf_activity(cell_types, datasets, type='bulk'):
    # --------- load data
    print('Loading data...')
    adata_dict = {dataset: retrieve_adata_bulk(dataset, type) for dataset in datasets}
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
    adata_dict = {dataset: retrieve_adata_bulk(dataset, type) for dataset in datasets}

    print('Calculating gene expression...')
    stats_store = []
    for cell_type in tqdm(cell_types, desc='cell types'):
        # ----------- calculate tf activity for all datasets
        for dataset in datasets:
            adata = adata_dict[dataset][adata_dict[dataset].obs['cell_type']==cell_type]
            if type == 'sc':
                sc.pp.normalize_total(adata)
                sc.pp.log1p(adata)
            write_feature_data(adata, dataset, cell_type, type, feature_type='gene_expression')

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

def association_with_age(adata, gene_col='tf', association_type='linear'):
    '''
    Calculate p-values for the linear regression of the top tfs across datasets with ageing,
    and apply FDR correction (Benjamini-Hochberg).
    '''
    p_value_store = []

    for gene in adata.var_names:
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

def compute_trend(df, pval_col='meta_p_adj', slope_col='slope', col='tf', min_degree=None):
    # Compute -log10(p_value_adj) for dot size
    df["neg_log10_adj_pval"] = -np.log10(df[pval_col])
    if 'trend' in df.columns:
        df.drop('trend', inplace=True, axis=1)

    # Function to assign trend based on slope sign and min_degree
    def determine_trend(x):
        pos = (x > 0).sum()
        neg = (x < 0).sum()
        total = len(x)
        
        if pos == total:
            return 'Increase in aging'
        elif neg == total:
            return 'Decrease in aging'
        else:
            # if min_degree is not None:
            #     if pos >= min_degree:
            #         return 'Increase in aging'
            #     elif neg >= min_degree:
            #         return 'Decrease in aging'
            #     else:
            #         return 'Inconsistent'
            # else:
            #     return 'Inconsistent'
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

    if False: # run decoupler
        mat = pd.DataFrame(
            data=adata.X.todense(),  
            columns=adata.var_names,  
            index=adata.obs.index  
        )

        tf_acts, tf_pvals = decoupler.run_ulm(mat, net, source='source', target='target', weight='weight', use_raw=False)
        # - formatize
        tf_acts = tf_acts.reset_index().melt(id_vars='index', var_name='source', value_name='activity')
     
        # obs = adata.obs[cols]
        obs = adata.obs.copy()
        obs = obs.reset_index()
        
        tf_acts['index'] = tf_acts['index'].astype(str)
        obs['index'] = obs['index'].astype(str)
        tf_acts = tf_acts.merge(obs, on='index', how='left').drop('index', axis=1)
        assert tf_acts.shape[0]==tf_acts.shape[0]
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
import pandas as pd
from scipy.stats import hypergeom
import numpy as np

def run_ora_local(gene_list, background_genes, gene_sets, min_size=5, max_size=500):
    """
    Perform Over-Representation Analysis (ORA).

    Parameters:
    - gene_list: set or list of input genes
    - background_genes: set or list of background genes (universe)
    - gene_sets: dict of pathway_name -> set/list of genes
    - min_size: minimum gene set size to consider
    - max_size: maximum gene set size to consider

    Returns:
    - pd.DataFrame with columns: Term, Overlap, P-value, Adjusted P-value (FDR), Gene Ratio, Genes
    """

    gene_list = set(gene_list)
    background_genes = set(background_genes)

    results = []

    M = len(background_genes)  # total genes in background
    n = len(gene_list & background_genes)  # overlap between input genes and background

    for term, term_genes in gene_sets.items():
        term_genes = set(term_genes)
        term_genes = term_genes & background_genes  # restrict to universe

        N = len(term_genes)
        if N < min_size or N > max_size:
            continue

        k = len(gene_list & term_genes)  # hits in gene list
        if k == 0:
            continue

        # Hypergeometric test: P(X ≥ k)
        pval = hypergeom.sf(k - 1, M, N, n)

        # Collect data
        results.append({
            "Term": term,
            "Gene Set Size": N,
            "Hits": k,
            "P-value": pval,
            "Gene Ratio": k / n,
            "Genes": list(gene_list & term_genes),
        })

    # Compile and adjust p-values
    df = pd.DataFrame(results)
    if not df.empty:
        df['FDR'] = np.minimum(1.0, df['P-value'] * len(df))  # Benjamini-Hochberg correction (simplified)
        df = df.sort_values("P-value")
    return df
def pathway_analysis_wrapper(df, pvalue_col='meta_p_adj'):
    import gseapy as gp
    from ciim.src.utils.util import get_genesets
    from gseapy import barplot, dotplot
    # all_genes = np.loadtxt(f"/vol/projects/jnourisa/prior/tf_all.csv", dtype=str).tolist()
    all_genes = np.loadtxt(f'/vol/projects/jnourisa/prior/gene_names.txt', dtype=str)
    gene_sets =  get_genesets()
    res2d_store = []
    for cell_type in df['cell_type'].unique():
    # for cell_type in ['CD8T']:
        for trend in df['trend'].unique():
        # for trend in ['Decrease in aging']:
            mask = (df['cell_type'] == cell_type) & (df['trend'] == trend)
            if mask.sum() == 0:
                continue
            stats_df = df[mask]
            # - prepare
            stats_df = stats_df[['tf', pvalue_col]]
            all_tfs = stats_df['tf'].unique().tolist()
            genes = stats_df[stats_df[pvalue_col]<0.05]['tf'].unique().tolist()
            if True:
                rr = gp.enrichr(gene_list=list(genes),
                                gene_sets=['MSigDB_Hallmark_2020'], #, 'KEGG_2021_Human'
                                organism='human', 
                                outdir=None, 
                                cutoff=1
                                # background=list(tf_all),
                                )
                res2d = rr.res2d
                res2d.rename(columns={'Adjusted P-value': 'FDR'}, inplace=True)
            else:
                res2d = run_ora_local(genes, background_genes=all_genes, gene_sets=gene_sets, min_size=1, max_size=500)
                res2d['Term'] = (
                res2d['Term']
                    .str.replace('HALLMARK_', '', regex=False)
                    .str.replace('_', ' ', regex=False)
                    # .str.title()
                )

            filter_col = 'FDR' #'FDR q-val'
            res2d = res2d[res2d[filter_col]<0.05]
            
            if res2d.shape[0] == 0:
                continue
            print(res2d.shape)
            res2d['cell_type'] = cell_type
            res2d['trend'] = trend
            res2d_store.append(res2d)
    if len(res2d_store) == 0:
        return None
    
    res2d_all = pd.concat(res2d_store)
    return res2d_all
