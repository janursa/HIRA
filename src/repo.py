
def diff_corr_all():
    ## - par
    par = {
        # 'dataset_file': 'input/dataset_1_2.h5ad',
        # 'save_file': f'output/net_all_fisher_dataset1_2.csv',
        'dataset_file': 'input/dataset_12.h5ad',
        'save_file': f'output/net_all_dataset12.csv',
        'mode': 'fisher',
        'major_celltypes': ['B cells',
                            'CD4+ T cells', 'MAIT cells', 'Myeloid cells', 'NK cells',
                            'TRAV1-2- CD8+ T cells', 'gd T cells'],
        'ref_age_group': '34-',
        'min_cells_per_gene': 500
    }
    
    genesets = get_genesets()
    target_genes = np.unique(np.concatenate(list(genesets.values())))
    # - dependencies
    adata = ad.read_h5ad(par['dataset_file'])
    mask_genes = adata.var_names.isin(target_genes) 
    i_run = 0
    if 'batch_group' in adata.obs:
        batch_groups = adata.obs['batch_group'].unique()
    else:
        adata.obs['batch_group'] = 'all'
        batch_groups = adata.obs['batch_group'].unique()
    for batch_group in batch_groups:
        batch_mask = adata.obs['batch_group'] == batch_group
        for i_cell_type, cell_type in enumerate(par['major_celltypes']+['-1']):
            if cell_type == '-1':
                cell_type_mask = np.asarray([True for i in range(adata.shape[0])])
            else:
                cell_type_mask = (adata.obs['cell_type'] == cell_type)

            for sample_age_group in adata.obs['age_group'].unique():
                # - mask ctr and sample
                mask_ctr = (adata.obs['age_group'] == par['ref_age_group']) & cell_type_mask & batch_mask
                mask_sample = (adata.obs['age_group'] == sample_age_group) & cell_type_mask & batch_mask

                adata_ctr = basic_qc(adata[mask_ctr, mask_genes], min_cells_per_gene=par['min_cells_per_gene'], min_genes_per_cell=10)
                adata_sample = basic_qc(adata[mask_sample, mask_genes], min_cells_per_gene=par['min_cells_per_gene'], min_genes_per_cell=10)

                if (adata_ctr.shape[0]==0) | (adata_sample.shape[0]==0):
                    continue
                    
                if (adata_ctr.shape[1]==0) | (adata_sample.shape[1]==0):
                    continue

                # - harmonize feature space 
                shared_genes = np.intersect1d(adata_ctr.var_names, adata_sample.var_names)
                adata_ctr = adata_ctr[:, shared_genes]
                adata_sample = adata_sample[:, shared_genes]
                # - normalize 
                X_norm = sc.pp.normalize_total(adata_ctr, inplace=False)['X']
                adata_ctr.layers['X_norm'] = sc.pp.log1p(X_norm, copy=True)
                X_norm = sc.pp.normalize_total(adata_sample, inplace=False)['X']
                adata_sample.layers['X_norm'] = sc.pp.log1p(X_norm, copy=True)

                # - actual subset 
                expression_ctr = adata_ctr.layers['X_norm'].todense().A
                expression_sample = adata_sample.layers['X_norm'].todense().A
                gene_names = shared_genes

                # - cell type info 
                cell_type_ctr = adata_ctr.obs.cell_type
                cell_type_sample = adata_sample.obs.cell_type

                # - actuall calcualtion
                net = diff_corr(expression_ctr, expression_sample, gene_names, sig_t=1, cell_type_ctr=cell_type_ctr, cell_type_sample=cell_type_sample, mode=par['mode'])
                net['age_group'] = sample_age_group
                net['batch_group'] = batch_group
                net['cell_type'] = cell_type
                

                # - store 
                if i_run == 0:
                    net_all = net
                else:
                    net_all = pd.concat([net_all, net], axis=0)
                i_run+=1
    
    net_all.to_csv(par['save_file'])

def diff_corr(expression_ctr, expression_sample, gene_names, 
                     cell_type_ctr, cell_type_sample, n_permutations=1000,
                     parallel=True, sig_t=.05, mode='zscore') -> pd.DataFrame:
    ''' Permutation based calculation of differentation correlation'''
    import warnings
    warnings.filterwarnings(
        "ignore",
        message="Received a view of an AnnData. Making a copy.",
        category=UserWarning
    )
    def compute_correlation(X):
        corr, _ = spearmanr(X, nan_policy='raise')
        expression_sample = expression_sample[:, (~mask_zero_std_ctr)&(~mask_zero_std_sample)]
        assert np.isnan(corr_sample).any()==False
        return corr
    # Your code here
    from joblib import Parallel, delayed
    assert mode in ['fisher', 'permut']

    
    # check if any gene has zero std -> would generate nan in the corr
    assert expression_ctr.shape[1] == expression_sample.shape[1]

    mask_zero_std_ctr = np.std(expression_ctr, axis=0)==0
    mask_zero_std_sample = np.std(expression_sample, axis=0)==0

    expression_ctr = expression_ctr[:, (~mask_zero_std_ctr)&(~mask_zero_std_sample)]
    expression_sample = expression_sample[:, (~mask_zero_std_ctr)&(~mask_zero_std_sample)]

    assert expression_ctr.shape[1] == expression_sample.shape[1]

    gene_names  = gene_names[(~mask_zero_std_ctr)&(~mask_zero_std_sample)]

    # stats
    n_samples_ctr = expression_ctr.shape[0]
    n_samples_sample = expression_sample.shape[0]
    n_genes = expression_ctr.shape[1]

    # Compute correlation matrices for young and old groups
    corr_ctr = compute_correlation(expression_ctr)
    corr_sample = compute_correlation(expression_sample)

    assert np.isnan(corr_ctr).any()==False
    assert np.isnan(corr_sample).any()==False

    # Compute differences in correlations
    corr_diff = corr_sample - corr_ctr
    assert np.isnan(corr_diff).any()==False

    if mode=='permut': # - permutation based p value
        # Combine the data
        combined_data = np.concatenate([expression_ctr, expression_sample], axis=0)

        # Run permutations in parallel
        if False:
            def single_permutation(seed): #
                np.random.seed(seed)  
                shuffled_data = np.random.permutation(combined_data)
                perm_ctr = shuffled_data[:n_samples_ctr, :]
                perm_sample = shuffled_data[n_samples_ctr:, :]
                
                perm_corr_ctr = compute_correlation(perm_ctr)
                perm_corr_sample = compute_correlation(perm_sample)
                
                perm_corr_diff = perm_corr_sample - perm_corr_ctr
                assert np.isnan(perm_corr_diff).any()==False
                return perm_corr_diff
        else: # stratified by cell type
            def single_permutation(seed): 
                np.random.seed(seed)
                perm_ctr = []
                perm_sample = []
                unique_cell_types = np.unique(cell_type_ctr)
                
                for cell_type in unique_cell_types:
                    idx_ctr = np.where(cell_type_ctr == cell_type)[0]
                    idx_sample = np.where(cell_type_sample == cell_type)[0]
                    combined_indices = np.concatenate([idx_ctr, idx_sample])
                    np.random.shuffle(combined_indices)

                    perm_ctr.append(combined_data[combined_indices[:len(idx_ctr)], :])
                    perm_sample.append(combined_data[combined_indices[len(idx_ctr):], :])

                perm_ctr = np.vstack(perm_ctr)
                perm_sample = np.vstack(perm_sample)

                perm_corr_ctr = compute_correlation(perm_ctr)
                perm_corr_sample = compute_correlation(perm_sample)
                return perm_corr_sample - perm_corr_ctr
        seeds = np.arange(n_permutations)  # Unique seed for each permutation
        
        if parallel:
            perm_diffs = Parallel(n_jobs=-1, backend="loky")(
                delayed(single_permutation)(seed) for seed in tqdm(seeds)
            )
            perm_diffs = np.array(perm_diffs)
        else:
            perm_diffs = np.zeros((n_permutations, n_genes, n_genes))
            for p, seed in tqdm(enumerate(seeds)):
                perm_diffs[p] = single_permutation(seed)
        
        
        # Compute p-values for observed differences
        p_values = np.ones((n_genes, n_genes))
        for i in range(n_genes):
            for j in range(i + 1, n_genes):
                observed_diff = corr_diff[i, j]
                perm_distribution = perm_diffs[:, i, j]
                p_value = (np.sum(np.abs(perm_distribution) >= np.abs(observed_diff)) + 1) / (n_permutations + 1) # - check this
                p_values[i, j] = p_value
                p_values[j, i] = p_value

        # Adjust p-values for multiple testing using FDR
        p_values_flat = p_values[np.triu_indices(n_genes, k=1)]
        _, p_values_corrected, _, _ = multipletests(p_values_flat, method='fdr_bh')

        p_values_corrected_matrix = np.zeros_like(p_values)
        p_values_corrected_matrix[np.triu_indices(n_genes, k=1)] = p_values_corrected
        p_values_corrected_matrix += p_values_corrected_matrix.T
    elif mode=='fisher': # fisher z test
        def fisher_z_test(corr1, n1, corr2, n2):
            """Perform Fisher's Z-Test for two correlation coefficients."""
            from scipy.stats import spearmanr, norm
            # Fisher's Z-transformation
            corr1 = np.clip(corr1, -0.9999, 0.9999)
            corr2 = np.clip(corr2, -0.9999, 0.9999)
            z1 = 0.5 * np.log((1 + corr1) / (1 - corr1))
            z2 = 0.5 * np.log((1 + corr2) / (1 - corr2))
            
            # Standard error of the Z-difference
            se_diff = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
            
            # Z-difference test statistic
            z_diff = (z2-z1) / se_diff
            
            # Two-tailed p-value
            p_value = 2 * norm.sf(np.abs(z_diff))
            
            return p_value
        
        p_values = fisher_z_test(corr_sample, n_samples_sample, corr_ctr, n_samples_ctr)
        p_values_corrected_matrix = p_values
    # - only keep the sig change
    mask_sig = p_values_corrected_matrix<sig_t 
    corr_diff[~mask_sig] = 0

    # - efficient melting 
    upper_triangle_indices = np.triu_indices_from(corr_diff, k=1)

    # Extract the source and target gene names based on the indices
    sources = np.array(gene_names)[upper_triangle_indices[0]]
    targets = np.array(gene_names)[upper_triangle_indices[1]]
    link = np.asarray(['_'.join(sorted([str(src), str(tgt)])) for src, tgt in zip(sources, targets)])

    # Extract the corresponding correlation values
    diff_values = corr_diff[upper_triangle_indices]
    
    ctr_values = corr_ctr[upper_triangle_indices]
    sample_values = corr_sample[upper_triangle_indices]
    p_values = p_values_corrected_matrix[upper_triangle_indices]


    mask_zeros = diff_values == 0 
    # Create a structured array
    data = np.column_stack((link[~mask_zeros], ctr_values[~mask_zeros].round(2), sample_values[~mask_zeros].round(2), diff_values[~mask_zeros].round(2), p_values[~mask_zeros].round(5)))

    # Convert to DataFrame
    net = pd.DataFrame(data, columns=['link', 'ctr_corr', 'corr', 'diff', 'adj_pvalue']).set_index('link')
    net = net.astype(float)
    return net


def plot_enrichment(df, figsize=(5, 6)):
    # Calculate -log10 of Adjusted P-value for a clearer visualization
    df['-log10(Adjusted P-value)'] = -np.log10(df['Adjusted P-value'])
    

    # Sort the DataFrame by significance (optional)
    df = df.sort_values(by='-log10(Adjusted P-value)', ascending=False)

    # Plotting the enrichment analysis
    plt.figure(figsize=figsize)
    sns.barplot(data=df, y='Term', x='-log10(Adjusted P-value)', palette="viridis")

    # Add plot labels and title
    plt.xlabel("-log10(Adjusted P-value)")
    plt.title("Enrichment Analysis Results")
    plt.tight_layout()

    
def link_to_matrix(df):
    df[['source', 'target']] = df['link'].str.split('_', expand=True)

    sources = df.source.tolist()
    targets = df.target.tolist()
    weights = df.weight.tolist()

    genes = np.unique(sources + targets)

    # Create index mapping for genes
    gene_idx = {gene: idx for idx, gene in enumerate(genes)}
    # Initialize the gene-gene matrix with zeros
    n_genes = len(genes)
    g_g_matrix = np.zeros((n_genes, n_genes))

    # Populate the matrix with weights
    for source, target, value in zip(sources, targets, weights):
        i, j = gene_idx[source], gene_idx[target]
        g_g_matrix[i, j] = value
        g_g_matrix[j, i] = value  # Make it symmetric
    net = pd.DataFrame(g_g_matrix, columns= genes, index=genes)
    return net 

def determine_sig_links(results_df, fold_change_t=2, pvalue_t=0.05):
    abs_min = results_df[['mean', 'mean_ctr']].values.ravel().min()
    results_df['mean_scalled'] = results_df['mean']+np.abs(abs_min)
    results_df['mean_ctr_scalled'] = results_df['mean_ctr']+np.abs(abs_min)

    results_df['fold_change'] = results_df['mean_scalled']/results_df['mean_ctr_scalled']

    flag = (results_df.pvalue<pvalue_t)&((results_df['fold_change']>fold_change_t) | (results_df['fold_change']<(1/fold_change_t)))
    results_df['sig'] = flag
    return results_df
def volcanic_plot_diff(df, diff_t=.1, pvalue_t=0.05):
    # Calculate log fold change and negative log10 p-value
    df['neg_log10_pvalue'] = -np.log10(df['pvalue'])  # Negative log10 of p-value

    # Create the volcano plot
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x='diff', y='neg_log10_pvalue', hue='sig_diff', style='sig', s=100, palette={False: 'grey', True: 'red'})

    # Add labels and title
    # plt.title('Volcano Plot')
    plt.xlabel('Mean Change')
    plt.ylabel('-Log10 P-value')
    plt.axhline(y=-np.log10(pvalue_t), color='red', linestyle='--', label=f'p-value = {pvalue_t}')
    plt.axvline(x=diff_t, color='red', linestyle='-.', label=f'Mean change = {diff_t}')  # Vertical line at x=0
    plt.axvline(x=-diff_t, color='red', linestyle='-.', label=f'Mean change = {diff_t}')  # Vertical line at x=0
    # plt.legend(title='Significance')
    plt.grid()

def volcanic_plot(df, fold_change_t=2, pvalue_t=0.05):
    # Calculate log fold change and negative log10 p-value
    df['log_fold_change'] = np.log2(df['fold_change'])  # Log2 transformation of fold change
    df['neg_log10_pvalue'] = -np.log10(df['pvalue'])  # Negative log10 of p-value

    # Create the volcano plot
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x='log_fold_change', y='neg_log10_pvalue', hue='sig', style='sig', s=100, palette={False: 'grey', True: 'red'})

    # Add labels and title
    # plt.title('Volcano Plot')
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10 P-value')
    plt.axhline(y=-np.log10(pvalue_t), color='red', linestyle='--', label=f'p-value = {pvalue_t}')
    plt.axvline(x=np.log2(fold_change_t), color='red', linestyle='-.', label=f'Fold change = {fold_change_t}')  # Vertical line at x=0
    plt.axvline(x=-np.log2(fold_change_t), color='red', linestyle='-.', label=f'Fold change = {fold_change_t}')  # Vertical line at x=0
    # plt.legend(title='Significance')
    plt.grid()



def denoise_func(X):
    from sklearn.decomposition import PCA, TruncatedSVD
    pca = TruncatedSVD(n_components=100) 
    X_pca = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_pca)
    return X_reconstructed

def corr_latent_space(X, n_components):
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_standardized = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X_standardized)  
    W_k = pca.components_.T 
    covariance_matrix = W_k @ W_k.T  
    std_dev = np.sqrt(np.diag(covariance_matrix))
    correlation_matrix = covariance_matrix / np.outer(std_dev, std_dev)
    return correlation_matrix
def calculate_coexp_all(adata_dir='input/adata.h5ad', normalize='sla', corr_method='spearman', write_file='output/coexp_adata_sla_spearman.h5ad', denoise=False, targeted=True):
    
    genesets = get_genesets()
    target_genes = np.unique(np.concatenate(list(genesets.values())))

    adata_all = ad.read_h5ad(adata_dir) 
    if targeted:
        adata_all = adata_all[:, adata_all.var_names.isin(target_genes)]

    i_all = 0
    for i, age_group in enumerate(adata_all.obs.age_group.unique()):
        adata_age_group= adata_all[adata_all.obs.age_group==age_group]
        
        coexp_age_group = calculate_coexp(adata_age_group, corr_method=corr_method, denoise=denoise, normalize=normalize)

        if i_all == 0:
            coexp_all = coexp_age_group
        else:
            coexp_all = pd.concat([coexp_all, coexp_age_group], axis=0).fillna(0)
        i_all+=1

    # to adata
    obs = coexp_all.index
    obs = obs.to_frame().reset_index(drop=True)
    var = pd.DataFrame(index=coexp_all.columns)
    coexp_adata = ad.AnnData(X=coexp_all.values, obs=obs, var=var)
    coexp_adata.write(write_file)

def calculate_coexp(adata, layer=None, group='age_donor', corr_method='pearson', denoise=False, normalize='sla'):

        for i_donor, age_donor in enumerate(tqdm(adata.obs[group].unique())):
            adata_age = adata[adata.obs[group].eq(age_donor), :]

            adata_age = basic_qc(adata_age, min_cells_per_gene=100, min_genes_per_cell=10)

            if normalize=='sla':
                sc.pp.normalize_total(adata_age)
                sc.pp.log1p(adata_age)
            elif normalize=='apr':
                sc.experimental.pp.normalize_pearson_residuals(adata_age)

            X_subset = adata_age.X
            try:
                X_subset = X_subset.todense().A
            except:
                X_subset = X_subset
            
            if denoise:
                X_subset = denoise_func(X_subset)
                # corr_matrix = corr_latent_space(X_subset, n_components=100)
   
            if corr_method=='pearson':
                # corr_matrix = np.corrcoef(X_subset.T)
                corr_matrix, p_values = fast_pearson_with_pvalues(X_subset)
            elif corr_method=='spearman':
                corr_matrix, p_values = spearmanr(X_subset, nan_policy='raise')
        
            # - correct the p value
            p_values_flat = p_values.flatten()
            _, p_values_corrected_flat, _, _ = multipletests(p_values_flat, method='fdr_bh')
            p_values_corrected = p_values_corrected_flat.reshape(p_values.shape)

            mask_non_sig =  p_values_corrected>=0.05
            corr_matrix[mask_non_sig] = 0

            # - melt
            net = efficient_melting(corr_matrix, adata_age.var_names)

            net['link'] = ['_'.join(sorted([str(src), str(tgt)])) for src, tgt in zip(net.source, net.target)]

            # Create a MultiIndex from the specified columns in adata_age_donor
            index_df = pd.DataFrame({'age_group':adata_age.obs.age_group.unique(), 'batch_group':adata_age.obs.batch_group.unique(), 'age_donor':adata_age.obs.age_donor.unique(), 'age':adata_age.obs.age.unique(), 'cell_count': [len(adata_age)]})
            index = pd.MultiIndex.from_frame(index_df, names=['age_group', 'batch_group', 'age_donor', 'age', 'cell_count'])

            # Create a DataFrame with weights using the new 'link' and the MultiIndex
            coexp_df = pd.DataFrame([net.weight.values], 
                                    columns=net.link, index=index)

            if i_donor == 0:
                coexp_all = coexp_df
            else:
                coexp_all = pd.concat([coexp_all, coexp_df], axis=0).fillna(0)

        return coexp_all

def batch_correction():
    adata_all = ad.read_h5ad('input/dataset_1.h5ad') 

    # - subset to one batch
    adata_all = adata_all[adata_all.obs.batch_group=='batch_1']

    # - corrected the data and store them in layers
    # adata_all.layers['combat_corrected'] = csr_matrix(np.zeros(adata_all.shape))
    for age_group in tqdm(adata_all.obs.age_group.unique()):
        mask_group = (adata_all.obs.age_group == age_group) 
        adata_group = adata_all[mask_group]

        if False: # run on raw count
            combat_corrected = sc.pp.combat(adata_group, key='donor_id', inplace=False)  
            combat_corrected[(adata_group.X.todense().A==0)] = 0
            combat_corrected[combat_corrected<0] = 0
        else: # run on normalized 
            sc.pp.normalize_total(adata_group, )
            sc.pp.log1p(adata_group)
            combat_corrected = sc.pp.combat(adata_group, key='donor_id', inplace=False)  

        combat_corrected = csr_matrix(combat_corrected)
        adata_all.write('input/dataset_1_corrected_batch1.h5ad') 

        # adata_all.layers['combat_corrected'][mask_group] = combat_corrected 
    adata_all.layers['counts'] = adata_all.X.copy()
    adata_all.write('input/dataset_1_corrected_batch1.h5ad') 


def sig_test_all(coexp_adata_file='output/coexp_adata_apr_spearman.h5ad', ctr_group='34-', col_contrast='age_group', col_link='link', save_file='output/links_pvalues_vs_34.csv'):
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from scipy import stats
    coexp_adata = ad.read_h5ad(coexp_adata_file)
    coexp_adata.obs.age_group = coexp_adata.obs.age_group.replace('-34', '34-')
    coexp_adata.obs.age_group = coexp_adata.obs.age_group.astype('str').astype('category')
    print(coexp_adata)

    # - keep only that are present in all 
    mask_zeros = (coexp_adata.X==0).any(axis=0)
    coexp_adata = coexp_adata[:, ~mask_zeros]
    print(coexp_adata)
    print(coexp_adata.obs.nunique())
    # - make a df 
    coexp_df = pd.DataFrame(coexp_adata.X, columns=coexp_adata.var_names, index=pd.MultiIndex.from_frame(coexp_adata.obs, names=coexp_adata.obs.columns))

    coexp_long_df = coexp_df.reset_index(level=['batch_group', 'age_group', 'age_donor']).melt(id_vars=['batch_group', 'age_group', 'age_donor'], var_name='link', value_name='weight')
    coexp_long_df.head()

    links = coexp_long_df[col_link].unique()
    contrast_groups = coexp_long_df[col_contrast].unique()

    # Initialize list for storing results
    data_store = []

    # Group by link and batch_group at the start
    grouped = coexp_long_df.groupby([col_link, col_contrast])

    for link in tqdm(links):
        # Compute control (34-) group once
        ctr_values = grouped.get_group((link, ctr_group)).weight.values
        ctr_mean = np.mean(ctr_values)
        ctr_median = np.median(ctr_values)
        # Iterate through age groups
        for group in contrast_groups:
            if group == ctr_group:
                continue
            df_a = grouped.get_group((link, group))
            sample_values = df_a.weight.values
            sample_mean = np.mean(sample_values)
            sample_median = np.median(sample_values)

            # Calculate p-value
            pvalue = stats.mannwhitneyu(ctr_values, sample_values)[1]
            # pvalue = stats.ttest_ind(ctr_values, sample_values)[1]

            # Store results
            data_store.append({
                col_link: link,
                col_contrast: group,
                'pvalue': pvalue,
                'mean': sample_mean,
                'median': sample_median,
                'mean_ctr': ctr_mean,
                'mean_median': ctr_median,
                'diff_median': sample_median - ctr_median,
                'diff': sample_mean-ctr_mean
            })
        # print(pd.DataFrame(data_store))
        # aa

    results_df = pd.DataFrame(data_store)
    results_df.to_csv(save_file)

def plot_heatmap_across_correlation_results():
    # - plot heatmap for -log10pvalues
    sig_results['neglogpvalue'] = -np.log10(sig_results['pvalue'])
    sig_results['link_agegroup'] = sig_results['link']+'/'+sig_results['age_group']
    sig_results_mat = sig_results.pivot(index='model', columns='link_agegroup', values='neglogpvalue').fillna(0)

    age_groups = [name.split('/')[1] for name in sig_results_mat.columns]


    # Map unique age groups to colors
    unique_groups = np.unique(age_groups)
    colors = sns.color_palette("Set2", len(unique_groups))
    age_group_colors = dict(zip(unique_groups, colors))

    # Reorder the columns of 'df' to group by age group
    df = sig_results_mat
    age_group_df = pd.DataFrame({'age_group': age_groups, 'column': df.columns})
    age_group_df = age_group_df.sort_values('age_group')
    ordered_columns = age_group_df['column'].values
    ordered_age_groups = age_group_df['age_group'].values

    # Reorder the DataFrame and color list
    df = df[ordered_columns]
    col_colors = [age_group_colors[age] for age in ordered_age_groups]

    # Perform hierarchical clustering on rows
    linkage_matrix = linkage(df.values, method='ward')

    # Plot clustered heatmap with ordered columns and col_colors
    g = sns.clustermap(
        df,
        row_linkage=linkage_matrix,
        col_cluster=False,
        cmap='viridis',
        xticklabels=False,
        col_colors=col_colors,
        figsize=(15, 4)
    )
    colorbar = g.cax  # Access the colorbar axis
    colorbar.set_title("-log10-pvalue", fontsize=12)

    # Create a legend for age group colors
    for age_group, color in age_group_colors.items():
        plt.plot([], [], marker="o", ms=10, ls="", mec=None, color=color, label=age_group)
    plt.legend(title="Age Groups", bbox_to_anchor=(1.5, 1), loc='upper left', borderaxespad=0.)
    g.ax_heatmap.set_xlabel("")

    plt.show()

def geneset_specific_distribution():
    if True: # violin plot for each geneset. p values are calculcated per group
        import seaborn as sns
        import pandas as pd
        import scipy.stats as stats
        
        coexp_adata = ad.read_h5ad('output/coexp_adata_sla_spearman_False.h5ad')

        n_tests = (len(genesets_dict))*4 # to be used for corection
        def do_violin(adata, ax):
            # Prepare correlation data
            corr_df = pd.DataFrame(adata.X, columns=adata.var_names)
            corr_df['age_group'] = adata.obs['age_group'].values
            corr_long = corr_df.melt(id_vars=['age_group'], var_name='link', value_name='correlation')
            corr_long = corr_long[corr_long['correlation'].abs() > 0.2] # cutt-ff on corr
            corr_long = corr_long[corr_long['correlation'] != 0]

            # Separate `34-` and other groups
            corr_long['is_reference'] = corr_long['age_group'] == '34-'
            
            p_values = {}
            reference_group = corr_long[corr_long['is_reference'] == True]

            # Perform Mann-Whitney U test for each other age group vs. '34-'
            for age_group in corr_long['age_group'].unique():
                if age_group == '34-':
                    continue 
                target_group = corr_long[(corr_long['age_group'] == age_group)]
                u_stat, p_value = stats.mannwhitneyu(reference_group['correlation'], target_group['correlation'])
                ad_p_value = p_value*n_tests
                p_values[age_group] = min([1, ad_p_value])

                reference_group_c = reference_group.copy()
                reference_group_c['age_group'] = age_group

                corr_long = pd.concat([corr_long, reference_group_c], axis=0)
            corr_long = corr_long[corr_long.age_group!='34-']
            corr_long.age_group = corr_long.age_group.astype('category')
            # Plot side-by-side violin plots
            sns.violinplot(
                x='age_group', y='correlation', hue='is_reference',
                data=corr_long, ax=ax, split=True, inner='quart', palette={True: "orange", False: "gray"}
            )
            

            # # Set limits and add p-values
            ax.set_ylim([-1.1, 1.1])
            for i, age_group in enumerate(p_values.keys()):
                ax.text(i, 0.9, f'{p_values[age_group]:.3f}', ha='center', va='bottom', color='black')

            ax.legend(labels=["Control (34-)", "Sample"], loc=(1.01,.5))
            
        for i , name in enumerate(genesets_dict.keys()):
        # for i , name in enumerate(['RNA polymerase 2']):
            geneset = genesets_dict[name]
            fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        
            geneset_links = ['_'.join(sorted(combination)) for combination in list(itertools.combinations(geneset,2))]
            coexp_adata_g = coexp_adata[:, coexp_adata.var_names.isin(geneset_links)]
            do_violin(coexp_adata_g, ax)

            ax.set_title(f'{name}')

            plt.tight_layout()        

def pca_analysis_across_ages():
    if True: 
        coexp_adata = ad.read_h5ad('output/coexp_adata_sla_spearman_False.h5ad')
        coexp_adata.obs.age_group = coexp_adata.obs.age_group.replace('-34', '34-')
        coexp_adata.obs.age_group = coexp_adata.obs.age_group.astype('str').astype('category')
        print(coexp_adata)

        # - keep only that are present in all 
        mask_zeros = (coexp_adata.X==0).any(axis=0)
        coexp_adata = coexp_adata[:, ~mask_zeros]
        print(coexp_adata)
        print(coexp_adata.obs.nunique())
        # - make a df 
        coexp_df = pd.DataFrame(coexp_adata.X, columns=coexp_adata.var_names, index=pd.MultiIndex.from_frame(coexp_adata.obs, names=coexp_adata.obs.columns))

        coexp_long_df = coexp_df.reset_index(level=['batch_group', 'age_group', 'age_donor']).melt(id_vars=['batch_group', 'age_group', 'age_donor'], var_name='link', value_name='weight')
        coexp_long_df.head()

        if True: # testing group difference for different pcs
            sc.pp.pca(coexp_adata)

            data_store = []
            for pca_i in [0, 1, 2, 4, 5, 6]:
                for age_group in coexp_adata.obs.age_group.unique():
                    if age_group == '34-':
                        continue 
                    mask_control = coexp_adata.obs.age_group == '34-'
                    mask_sample = coexp_adata.obs.age_group == age_group
                    ctr_pca = coexp_adata[mask_control, :].obsm['X_pca'][:, pca_i]
                    ctr_sample = coexp_adata[mask_sample, :].obsm['X_pca'][:, pca_i]

                    data = {'age_group':age_group, 'PCA': pca_i, 'pvalue': stats.mannwhitneyu(ctr_pca, ctr_sample)[1]}
                    data_store.append(data)
            df_pca_pvalues = pd.DataFrame(data_store)
            df_pca_pvalues['fdr'] = multipletests(df_pca_pvalues['pvalue'], alpha=0.05, method='fdr_bh')[1]

            print(df_pca_pvalues.pivot(index='age_group', columns='PCA', values='fdr'))
    if False:
            # -- which links are connected to pca 2
            pc2_loadings = pd.DataFrame({'share':coexp_adata.varm['PCs'][:, 1]}, index= coexp_adata.var_names)
            pc2_loadings = pc2_loadings.sort_values(by='share', ascending=False, key=abs)[:500]
    if False:
            # - pca plot
            pca_df = pd.DataFrame(coexp_adata.obsm['X_pca'][:, :6], columns=[f'PC{i+1}' for i in range(6)])

            pca_df['age_group'] = coexp_adata.obs['age_group'].values
            # Create pairplot of all combinations of PC1 to PC6, colored by age_group
            sns.pairplot(pca_df, hue='age_group', palette=colors_blind)  # You can choose any palette you prefer
            plt.suptitle("Pairwise PC1 to PC6 Combinations", y=1.02)
            plt.show()

            # - umap
            coexp_adata.obs[['age','donor_id','sample']] = coexp_adata.obs.age_donor.str.split('_', expand=True)
            # Compute UMAP after PCA
            sc.pp.neighbors(coexp_adata, n_pcs=10)
            sc.tl.umap(coexp_adata)

            # Plot UMAP
            sc.pl.umap(coexp_adata, color=['age_group', 'batch_group'])  # or use any category in `obs` 
def sbatch():
    import subprocess
    from itertools import product

    # Example usage
    normalize_options = ['sla']  # Replace with actual normalization options
    corr_method_options = ['pearson', 'spearman']  # Replace with actual correlation methods
    denoise_options = [True, False]  # Boolean argument options

    i_job = 0
    for normalize, corr_method, denoise in product(normalize_options, corr_method_options, denoise_options):
        # Prepare the sbatch command with arguments
        if True:
            sbatch_command = [
                "sbatch",
                f"--job-name=coexp_analysis_{i_job}",
                "--output=logs/output_%j.log",
                "--error=logs/error_%j.log",
                "--time=02:00:00",
                "--cpus-per-task=4",
                "--mem=250G",
                "--partition=cpu",
                f"scripts/sbatch/corr_analysis.sh",
                f"{normalize}",
                f"{corr_method}",
            ]
        else:
            sbatch_command = [
                "bash",
                f"scripts/sbatch/corr_analysis.sh",
                f"{normalize}",
                f"{corr_method}",
            ]
            
        if denoise:
            sbatch_command += ['denoise']
            
        if os.path.exists(f'output/links_pvalues_vs_34_{normalize}_{corr_method}_{denoise}.csv'):
            continue
        # Submit the job
        print(f"Submitting job with normalize={normalize}, corr_method={corr_method}, denoise={denoise}")
        subprocess.run(sbatch_command)
        i_job+=1

def jaccard_cosine_nets():
    nets_equal_size = {key: net.sort_values(by='weight', ascending=False, key=abs)[:min_net_size] for key, net in nets.items()} # keep top 100 links
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)
    cousine_matrix, fig = cosine_similarity(nets_equal_size, col_name='link', weight_col='weight', figsize=(5.5, 4.5), ax=ax1, title=f'cousine similarity')
    jaccard_matrix, fig = jaccard_similarity(nets_equal_size, col_name='link', figsize=(5.5, 4.5), ax=ax2, title=f'jaccard similarity')
    plt.tight_layout()

    
def extract_geneset_data(coexp_all):
    genesets = get_genesets()
    coexp_geneset_dfs = {}
    mean_coexp_geneset_dict = {}
    results_list = []
    for name, geneset in genesets.items():
        geneset_links = ['_'.join(sorted(combination)) for combination in list(itertools.combinations(geneset,2))]
        geneset_links_present = np.intersect1d(geneset_links, coexp_all.columns)
        print(f"{name}: from {len(geneset_links)} links, present: {len(geneset_links_present)}")
        coexp_genesets = coexp_all[geneset_links_present]
        coexp_geneset_dfs[name] = coexp_genesets
        mean_coexp_geneset_dict[name] = coexp_genesets.abs().mean(axis=1)

        for index, row in coexp_genesets.iterrows():
            for link, sub_row in row.to_frame().iterrows():
                results_list.append({
                                    'geneset': name,
                                    'age': index,
                                    'link': link,
                                    'value': sub_row[0]
                                })
    coexp_geneset_melted = pd.DataFrame(results_list)
    coexp_geneset_mean = pd.DataFrame(mean_coexp_geneset_dict)
    return coexp_geneset_melted, coexp_geneset_mean


def process_grns():
    genesets = get_genesets()
    all_genesets = np.unique(np.concatenate(list(genesets.values())))

    par['models_dir'] = 'output/grn_models/pearson_corr_scgen_pearson'
    for i, model in enumerate(par['models']):
        net = pd.read_csv(f"{par['models_dir']}/{model}.csv", index_col=0) 
        print(model, 'size :', len(net))
        net['link'] = net['source'].astype(str) + '_' + net['target'].astype(str)
        if True: # subset to prior genesets
            geneset_links = ['_'.join(sorted(combination)) for combination in list(itertools.combinations(all_genesets,2))]
            net = net[net.link.isin(geneset_links)]
        # - make the links as column and weight as values
        coexp_df = pd.DataFrame(net.weight.values.reshape(1, len(net)), columns=net.link.values, index=[model])
        if i == 0:
            coexp_all = coexp_df
        else:
            coexp_all = pd.concat([coexp_all, coexp_df], axis=0).fillna(0)
        if i == 0:
            min_net_size = len(net)
        else:
            min_net_size = min([min_net_size, len(net)])   
        print(coexp_all.shape) 
    coexp_all.to_csv('output/full/coexp_all.csv')
    # print([len(net) for k, net in nets.items()])
    print('ratio of zeros:', (coexp_all==0).sum().sum()/coexp_all.size)