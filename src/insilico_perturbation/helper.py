import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

from ciim.src.tf_activity.helper import retrieve_sig_stats


def compute_tf_slopes(adata, sig_tfs):
    gene_names = adata.var_names.str.split('//').str[0]
    sig_tfs = [tf for tf in sig_tfs if tf in gene_names]
    mask_genes = gene_names.isin(sig_tfs)
    assert len(mask_genes) == adata.n_vars  # should now pass
    adata.var.index = adata.var.index.astype('category')

    adata_sig = adata[:, mask_genes].copy()
    X = pd.DataFrame(adata_sig.X, columns=adata_sig.var_names)
    ages = adata_sig.obs['age'].astype(float).values.reshape(-1, 1)
    slopes = {}
    for tf in adata_sig.var_names:
        tf_values = X[tf].values.reshape(-1, 1)
        if np.all(np.isnan(tf_values)) or np.all(tf_values == tf_values[0]):
            continue
        model = LinearRegression()
        model.fit(ages, tf_values)
        slopes[tf] = model.coef_.item()

    slope_df = pd.DataFrame.from_dict(slopes, orient='index', columns=['slope'])
    return slope_df

def perturb_tf(adata, tfs, slope_df, years=10):
    print('Number of tfs to perturb:', len(tfs))

    # Extract gene names (first part before //)
    gene_names = adata.var_names.str.split('//').str[0]
    slope_genes = slope_df.index.str.split('//').str[0]

    # Create a DataFrame from the expression matrix with var_names as columns
    expr_df = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
                           columns=adata.var_names,
                           index=adata.obs_names)

    # Iterate over transcription factors
    for tf in tfs:
        if tf not in gene_names.values or tf not in slope_genes.values:
            continue

        # Get the full var_name matching this TF in adata and slope_df
        tf_varnames = adata.var_names[gene_names == tf]
        tf_slope_names = slope_df.index[slope_genes == tf]

        for varname in tf_varnames.intersection(tf_slope_names):
            delta = slope_df.loc[varname].values
            if pd.isna(delta).any():
                continue
            expr_df[varname] += delta * years

    # Create a copy of the original AnnData and replace X
    adata_perturb = adata.copy()
    adata_perturb.X = expr_df.values

    return adata_perturb
def experiment_perturb_tfs(dataset, cell_type, data_type, tfs, n_donors=20, reg_type='ridge'):
    from ciim.src.clock.helper import prepare_input, predict_age
    import anndata as ad
    # - prepare the input and select donors 
    adata = prepare_input(dataset, cell_type, feature_type='tf_activity', data_type=data_type)
    adata.obs['donor_age'] = adata.obs['donor_id'].astype(str) + '_' + adata.obs['age'].astype(str)
    donors = adata.obs['donor_age'].unique()
    np.random.seed(0)
    donors = np.random.choice(donors, n_donors, replace=False)
    adata = adata[adata.obs['donor_age'].isin(donors)]

    # - get the significant TFs and compute slopes
    if tfs == 'aging_tfs': # perturb the sig tfs 
        print('Perturbing the aging TFs')
        stats_sig = retrieve_sig_stats(type='bulk', race='both', filter_inconsistent=True)
        stats_sig = stats_sig[stats_sig['cell_type'] == cell_type]
        tfs = stats_sig['tf'].unique()
    elif tfs == 'all_tfs': # perturb all tfs
        print('Perturbing all TFs')
        tfs = adata.var_names
    elif isinstance(tfs, list): # perturb the given tfs
        print('Perturbing the given TFs')
        tfs = [tf for tf in tfs if tf in adata.var_names]
    else:
        raise ValueError("perturb_coverage should be either 'aging_tfs' or 'all_tfs'")
    slope_df = compute_tf_slopes(adata.copy(), tfs)

    trend = 'increase'  # specify the trend for perturbation
    if trend == 'increase':
        slope_df = slope_df
    elif trend == 'decrease':
        slope_df = -slope_df
    else:
        raise ValueError("trend should be either 'increase' or 'decrease'")

    # - perturb the TFs and create a new adata object
    adata_perturb = perturb_tf(adata.copy(), tfs, slope_df)
    # - combine the adatas and predict age
    ctr = 'Unperturbed'
    treatment = 'Perturbed'

    adata.obs['condition'] = ctr
    adata_perturb.obs['condition'] = treatment
    adata_combined = ad.concat([adata, adata_perturb], axis=0)
    adata_combined = predict_age(adata_combined, cell_type, feature_type='tf_activity', data_type=data_type, reg_type=reg_type)
    obs_combined = adata_combined.obs.copy()
    # - subset and calculate age acceleration
    df_pivot = obs_combined.pivot(index='donor_age', columns='condition', values='predicted_age')
    df_pivot = df_pivot.dropna(subset=[ctr, treatment])
    df_pivot['diff'] = df_pivot[treatment] - df_pivot[ctr]
    df_pivot['cell_type'] = cell_type
    df_pivot['dataset'] = dataset

    return df_pivot