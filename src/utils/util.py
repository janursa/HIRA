
import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import sys
import scanpy as sc
import matplotlib.pyplot as plt

from task_grn_inference.src.utils.util import sum_by, read_gmt


def get_genesets():

    geneset_file = '../input/prior/h.all.v2024.1.Hs.symbols.gmt'
    genesets_all = read_gmt(geneset_file) 
    genesets_all = {key:gs['genes'] for key, gs in genesets_all.items()}
    # extract relevant sets 
    # gene_sets = {}
    # # map_dict = {
    # #     'HALLMARK_PI3K_AKT_MTOR_SIGNALING': 'PI3K/AKT/MTOR',
    # #     'HALLMARK_MTORC1_SIGNALING': 'MTORC1',
    # #     'HALLMARK_P53_PATHWAY': 'P53',
    # #     'HALLMARK_TNFA_SIGNALING_VIA_NFKB': 'TNFA/NFKB',
    # #     'HALLMARK_TGF_BETA_SIGNALING': 'TGF-Beta',
    # #     'HALLMARK_WNT_BETA_CATENIN_SIGNALING': 'WNT-Beta Catenin',
    # #     'HALLMARK_OXIDATIVE_PHOSPHORYLATION': 'Oxidative Phos.'
    # # }
    # for key, key_simple in genesets_all.items():
    #     gene_sets[key_simple] = genesets_all[key]

    # geneset_file = '../input/prior/c5.all.v2024.1.Hs.symbols.gmt'
    # genesets_all = read_gmt(geneset_file) 
    # genesets_all = {key:gs['genes'] for key, gs in genesets_all.items()}
    # # extract relevant sets 
    # gene_sets_andreas = {}
    # map_dict = {'GOMF_ANTIGEN_BINDING': 'Antigen binding', 
    #             'GOCC_NUCLEOSOME': 'Nucleosome', 
    #             'GOMF_EXTRACELLULAR_MATRIX_BINDING': 'ECM-binding', 
    #             'GOCC_RNA_POLYMERASE_II_CORE_COMPLEX': 'RNA polymerase 2'}
    # for key, key_simple in map_dict.items():
    #     gene_sets_andreas[key_simple] = genesets_all[key]

    # gene_sets = {**gene_sets, **gene_sets_andreas}
    return genesets_all
def get_gene2pathway():
    genesets_dict = get_genesets()
    target_genes = np.unique(np.concatenate(list(genesets_dict.values())))

    # Create a DataFrame for pathway annotations
    pathway_assignments = []
    for pathway, genes in genesets_dict.items():
        for gene in genes:
            pathway_assignments.append((gene, pathway))

    pathway_df = pd.DataFrame(pathway_assignments, columns=['gene', 'pathway']).set_index('gene')
    return pathway_df


def add_root_sample(adata):
    '''
    Add the root sample (the one with the minimum age) to the adata object for pseudotime analysis.
    '''
    if 'age' not in adata.obs.columns:
        raise ValueError("Column 'age' not found in adata.obs.")

    # Find the cell with the minimum age
    min_age = adata.obs['age'].min()
    root_cells = adata.obs.index[adata.obs['age'] == min_age]

    if len(root_cells) == 0:
        raise ValueError("No samples found to use as root.")

    # Choose the first one (or could sort and choose consistently)
    root_cell = root_cells[0]
    root_cell_idx = np.where(adata.obs.index == root_cell)[0][0]

    adata.uns['iroot'] = root_cell_idx
def run_dpt(adata, n_neighbors=10, n_comps=10):
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
    sc.tl.diffmap(adata, n_comps=n_comps)
    sc.tl.dpt(adata)
def flesh_out_collectri():
    import omnipath as op
    CollecTRI = op.interactions.CollecTRI.get(genesymbols=True, organism='human', loops=False)
    def extract_sources(source_str):
        # Split and clean each source tag
        parts = source_str.split(';')
        sources = []
        for s in parts:
            s = s.strip().replace('CollecTRI', '').replace('_', '').strip(';').strip()
            if s:
                sources.append(s)
        return sources

    # Expand the DataFrame
    expanded_rows = []
    for _, row in CollecTRI.iterrows():
        cleaned_sources = extract_sources(row['sources'])
        weight = 1 if row['is_stimulation'] else -1
        for ref in cleaned_sources:
            if ref in ['NTNU.Curated']:
                # Skip this source
                continue
            expanded_rows.append({
                'source': row['source_genesymbol'],
                'target': row['target_genesymbol'],
                'weight': weight,
                'ref': ref  # this is now a single cleaned source per row
            })

    # Create the new curated DataFrame
    curated_net = pd.DataFrame(expanded_rows)
    curated_net.to_csv('/vol/projects/jnourisa/prior/collectri_with_source.csv', index=False)
# - pseudotime analysis
def run_pseudotime_analysis(adata, seed=32):
    # - add root age: #TODO: run this multiple times to choose different root cells 
    add_root_sample(adata)
    # - dpt 
    run_dpt(adata)
    if True:
        # - visualization
        sc.tl.umap(adata)
        # sc.pl.umap(adata, color=['dpt_pseudotime', 'age'], cmap='viridis', show=True, size=3*(adata.obs['cell_count'] / adata.obs['cell_count'].max() * 100))
        # sc.pl.pca(adata, color=['dpt_pseudotime', 'age'], cmap='viridis', show=True, size=3*(adata.obs['cell_count'] / adata.obs['cell_count'].max() * 100))
    return adata
def get_canonical_pathways():
    geneset_file = '/home/jnourisa/projs/ongoing/ciim/input/prior/h.all.v2024.1.Hs.symbols.gmt'
    genesets_all = read_gmt(geneset_file) 
    genesets_all = {key: gs['genes'] for key, gs in genesets_all.items()}

    # Create a list of gene-to-pathway mappings (one-to-one mapping)
    gene_to_pathway_list = [
        (gene, pathway)
        for pathway, genes in genesets_all.items()
        for gene in genes
    ]

    # Convert the list to a DataFrame
    df_pathway = pd.DataFrame(gene_to_pathway_list, columns=["gene", "pathway"])
    df_pathway = df_pathway.set_index("gene")
    df_pathway['pathway'] = df_pathway['pathway'].str.replace('HALLMARK_','')
    df_pathway['pathway'] = df_pathway['pathway'].str.replace('_',' ')
    df_pathway['pathway'] = df_pathway['pathway'].str.title()

    return df_pathway

def stability_selection_booststrap(X, y, n_bootstrap=100, top_k=10):
    """
    Perform stability selection by bootstrapping and selecting important features.
    
    Parameters:
    - X: Feature matrix
    - y: Target vector
    - n_bootstrap: Number of bootstrap iterations.
    - top_k: Number of top features to select.
    
    Returns:
    - top_predictors: List of selected top-k most important features.
    """
    feature_coeffs = np.zeros((n_bootstrap, X.shape[1]))
    
    for i in range(n_bootstrap):
        # Bootstrap sampling
        X_resampled, y_resampled = resample(X, y, random_state=np.random.randint(0, 10000))
        
        # Scale data
        scaler = StandardScaler()
        X_resampled = scaler.fit_transform(X_resampled)

        # Fit Ridge regression
        model = Ridge(alpha=1)
        model.fit(X_resampled, y_resampled)
        
        # Store absolute coefficients
        feature_coeffs[i, :] = np.abs(model.coef_)

    # Compute median coefficient magnitude per feature
    feature_importance = np.median(feature_coeffs, axis=0)
    
    # Get indices of the top-k most important features
    top_features_idx = np.argsort(feature_importance)[-top_k:]
    
    return top_features_idx


def fit_final_model(X, y, top_features_idx):
    """
    Train a final model using only the selected most important features.
    
    Parameters:
    - X: Feature matrix
    - y: Target vector
    - top_features_idx: Indices of the top selected features.
    
    Returns:
    - model_final: Fitted Ridge regression model.
    - y_pred: Predictions of the final model.
    - r2: R² score of the final model.
    - spearman: Spearman correlation of the final model.
    """
    X_selected = X[:, top_features_idx]
    
    # Scale data
    scaler = StandardScaler()
    X_selected = scaler.fit_transform(X_selected)

    # Fit final model
    model_final = Ridge(alpha=1)
    model_final.fit(X_selected, y)
    
    # Predict and evaluate
    y_pred = model_final.predict(X_selected)
    r2 = r2_score(y, y_pred)
    spearman = spearmanr(y_pred, y)[0]
    
    return model_final, y_pred, r2, spearman

def stability_selection_shap(X, y,  top_q=80):
    """
    Perform stability selection using SHAP values for feature importance.

    Parameters:
    - X: Feature matrix
    - y: Target vector
    - n_bootstrap: Number of bootstrap iterations.
    - top_k: Number of top features to select.

    Returns:
    - top_predictors: List of selected top-q most important features.
    """
    
    # Scale data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    # Fit a model (e.g., RandomForest for SHAP)
    # model = RandomForestRegressor(n_estimators=100, random_state=42)
    model = Ridge(random_state=42)

    model.fit(X, y)

    # Compute SHAP values
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Compute mean absolute SHAP values for feature importance
    feature_importances = np.abs(shap_values.values).mean(axis=0)


    # Compute the q percentile threshold
    threshold = np.percentile(feature_importances, top_q)

    # Select features above the threshold
    top_features_idx = np.where(feature_importances >= threshold)[0]


    return top_features_idx
def find_robust_predictors(adata, target, top_q=90):
    """
    Main function to perform stability selection, feature importance, and model evaluation.
    
    Parameters:
    - adata:  adata  
    - target: Column name for the target variable.
    - top_q: q of top features to select.
    """
    # Prepare data
    X = adata.X
    X = X.toarray() if scipy.sparse.issparse(X) else X
    y = adata.obs[target].values
    feature_names = adata.var_names

    # Stability selection
    top_features_idx = stability_selection_shap(X, y, top_q=top_q)
    top_predictors = feature_names[top_features_idx].values
    # print(f"Selected {len(top_predictors)} most important features.")

    # Fit final model with selected features
    model_final, y_pred, r2, spearman = fit_final_model(X, y, top_features_idx)

    # Print final model evaluation
    print(f"Final Model R²: {r2:.2f}, Spearman: {spearman:.2f}")
    # print("Most important predictors:", list(top_predictors))
    
    return list(top_predictors), r2

def basic_qc(adata, min_genes_per_cell = 200, max_genes_per_cell = 5000, min_cells_per_gene = 10):
    mt = adata.var_names.str.startswith('MT-')
    print('shape before ', adata.shape)
    # 1. stats
    total_counts = adata.X.sum(axis=1)
    n_genes_by_counts = (adata.X > 0).sum(axis=1)
    # mt_frac = adata[:, mt].X.sum(axis=1) / total_counts
    
    low_gene_filter = (n_genes_by_counts < min_genes_per_cell)
    high_gene_filter = (n_genes_by_counts > max_genes_per_cell)
    # mt_filter = mt_frac > max_mt_frac

    # 2. Filter cells
    # print(f'Number of cells removed: below min gene {low_gene_filter.sum()}, exceed max gene {high_gene_filter.sum()}')
    mask_cells=  (~low_gene_filter)& \
                 (~high_gene_filter)
                #  (~mt_filter)
    # 3. Filter genes
    n_cells = (adata.X!=0).sum(axis=0)
    mask_genes = n_cells>min_cells_per_gene
    adata_f = adata[mask_cells, mask_genes]
    print('shape after ', adata_f.shape)
    return adata_f

