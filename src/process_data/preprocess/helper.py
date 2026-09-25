
import os
default_n_threads = 3 # Change this based on the number of threads you want to use (equal to the number of cores in your machine (--cpus-per-task in the SLURM script))
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"
###
import numpy as np
import scanpy as sc
import seaborn as sns
import pandas as pd
import anndata as ad
import gc
import sqlite3
from hira.src.config import (get_config, PRIOR_DIR, SUB_CT_LABEL, MAJOR_CT_LABEL,
                             DISCOVERY_COHORTS)

RAW_DIR = os.path.join(os.environ.get('HIRA_RAW_DIR', '/vol/projects/CIIM'),
                       'Healthy_Single_Cell_Data', 'initial_data_downloaded')

# Cohorts read straight from the pinned public download instead of a CIIM
# `count_matrix/*_CMtx.h5ad`. Same files Ali's data_collection.r reads; versions pinned
# by CELLxGENE dataset_version_id (see README > Pinning CELLxGENE versions).
RAW_SOURCES = {
    'onek1k': f'{RAW_DIR}/OneK1K/08984b3c-3189-4732-be22-62f1fe8f15a4.h5ad',
    'aida': f'{RAW_DIR}/AIDA_v2/d991ef8d-7f98-4617-ad56-42d78b1f417a.h5ad',
}

# CellTypist's majority_voting over-clusters (HVG -> scale -> PCA -> kNN -> leiden), which is
# precision-sensitive: the same counts at float32 move ~3.5% of the votes. The public download
# stores float32 where the CMtx stored float64, so pin the cohort's original dtype.
RAW_X_DTYPE = {'onek1k': 'float64'}

# Cohorts whose X isn't integer counts, and where the counts actually live.
# Can't be folded into curate_raw: X is unassignable on a read-only backed file,
# so this is applied once at the point a chunk is materialised.
COUNTS_SOURCE = {'aida': 'raw', 'op': 'layer:counts'}

def use_counts_X(chunk, dataset, cell_mask=None, gene_mask=None):
    """Point chunk.X at the cohort's integer counts. No-op when X already is."""
    import scipy.sparse as sp
    src = COUNTS_SOURCE.get(dataset)
    if src == 'layer:counts' and 'counts' in chunk.layers:
        X = chunk.layers.pop('counts')
        chunk.X = (X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)).astype(np.int32)
    elif src == 'raw':
        # format_data strips .raw, so go back to the file; .raw there has the same
        # var order as X, which is what cell_mask/gene_mask are positioned against.
        src_ad = ad.read_h5ad(RAW_SOURCES[dataset], backed='r')
        if cell_mask is None:  # test path: no masks, align on obs names in file order
            cell_mask = src_ad.obs_names.isin(chunk.obs_names)
            chunk = chunk[src_ad.obs_names[cell_mask]].copy()
        X = src_ad.raw.X[np.asarray(cell_mask)]  # backed sparse indexer rejects a pandas Series
        if gene_mask is not None:
            X = X[:, gene_mask]
        assert X.shape == chunk.shape, f'counts {X.shape} != chunk {chunk.shape}'
        chunk.X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    return chunk

# org.Hs.eg.db ships a plain SQLite; stdlib sqlite3 reproduces AnnotationDbi::select
# byte-for-byte, so no R and no extra image. Version matters: 3.19 renames ~322 genes.
ORG_HS_SQLITE = f'{PRIOR_DIR}/org.Hs.eg.db_3.16.0.sqlite'

def format_columns_soundlife(adata):
    """
    Add standardized column names while keeping all original columns.
    Maps SoundLife-specific columns to standard pipeline format.
    """
    print('Formatting columns for soundlife...')
    
    # Add standardized columns (keep originals)
    adata.obs['donor_id'] = adata.obs['subject.subjectGuid'].astype(str)
    adata.obs['age'] = pd.to_numeric(adata.obs['sample.subjectAgeAtDraw'], errors='coerce')
    adata.obs['race'] = adata.obs['subject.ethnicity'].astype(str)
    adata.obs['sex'] = adata.obs['subject.biologicalSex'].astype(str)
    adata.obs['visitName'] = adata.obs['sample.visitName'].astype(str)

    
    # Extract vaccinated, year, and day from sample.visitName
    # Examples: "Flu Year 1 Day 0", "Immune Variation Day 7", "Flu Year 2 Stand-Alone"
    # Study Timeline:
    # - Flu Year 1 & 2: Vaccinated cohorts
    #   - Day 0: PRE-vaccination baseline (vaccinated=False)
    #   - Day 7: POST-vaccination (vaccinated=True)
    #   - Day 90: POST-vaccination (vaccinated=True)
    # - Immune Variation: Control cohort (never vaccinated, always False)
    def parse_visit_name(visit_name):
        visit_str = str(visit_name)
        
        import re
        
        # Extract day first (e.g., "Day 0", "Day 7", "Day 90", or None for "Stand-Alone")
        day_match = re.search(r'Day (\d+)', visit_str)
        day = day_match.group(1) if day_match else None
        
        # Determine vaccinated status based on cohort and day
        if 'Flu Year' in visit_str:
            # Flu Year cohort - extract year number
            year_match = re.search(r'Flu Year (\d+)', visit_str)
            year = year_match.group(1) if year_match else None
            
            # Day 0 is PRE-vaccination (baseline), Day 7 and Day 90 are POST-vaccination
            if day == '0':
                vaccinated = 0  # PRE-vaccination baseline
            elif day in ['7', '90']:
                vaccinated = 1  # POST-vaccination
            else:
                vaccinated = None  # Unknown day (e.g., Stand-Alone)
        elif 'Immune Variation' in visit_str:
            # Immune Variation cohort - never vaccinated
            vaccinated = 0
            year = None
        else:
            vaccinated = None
            year = None
        
        return pd.Series({'vaccinated': vaccinated, 'year': year, 'day': day})
    
    # Apply parsing to all visitNames
    parsed = adata.obs['visitName'].apply(parse_visit_name)
    adata.obs['year'] = parsed['year'].astype(str)
    adata.obs['day'] = parsed['day'].astype(str)
    adata.obs['vaccinated'] = parsed['vaccinated']
    return adata
def map_cell_types_soundlife(adata):
    """
    Map AIFI_L2 annotations to major cell types and standardized sub cell types.
    Sets cell_type (Major_CT) and Sub_CT columns with standardized nomenclature.
    """
    print('Mapping cell types from AIFI_L2...')
    
    # Define mapping from AIFI_L2 to major cell types
    cell_type_mapping = {
        # CD4T
        'Memory CD4 T cell': 'CD4T',
        'Naive CD4 T cell': 'CD4T',
        'Treg': 'CD4T',
        
        # CD8T
        'Memory CD8 T cell': 'CD8T',
        'Naive CD8 T cell': 'CD8T',
        'MAIT': 'CD8T',
        'CD8aa': 'CD8T',
        
        # NK
        'CD56bright NK cell': 'NK',
        'CD56dim NK cell': 'NK',
        'Proliferating NK cell': 'NK',
        
        # B cells
        'Memory B cell': 'B',
        'Naive B cell': 'B',
        'Transitional B cell': 'B',
        'Effector B cell': 'B',
        'Plasma cell': 'B',
        
        # Monocytes
        'CD14 monocyte': 'MONO',
        'CD16 monocyte': 'MONO',
        'Intermediate monocyte': 'MONO',
    }
    # Define mapping from AIFI_L2 to standardized sub cell types
    # This maps to the SUB_CTS defined in config.py
    sub_cell_type_mapping = {
        # CD4T subtypes
        'Naive CD4 T cell': 'Tcm_Naive_CD4',
        'Memory CD4 T cell': 'Tem_Effector_CD4',
        'Treg': 'Treg',
        
        # CD8T subtypes
        'Naive CD8 T cell': 'Tcm_Naive_CD8',
        'Memory CD8 T cell': 'Tem_Trm_CD8',
        'MAIT': 'MAIT',
        'CD8aa': 'CD8a/a',
        
        # NK subtypes
        'CD56bright NK cell': 'CD16_NK',
        'CD56dim NK cell': 'CD16_NK',
        'Proliferating NK cell': 'CD16_NK',
        
        # B cell subtypes
        'Naive B cell': 'Naive_B',
        'Memory B cell': 'Memory_B',
        'Transitional B cell': 'Naive_B',
        'Effector B cell': 'Memory_B',
        'Plasma cell': 'Plasma_B',
        
        # Monocyte subtypes
        'CD14 monocyte': 'Classic_MONO',
        'CD16 monocyte': 'NonClassic_MONO',
        'Intermediate monocyte': 'Classic_MONO',
    }
    
    # Store original annotations under _original labels (CellTypist will set MAJOR_CT_LABEL/SUB_CT_LABEL)
    adata.obs['Major_CT_original'] = adata.obs['AIFI_L2'].map(cell_type_mapping)
    adata.obs['Sub_CT_original'] = adata.obs['AIFI_L2'].map(sub_cell_type_mapping)

    # Keep original AIFI_L2 for reference
    adata.obs['AIFI_L2_original'] = adata.obs['AIFI_L2'].astype(str)
    return adata

def remove_attributes(adata, keep_layers=False):
    attrs = ['uns', 'raw', 'obsm', 'varm', 'varp']
    if not keep_layers:
        attrs.append('layers')
    for attr in attrs:
        if hasattr(adata, attr):
            delattr(adata, attr)
    return adata
def _ens2sym(db=ORG_HS_SQLITE):
    """{ENSG: SYMBOL}, first hit per ENSG — matches select() + !duplicated(ENSEMBL)."""
    q = 'select e.ensembl_id, g.symbol from ensembl e join gene_info g on g._id = e._id'
    out = {}
    with sqlite3.connect(db) as con:
        for e, s in con.execute(q):
            out.setdefault(e, s)
    return out


def _make_unique(names):
    """R's make.unique: first occurrence bare, then .1, .2, ..."""
    seen, out = {}, []
    for n in names:
        if n in seen:
            seen[n] += 1
            n = f'{n}.{seen[n]}'
            seen.setdefault(n, 0)
        else:
            seen[n] = 0
        out.append(n)
    return out


def _seurat_counts(adata, keep, chunk=200_000):
    """Seurat nCount_RNA / nFeature_RNA over the kept genes, streamed over backed X."""
    n_count = np.empty(adata.n_obs, dtype=np.float64)
    n_feat = np.empty(adata.n_obs, dtype=np.int32)
    for i in range(0, adata.n_obs, chunk):
        x = adata.X[i:i + chunk][:, keep]
        n_count[i:i + chunk] = np.asarray(x.sum(axis=1)).ravel()
        n_feat[i:i + chunk] = x.getnnz(axis=1)
    return n_count, n_feat


def _raw_var(names):
    """The gene-name column format_data looks up below, plus the index."""
    return pd.DataFrame({'gene_name': names}, index=pd.Index(names, name='gene_name'))


def _curate_onek1k(adata):
    ens = adata.var_names.to_numpy(dtype=str)
    sym = _ens2sym()
    keep = np.array([e in sym for e in ens])

    names = ens.astype(object)  # not <U15: symbols are longer than an ENSG id
    names[keep] = _make_unique([sym[e] for e in ens[keep]])
    adata.var = _raw_var(names)

    n_count, n_feat = _seurat_counts(adata, keep)
    obs = adata.obs
    adata.obs = pd.DataFrame({
        'age': obs['age'].to_numpy(),
        'batch_info': 'Data1_' + obs['pool_number'].astype(str),
        # CMtx strips the spaces from Seurat's Azimuth labels ("CD4 TCM" -> "CD4TCM")
        'ct_major_published': obs['predicted.celltype.l2'].astype(str).str.replace(' ', '', regex=False),
        'donor_id': 'Data1_' + obs['donor_id'].astype(str),
        'nCount_RNA': n_count,
        'nFeature_RNA': n_feat,
        'orig.ident': 'Data1',
        'sex': obs['sex'].astype(str).map({'male': 'M', 'female': 'F'}),
    }, index=obs.index)
    # ponytail: genes with no symbol keep their ENSG id and are dropped by the
    # gene_names.txt mask in script.py — no second backed slice needed here.
    return adata


def _curate_aida(adata):
    obs = adata.obs
    adata.var = _raw_var(adata.var['feature_name'].astype(str).to_numpy())
    # batch_info per donor_id from the AIDA supplementary metadata, same source and mapping
    # as Ali's AIDAv2.ipynb (cells 9-10) -- reproduces the backup's batch_info 1:1 (625/625
    # donors). The barcode suffix ("<library>_L00x") looks similar but is per-library, not
    # per-donor, and does not match the backup.
    metadata = pd.read_excel(f'{RAW_DIR}/AIDA_v2/mmc1.xlsx', sheet_name=0, header=1).rename(
        columns={'DCP_ID': 'donor_id', 'scRNA-seq Experimental Batch': 'batch_info'})
    batch_by_donor = metadata[['donor_id', 'batch_info']].drop_duplicates(
        subset='donor_id').set_index('donor_id')['batch_info']
    adata.obs = pd.DataFrame({
        'age': obs['development_stage'].astype(str).str.extract(r'(\d+)', expand=False).astype(float),
        'batch_info': obs['donor_id'].astype(str).map(batch_by_donor),
        'ct_major_published': obs['author_cell_type'].astype(str),
        'donor_id': obs['donor_id'].astype(str),
        'orig.ident': 'Data13',
        'race': obs['self_reported_ethnicity'].astype(str),
        'sex': obs['sex'].astype(str).map({'male': 'M', 'female': 'F'}),
    }, index=obs.index)
    return adata


_CURATORS = {'onek1k': _curate_onek1k, 'aida': _curate_aida}


def curate_raw(adata, dataset_name):
    """Pinned public download -> the var/obs schema the pipeline expects from a CMtx.

    Provenance: /vol/projects/CIIM/Healthy_Single_Cell_Data/scripts/data_collection.r
    (onek1k) and /vol/projects/aehsani/ImmuneAgeing/.../scripts/AIDAv2.ipynb (aida).
    """
    return _CURATORS[dataset_name](adata)


def format_data(adata, dataset_name):
    # Read from the public download rather than a CIIM count matrix? rebuild the
    # CMtx schema first so everything below is unchanged.
    if str(getattr(adata, 'filename', '') or '') == RAW_SOURCES.get(dataset_name):
        adata = curate_raw(adata, dataset_name)

    config = get_config(dataset_name)
    bulk_group = config.bulk_group
    bulk_group_col = 'bulk_group'
    
    # Soundlife-specific formatting
    if dataset_name == 'soundlife':
        adata = format_columns_soundlife(adata)
        adata.var.index.name = 'gene_name'
        adata.var = adata.var.reset_index()[['gene_name']].set_index('gene_name')

    # ParseBioscience-specific formatting
    elif dataset_name == 'parsebioscience':
        adata = format_columns_parsebioscience(adata)
        adata.var.index.name = 'gene_name'
        adata.var = adata.var.reset_index()[['gene_name']].set_index('gene_name')

    elif dataset_name == 'op':
        adata.obs = adata.obs.rename(columns={'sm_name':'perturbation'})
        adata.obs['is_control'] = adata.obs['perturbation'].isin(['Dimethyl Sulfoxide'])
        adata.obs['is_positive_control'] = adata.obs['perturbation'].isin(['Dabrafenib', 'Belinostat'])
        
        meta = pd.DataFrame({
            "donor_id": ['Donor 1', 'Donor 2', 'Donor 3'],
            "age": [45, 52, 45],
            "sex": ["Female", "Male", "Male"]
        })
        # join metadata into obs
        adata.obs = adata.obs.merge(meta, left_on='donor_id', right_on='donor_id', how='left')
    
    # Handle gene names for datasets that haven't been processed above
    if dataset_name not in ['soundlife', 'parsebioscience']:
        if 'gene_name' in adata.var.columns:
            gene_name = 'gene_name'
        elif 'Gene' in adata.var.columns:
            gene_name = 'Gene'
        elif 'gene_symbols' in adata.var.columns:
            gene_name = 'gene_symbols'
        elif 'feature_name' in adata.var.columns:
            gene_name = 'feature_name'
        elif 'features' in adata.var.columns:
            gene_name = 'features'
        elif dataset_name in ['abf300', 'op']:
            gene_name = 'gene_name'
            adata.var.index.name = gene_name
            adata.var = adata.var.reset_index()
        else:
            print('\n',adata.var)
            raise ValueError("No gene name column found in adata.var")
        adata.var.rename(columns={gene_name: 'gene_name'}, inplace=True)
        # only keep gene_name column
        adata.var =  adata.var[['gene_name']].set_index('gene_name')

    # Gene symbols can collide (e.g. multiple Ensembl IDs -> same symbol); dedup
    # so var_names stays unique for downstream concat/aggregation.
    adata.var_names_make_unique()

    # Common processing for all datasets
    adata.obs = adata.obs.astype('str')
    adata.obs.rename(columns={'perturbation':'condition', 'disease':'condition', 'treatment':'condition'}, inplace=True)
    adata.obs['dataset'] = dataset_name
    adata.obs['donor_age'] = adata.obs['age'].astype(str) + '_' + adata.obs['donor_id'].astype(str)
    adata = remove_attributes(adata, keep_layers=(dataset_name == 'op'))
    adata.obs[bulk_group_col] = adata.obs[bulk_group].astype(str).agg('_'.join, axis=1)
    return adata
def basic_qc(adata, run_test, max_pct_mt=20.0, doublets=False):
    print('Shape before filtering:', adata.shape, flush=True)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    min_genes = 2 if run_test else 100

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_cells(adata, max_genes=5000)
    if not run_test:
        adata = adata[adata.obs['pct_counts_mt'] < max_pct_mt].copy()
    if doublets and not run_test:
        # ponytail: scanpy's scrublet, batched per bulk_group, no extra dependency.
        # Per-group so doublets are simulated within a sample; cross-sample doublets
        # can't exist. Only enabled for CXCL9 (smallest cohort, not pre-QC'd upstream);
        # the demuxed public cohorts already had cross-donor doublets removed.
        sc.pp.scrublet(adata, batch_key='bulk_group')
        n_doublet = int(adata.obs['predicted_doublet'].sum())
        print(f'Scrublet: removing {n_doublet:,} predicted doublets '
              f'({100*n_doublet/adata.n_obs:.1f}%)', flush=True)
        adata = adata[~adata.obs['predicted_doublet']].copy()
    # No gene filtering here: the donor-scaled min_cells made the detection threshold
    # depend on cohort donor count (4.8x spread). Keeping var identical across chunks
    # also makes the ad.concat(join='inner') below exact. Genes are filtered once
    # after concat in script.py; selection proper is left to downstream.
    print('Shape after filtering:', adata.shape, flush=True)
    assert adata.shape[0] > 0, "No cells left after QC filtering."
    return adata


def annotate_celltypist_subsample_knn(adata, subsample_n=150_000, leiden_resolution=5.0):
    """
    Fast CellTypist annotation preserving graph-based local neighborhood structure.

    Strategy (Option 1 — subsample kNN + label transfer):
    1. Subsample ~150k cells, run PCA → kNN graph → Leiden clustering.
    2. Project all remaining cells into the same PCA space; assign each to its
       nearest neighbor's cluster using an approximate kNN index (pynndescent).
    3. Run CellTypist (majority_voting=False) per cluster on small chunks.
    4. Global majority vote per cluster → final labels.

    This preserves the manifold geometry that MiniBatchKMeans misses, at a
    fraction of the memory/time cost of running kNN on the full dataset.
    """
    import scipy.sparse as sp
    import uuid
    import celltypist
    from celltypist import models
    from pynndescent import NNDescent

    print('Annotating cell types (subsample kNN + label transfer + CellTypist)...', flush=True)
    obs_org = adata.obs.copy()

    # --- Back up raw counts ---
    scratch_dir = os.environ.get('HIRA_SCRATCH', '/tmp')
    tmp_counts = os.path.join(scratch_dir, f'counts_backup_{uuid.uuid4().hex}.npz')
    print(f'Backing up raw counts to {tmp_counts}...', flush=True)
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    sp.save_npz(tmp_counts, adata.X.tocsr())

    # --- Normalize + HVG on full data (in-place, sparse, no extra copy) ---
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if adata.X.dtype != np.float32:
        adata.X = adata.X.astype(np.float32)

    # --- Subsample for graph construction ---
    n_sub = min(subsample_n, adata.n_obs)
    rng = np.random.default_rng(0)
    sub_idx = rng.choice(adata.n_obs, size=n_sub, replace=False)
    sub_idx.sort()
    print(f'Subsampling {n_sub:,} cells for kNN+Leiden graph...', flush=True)

    adata_sub = adata[sub_idx].copy()
    sc.pp.highly_variable_genes(adata_sub, n_top_genes=3000, flavor='seurat')
    hvg_mask = adata_sub.var['highly_variable'].values
    adata.var['highly_variable'] = hvg_mask
    hvg_idx = np.where(hvg_mask)[0].astype(np.int32)

    # PCA on subsample via sklearn (stores mean + components for projection)
    from sklearn.decomposition import PCA
    X_sub_hvg = adata_sub[:, hvg_mask].X
    if sp.issparse(X_sub_hvg):
        X_sub_hvg = X_sub_hvg.toarray()
    pca = PCA(n_components=50, random_state=0)
    sub_pca = pca.fit_transform(X_sub_hvg).astype(np.float32)
    del X_sub_hvg

    # kNN + Leiden on subsample PCA embedding
    adata_sub.obsm['X_pca'] = sub_pca
    sc.pp.neighbors(adata_sub, n_neighbors=15, use_rep='X_pca')
    sc.tl.leiden(adata_sub, resolution=leiden_resolution, key_added='over_clustering')
    sub_clusters = adata_sub.obs['over_clustering'].values
    n_clusters = adata_sub.obs['over_clustering'].nunique()
    print(f'Leiden produced {n_clusters} clusters on subsample', flush=True)
    del adata_sub; gc.collect()

    # --- Project ALL cells into PCA space & transfer cluster labels ---
    print('Projecting all cells to PCA space...', flush=True)
    X_hvg_full = adata.X[:, hvg_idx]
    if sp.issparse(X_hvg_full):
        X_hvg_full = X_hvg_full.toarray()
    full_pca = pca.transform(X_hvg_full).astype(np.float32)
    del X_hvg_full, pca; gc.collect()

    # Build approximate kNN index on subsample PCA, query all cells
    print('Building approximate kNN index (pynndescent)...', flush=True)
    index = NNDescent(sub_pca, n_neighbors=15, random_state=0, n_jobs=-1)
    index.prepare()
    print('Querying all cells against subsample index...', flush=True)
    neighbors, _ = index.query(full_pca, k=1)
    del full_pca, index; gc.collect()

    # Each cell inherits the cluster of its nearest subsample neighbor
    all_clusters = pd.Series(sub_clusters[neighbors[:, 0]], index=adata.obs_names, dtype=str)
    # Subsample cells keep their own labels (authoritative)
    all_clusters.iloc[sub_idx] = sub_clusters
    print(f'Cluster label transfer done. {all_clusters.nunique()} unique clusters.', flush=True)

    # --- CellTypist per cluster, then global majority vote ---
    models.download_models(force_update=False)
    model = models.Model.load(model='Immune_All_Low.pkl')

    ct_labels = pd.Series(index=adata.obs_names, dtype=str)
    unique_clusters = all_clusters.unique()
    print(f'Running CellTypist on {len(unique_clusters)} clusters...', flush=True)
    for i, cl in enumerate(unique_clusters):
        idx = np.where(all_clusters.values == cl)[0]
        chunk = adata[idx].copy()
        preds = celltypist.annotate(chunk, model=model, majority_voting=False)
        ct_labels.iloc[idx] = preds.predicted_labels['predicted_labels'].values
        del chunk, preds; gc.collect()
        if (i + 1) % 50 == 0:
            print(f'  CellTypist: {i+1}/{len(unique_clusters)} clusters done', flush=True)

    # Global majority vote per cluster
    print('Running global majority voting...', flush=True)
    tmp_df = pd.DataFrame({'cluster': all_clusters, 'ct': ct_labels})
    majority = tmp_df.groupby('cluster')['ct'].agg(lambda x: x.value_counts().index[0])
    majority_labels = all_clusters.map(majority)

    mapping = {
        'Tcm/Naive helper T cells': 'CD4T',
        'CD16+ NK cells': 'NK',
        'Classical monocytes': 'MONO',
        'Tem/Temra cytotoxic T cells': 'CD8T',
        'Tem/Effector helper T cells': 'CD4T',
        'Tcm/Naive cytotoxic T cells': 'CD8T',
        'B cells': 'B',
        'Naive B cells': 'B',
        'Tem/Trm cytotoxic T cells': 'CD8T',
        'Memory B cells': 'B',
        'Non-classical monocytes': 'MONO',
        'MAIT cells': 'CD8T',
        'Regulatory T cells': 'CD4T',
        'Cycling T cells': 'CD4T',
        'DC2': 'DC',
        'pDC': 'DC',
        'Intermediate macrophages': 'MONO',
        'NK cells': 'NK',
        'Plasma cells': 'B',
        'HSC/MPP': 'HSC',
        'Age-associated B cells': 'B',
        'DC1': 'DC',
        'Megakaryocytes/platelets': 'Megakaryocyte',
        'Plasmablasts': 'B',
        'ILC': 'ILC',
        'CD8a/a': 'CD8T',
        'Double-positive thymocytes': 'T',
        'Late erythroid': 'Erythroid',
    }
    mapping_sub = {
        'Tcm/Naive helper T cells': 'Tcm_Naive_CD4',
        'CD16+ NK cells': 'CD16_NK',
        'Classical monocytes': 'Classic_MONO',
        'Tem/Temra cytotoxic T cells': 'Tem_Temra_CD8',
        'Tem/Effector helper T cells': 'Tem_Effector_CD4',
        'Tcm/Naive cytotoxic T cells': 'Tcm_Naive_CD8',
        'Naive B cells': 'Naive_B',
        'Tem/Trm cytotoxic T cells': 'Tem_Trm_CD8',
        'Memory B cells': 'Memory_B',
        'B cells': 'Bcells',
        'Non-classical monocytes': 'NonClassic_MONO',
        'MAIT cells': 'MAIT',
        'Regulatory T cells': 'Treg',
        'DC2': 'DC2',
        'pDC': 'pDC',
        'Intermediate macrophages': 'Int_Macrophage',
        'NK cells': 'NK',
        'Plasma cells': 'Plasma_B',
        'HSC/MPP': 'HSC/MPP',
        'Age-associated B cells': 'Aged_B',
        'DC1': 'DC1',
        'Megakaryocytes/platelets': 'Platelet',
        'Plasmablasts': 'Plasmablasts_B',
        'ILC': 'ILC',
        'CD8a/a': 'CD8a/a',
        'Double-positive thymocytes': 'T',
        'Late erythroid': 'Late_Erythroid',
    }
    obs_org[MAJOR_CT_LABEL] = majority_labels.map(mapping).fillna('Others').values
    obs_org[SUB_CT_LABEL]   = majority_labels.map(mapping_sub).fillna('Others').values

    adata.obs = obs_org
    adata.X = sp.load_npz(tmp_counts)
    os.remove(tmp_counts)
    return adata


def annotate_celltypist_fast(adata, n_clusters=500):
    """
    Memory-efficient CellTypist annotation via donor-chunked processing.

    Processing is done one donor at a time to avoid holding the full normalized
    matrix (e.g. 13M x 40k float32) in memory.  The key insight is that
    MiniBatchKMeans.partial_fit is incremental, so we can train across all donors
    without ever materialising a full normalised dataset.  CellTypist prediction
    (majority_voting=False) is also run per donor chunk, and global majority voting
    is performed at the end over the shared cluster labels.

    Memory footprint at any point: one donor's normalised slice + the cluster
    centre matrix (n_clusters x n_hvg x float32 ~ negligible).
    """
    import scipy.sparse as sp
    from sklearn.cluster import MiniBatchKMeans
    import uuid
    import celltypist
    from celltypist import models

    print('Annotating cell types (donor-chunked MiniBatchKMeans + CellTypist)...', flush=True)
    obs_org = adata.obs.copy()

    # --- Back up raw counts to scratch disk ---
    scratch_dir = os.environ.get('HIRA_SCRATCH', '/tmp')
    tmp_counts = os.path.join(scratch_dir, f'counts_backup_{uuid.uuid4().hex}.npz')
    print(f'Backing up raw counts to {tmp_counts}...', flush=True)
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    sp.save_npz(tmp_counts, adata.X.tocsr())

    # --- Detect donor column (first available from standard names) ---
    donor_col = next((c for c in ['donor_id', 'donor', 'sample', 'batch_info'] if c in adata.obs.columns), None)
    if donor_col is None:
        raise ValueError("No donor column found in adata.obs; cannot chunk by donor.")
    donors = adata.obs[donor_col].unique().tolist()
    print(f'Processing {len(donors)} donors as chunks...', flush=True)

    # --- Pass 1: compute global HVGs on a random subsample (memory-cheap) ---
    # Use at most 200k cells to determine HVGs, spread across donors.
    subsample_per_donor = max(1, min(200_000 // len(donors), 5_000))
    sub_indices = []
    for d in donors:
        idx = np.where(adata.obs[donor_col].values == d)[0]
        chosen = idx[:subsample_per_donor]
        sub_indices.append(chosen)
    sub_indices = np.concatenate(sub_indices)
    adata_sub = adata[sub_indices].copy()
    sc.pp.normalize_total(adata_sub, target_sum=1e4)
    sc.pp.log1p(adata_sub)
    sc.pp.highly_variable_genes(adata_sub, n_top_genes=3000, flavor='seurat')
    hvg_mask = adata_sub.var['highly_variable'].values
    hvg_cols = np.where(hvg_mask)[0].astype(np.int32)
    # Copy HVG flags back to main adata.var so downstream steps have them
    adata.var['highly_variable'] = hvg_mask
    del adata_sub; gc.collect()
    print(f'HVGs determined: {len(hvg_cols)} genes', flush=True)

    # --- Pass 2: incremental MiniBatchKMeans fit (per donor batch) ---
    n_clusters_actual = min(n_clusters, adata.n_obs)
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters_actual, random_state=0, n_init=3,
        batch_size=10_000, max_iter=100,
    )

    # Group donors into batches: accumulate until we have >= n_clusters_actual cells.
    # This ensures each partial_fit call has enough samples, even in test mode.
    def _donor_batches(donors, obs_donor_col, min_cells):
        batch, batch_size = [], 0
        for d in donors:
            n = int((obs_donor_col == d).sum())
            batch.append(d)
            batch_size += n
            if batch_size >= min_cells:
                yield batch
                batch, batch_size = [], 0
        if batch:
            yield batch

    for i, batch_donors in enumerate(_donor_batches(donors, adata.obs[donor_col], n_clusters_actual)):
        idx = np.where(adata.obs[donor_col].isin(batch_donors).values)[0]
        chunk = adata[idx].copy()
        sc.pp.normalize_total(chunk, target_sum=1e4)
        sc.pp.log1p(chunk)
        X_hvg = chunk.X[:, hvg_cols].tocsr().astype(np.float32)
        if X_hvg.indices.dtype != np.int32:
            X_hvg.indices = X_hvg.indices.astype(np.int32)
            X_hvg.indptr  = X_hvg.indptr.astype(np.int32)
        kmeans.partial_fit(X_hvg)
        del chunk, X_hvg; gc.collect()
    print(f'  MiniBatchKMeans fit done ({i+1} batches)', flush=True)

    # --- Pass 3: predict cluster labels + CellTypist per donor batch ---
    models.download_models(force_update=False)
    model = models.Model.load(model='Immune_All_Low.pkl')

    cluster_labels = pd.Series(index=adata.obs_names, dtype=str)
    ct_labels      = pd.Series(index=adata.obs_names, dtype=str)

    for i, batch_donors in enumerate(_donor_batches(donors, adata.obs[donor_col], n_clusters_actual)):
        idx = np.where(adata.obs[donor_col].isin(batch_donors).values)[0]
        chunk = adata[idx].copy()
        sc.pp.normalize_total(chunk, target_sum=1e4)
        sc.pp.log1p(chunk)

        # Cluster labels
        X_hvg = chunk.X[:, hvg_cols].tocsr().astype(np.float32)
        if X_hvg.indices.dtype != np.int32:
            X_hvg.indices = X_hvg.indices.astype(np.int32)
            X_hvg.indptr  = X_hvg.indptr.astype(np.int32)
        labels = kmeans.predict(X_hvg).astype(str)
        cluster_labels.iloc[idx] = labels
        del X_hvg; gc.collect()

        # CellTypist prediction (no majority voting yet — done globally below)
        preds = celltypist.annotate(chunk, model=model, majority_voting=False)
        ct_labels.iloc[idx] = preds.predicted_labels['predicted_labels'].values
        del chunk, preds; gc.collect()

        if (i + 1) % 10 == 0 or i == 0:
            print(f'  CellTypist predict: batch {i+1} done', flush=True)

    # --- Global majority voting: per cluster, pick the most common CT label ---
    print('Running global majority voting...', flush=True)
    tmp_df = pd.DataFrame({'cluster': cluster_labels, 'ct': ct_labels})
    majority = tmp_df.groupby('cluster')['ct'].agg(lambda x: x.value_counts().index[0])
    majority_labels = cluster_labels.map(majority)
    print('Cell types annotated successfully!', flush=True)

    # --- Apply Major/Sub CT mappings ---
    mapping = {
        'Tcm/Naive helper T cells': 'CD4T',
        'CD16+ NK cells': 'NK',
        'Classical monocytes': 'MONO',
        'Tem/Temra cytotoxic T cells': 'CD8T',
        'Tem/Effector helper T cells': 'CD4T',
        'Tcm/Naive cytotoxic T cells': 'CD8T',
        'B cells': 'B',
        'Naive B cells': 'B',
        'Tem/Trm cytotoxic T cells': 'CD8T',
        'Memory B cells': 'B',
        'Non-classical monocytes': 'MONO',
        'MAIT cells': 'CD8T',
        'Regulatory T cells': 'CD4T',
        'Cycling T cells': 'CD4T',
        'DC2': 'DC',
        'pDC': 'DC',
        'Intermediate macrophages': 'MONO',
        'NK cells': 'NK',
        'Plasma cells': 'B',
        'HSC/MPP': 'HSC',
        'Age-associated B cells': 'B',
        'DC1': 'DC',
        'Megakaryocytes/platelets': 'Megakaryocyte',
        'Plasmablasts': 'B',
        'ILC': 'ILC',
        'CD8a/a': 'CD8T',
        'Double-positive thymocytes': 'T',
        'Late erythroid': 'Erythroid',
    }
    mapping_sub = {
        'Tcm/Naive helper T cells': 'Tcm_Naive_CD4',
        'CD16+ NK cells': 'CD16_NK',
        'Classical monocytes': 'Classic_MONO',
        'Tem/Temra cytotoxic T cells': 'Tem_Temra_CD8',
        'Tem/Effector helper T cells': 'Tem_Effector_CD4',
        'Tcm/Naive cytotoxic T cells': 'Tcm_Naive_CD8',
        'Naive B cells': 'Naive_B',
        'Tem/Trm cytotoxic T cells': 'Tem_Trm_CD8',
        'Memory B cells': 'Memory_B',
        'B cells': 'Bcells',
        'Non-classical monocytes': 'NonClassic_MONO',
        'MAIT cells': 'MAIT',
        'Regulatory T cells': 'Treg',
        'DC2': 'DC2',
        'pDC': 'pDC',
        'Intermediate macrophages': 'Int_Macrophage',
        'NK cells': 'NK',
        'Plasma cells': 'Plasma_B',
        'HSC/MPP': 'HSC/MPP',
        'Age-associated B cells': 'Aged_B',
        'DC1': 'DC1',
        'Megakaryocytes/platelets': 'Platelet',
        'Plasmablasts': 'Plasmablasts_B',
        'ILC': 'ILC',
        'CD8a/a': 'CD8a/a',
        'Double-positive thymocytes': 'T',
        'Late erythroid': 'Late_Erythroid',
    }
    obs_org[MAJOR_CT_LABEL] = majority_labels.map(mapping).fillna('Others').values
    obs_org[SUB_CT_LABEL]   = majority_labels.map(mapping_sub).fillna('Others').values

    # --- Restore raw counts ---
    adata.obs = obs_org
    adata.X = sp.load_npz(tmp_counts)
    os.remove(tmp_counts)
    return adata


def _annotate_celltypist_majority_voting(adata):
    """CellTypist annotation using its built-in majority_voting=True (simpler, for smaller datasets)."""
    import celltypist
    from celltypist import models

    print('Annotating cell types (CellTypist majority_voting=True)...')
    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    models.download_models(force_update=False)
    model = models.Model.load(model='Immune_All_Low.pkl')
    obs_org = adata.obs.copy()
    predictions = celltypist.annotate(adata, model=model, majority_voting=True, use_GPU=False)
    print('Cell types annotated successfully!')
    adata_for_celltypist = predictions.to_adata()

    mapping = {
        'Tcm/Naive helper T cells': 'CD4T',
        'CD16+ NK cells': 'NK',
        'Classical monocytes': 'MONO',
        'Tem/Temra cytotoxic T cells': 'CD8T',
        'Tem/Effector helper T cells': 'CD4T',
        'Tcm/Naive cytotoxic T cells': 'CD8T',
        'B cells': 'B',
        'Naive B cells': 'B',
        'Tem/Trm cytotoxic T cells': 'CD8T',
        'Memory B cells': 'B',
        'Non-classical monocytes': 'MONO',
        'MAIT cells': 'CD8T',
        'Regulatory T cells': 'CD4T',
        'Cycling T cells': 'CD4T',
        'DC2': 'DC',
        'pDC': 'DC',
        'Intermediate macrophages': 'MONO',
        'NK cells': 'NK',
        'Plasma cells': 'B',
        'HSC/MPP': 'HSC',
        'Age-associated B cells': 'B',
        'DC1': 'DC',
        'Megakaryocytes/platelets': 'Megakaryocyte',
        'Plasmablasts': 'B',
        'ILC': 'ILC',
        'CD8a/a': 'CD8T',
        'Double-positive thymocytes': 'T',
        'Late erythroid': 'Erythroid',
    }
    mapping_sub = {
        'Tcm/Naive helper T cells': 'Tcm_Naive_CD4',
        'CD16+ NK cells': 'CD16_NK',
        'Classical monocytes': 'Classic_MONO',
        'Tem/Temra cytotoxic T cells': 'Tem_Temra_CD8',
        'Tem/Effector helper T cells': 'Tem_Effector_CD4',
        'Tcm/Naive cytotoxic T cells': 'Tcm_Naive_CD8',
        'Naive B cells': 'Naive_B',
        'Tem/Trm cytotoxic T cells': 'Tem_Trm_CD8',
        'Memory B cells': 'Memory_B',
        'B cells': 'Bcells',
        'Non-classical monocytes': 'NonClassic_MONO',
        'MAIT cells': 'MAIT',
        'Regulatory T cells': 'Treg',
        'DC2': 'DC2',
        'pDC': 'pDC',
        'Intermediate macrophages': 'Int_Macrophage',
        'NK cells': 'NK',
        'Plasma cells': 'Plasma_B',
        'HSC/MPP': 'HSC/MPP',
        'Age-associated B cells': 'Aged_B',
        'DC1': 'DC1',
        'Megakaryocytes/platelets': 'Platelet',
        'Plasmablasts': 'Plasmablasts_B',
        'ILC': 'ILC',
        'CD8a/a': 'CD8a/a',
        'Double-positive thymocytes': 'T',
        'Late erythroid': 'Late_Erythroid',
    }
    adata_for_celltypist.obs[MAJOR_CT_LABEL] = adata_for_celltypist.obs['majority_voting'].apply(lambda x: mapping.get(x, 'Others'))
    adata_for_celltypist.obs[SUB_CT_LABEL] = adata_for_celltypist.obs['majority_voting'].apply(lambda x: mapping_sub.get(x, 'Others'))
    obs_org = obs_org.join(adata_for_celltypist.obs[[MAJOR_CT_LABEL, SUB_CT_LABEL]])
    adata.obs = obs_org
    adata.X = adata.layers['counts']
    del adata.layers
    return adata


def annotate_celltypes(adata, dataset):
    if dataset == 'soundlife':
        print('Using original AIFI_L2 annotations for soundlife (skipping CellTypist)...')
        adata = map_cell_types_soundlife(adata)
        adata.obs[MAJOR_CT_LABEL] = adata.obs['Major_CT_original']
        adata.obs[SUB_CT_LABEL] = adata.obs['Sub_CT_original']
        return adata
    elif dataset == 'parsebioscience':
        print('Using original cell type annotations for parsebioscience (skipping CellTypist)...')
        adata = map_cell_types_parsebioscience(adata)
        adata.obs[MAJOR_CT_LABEL] = adata.obs['Major_CT_original']
        adata.obs[SUB_CT_LABEL] = adata.obs['Sub_CT_original']
        return adata

    # Env var override: allows testing alternative methods without code changes
    method = os.environ.get('HIRA_ANNOTATE_METHOD', '')
    if method == 'subsample_knn':
        print(f'Using subsample kNN + label transfer annotation (HIRA_ANNOTATE_METHOD=subsample_knn).')
        return annotate_celltypist_subsample_knn(adata)
    if method == 'majority_voting':
        print(f'Using CellTypist majority_voting=True (HIRA_ANNOTATE_METHOD=majority_voting).')
        return _annotate_celltypist_majority_voting(adata)

    return _annotate_celltypist_majority_voting(adata)

def format_columns_parsebioscience(adata):
    """
    Add standardized column names while keeping all original columns.
    Maps ParseBioscience-specific columns to standard pipeline format.
    This is called BEFORE cell type annotation.
    """
    print('Formatting columns for parsebioscience...')
    
    # Basic metadata formatting
    adata.obs['is_control'] = adata.obs['treatment'] == 'PBS'
    adata.obs = adata.obs[['cell_type', 'cytokine', 'donor', 'is_control', 'bc1_well']]
    adata.obs = adata.obs.rename({'donor': 'donor_id', 'cytokine': 'condition', 'bc1_well': 'well'}, axis=1)
    
    # Store original cell type annotation for later mapping
    adata.obs['cell_type_original'] = adata.obs['cell_type'].astype(str)
    
    # Additional metadata
    adata.obs['perturbation_type'] = 'cytokine'

    # Create age mapping from the donor information
    donor_age_map = {
        'Donor1': 75,
        'Donor2': 34,
        'Donor3': 68,
        'Donor4': 59,
        'Donor5': 41,
        'Donor6': 38,
        'Donor7': 45,
        'Donor8': 52,
        'Donor9': 38,
        'Donor10': 42,
        'Donor11': 46,
        'Donor12': 36
    }

    # Map the age to obs based on donor_id
    adata.obs['age'] = adata.obs['donor_id'].map(donor_age_map)
    
    return adata

def map_cell_types_parsebioscience(adata):
    """
    Map ParseBioscience annotations to major cell types and standardized sub cell types.
    Sets cell_type (Major_CT) and Sub_CT columns with standardized nomenclature.
    This is called AFTER basic formatting, similar to soundlife.
    """
    print('Mapping cell types from ParseBioscience original annotations...')
    
    # Define mapping to major cell types
    major_cell_type_map = {
        'B Intermediate/Memory': 'B',
        'B Naive': 'B',
        'CD14 Mono': 'MONO',
        'CD16 Mono': 'MONO',
        'CD4 Memory': 'CD4T',
        'CD4 Naive': 'CD4T',
        'Treg': 'CD4T',
        'CD8 Memory': 'CD8T',
        'CD8 Naive': 'CD8T',
        'NK': 'NK',
        'NK CD56bright': 'NK',
        'NKT': 'NK',
        'MAIT': 'CD8T',
        'ILC': 'NK',
        'Plasmablast': 'B',
        'HSPC': None,
        'cDC': None,
        'pDC': None
    }
    
    # Define mapping to standardized sub cell types (matching config.py SUB_CTS)
    sub_cell_type_map = {
        'B Intermediate/Memory': 'Memory_B',
        'B Naive': 'Naive_B',
        'CD14 Mono': 'Classic_MONO',
        'CD16 Mono': 'NonClassic_MONO',
        'CD4 Memory': 'Tem_Effector_CD4',
        'CD4 Naive': 'Tcm_Naive_CD4',
        'Treg': 'Treg',
        'CD8 Memory': 'Tem_Trm_CD8',
        'CD8 Naive': 'Tcm_Naive_CD8',
        'NK': 'CD16_NK',
        'NK CD56bright': 'CD16_NK',
        'NKT': 'CD16_NK',
        'MAIT': 'MAIT',
        'ILC': 'CD16_NK',
        'Plasmablast': 'Plasmablasts_B',
        'HSPC': None,
        'cDC': None,
        'pDC': None
    }
    
    # Map major and sub cell types to _original columns (CellTypist will set MAJOR_CT_LABEL/SUB_CT_LABEL)
    adata.obs['Major_CT_original'] = adata.obs['cell_type_original'].map(major_cell_type_map)
    adata.obs['Sub_CT_original'] = adata.obs['cell_type_original'].map(sub_cell_type_map)
    adata.obs['cell_type'] = adata.obs['Major_CT_original']
    
    # Count unmapped cells
    unmapped_major = adata.obs['Major_CT_original'].isna().sum()
    unmapped_sub = adata.obs['Sub_CT_original'].isna().sum()
    total = adata.shape[0]
    
    print(f'Unmapped major cell types: {unmapped_major:,} ({unmapped_major/total*100:.2f}%)')
    print(f'Unmapped sub cell types: {unmapped_sub:,} ({unmapped_sub/total*100:.2f}%)')
    
    # Filter out unmapped cell types (use .copy() to avoid view issues in backed mode)
    adata = adata[~adata.obs['cell_type'].isna(), :].copy()
    print(f'Shape after filtering unmapped cell types: {adata.shape}')
    
    # Show cell type distribution
    print('\nOriginal major cell type distribution:')
    print(adata.obs['Major_CT_original'].value_counts())
    
    print(f'\nOriginal sub cell type distribution:')
    print(adata.obs['Sub_CT_original'].value_counts())
    
    return adata

def _selfcheck_curation():
    """`python -m hira.src.process_data.preprocess.helper` — curate_raw vs the CIIM CMtx."""
    import h5py

    def col(grp, k):
        o = grp[k]
        return o['categories'][:].astype(str)[o['codes'][:]] if hasattr(o, 'keys') else o[:]

    for dataset, key in {'onek1k': 'data1', 'aida': 'data13'}.items():
        adata = curate_raw(ad.read_h5ad(RAW_SOURCES[dataset], backed='r'), dataset)
        with h5py.File(os.path.join(os.path.dirname(RAW_DIR), 'count_matrix',
                                    f'{key}_CMtx.h5ad')) as f:
            want_var = col(f['var'], f['var'].attrs['_index']).astype(str)
            want = {k: col(f['obs'], k) for k in f['obs'] if not k.startswith('_')}
        got_var = adata.var_names.to_numpy(dtype=str)
        if dataset == 'onek1k':  # unmapped genes keep their ENSG id, dropped downstream
            got_var = np.array([s for s in got_var if not s.startswith('ENSG')])
        assert np.array_equal(got_var, want_var), f'{dataset}: var mismatch'
        for k, w in want.items():
            g = adata.obs[k].to_numpy()
            ok = np.allclose(g.astype(float), w.astype(float)) if w.dtype.kind in 'if' \
                else np.array_equal(g.astype(str), w.astype(str))
            assert ok, f'{dataset}: {k} mismatch, {g[:3]} vs {w[:3]}'
        print(f'{dataset}: ok, {len(got_var)} genes, {adata.n_obs} cells, '
              f'{len(want)} obs columns')


if __name__ == '__main__':
    _selfcheck_curation()
