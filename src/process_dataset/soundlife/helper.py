import os
default_n_threads = 3
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
import gc

def format_columns(adata):
    """
    Add standardized column names while keeping all original columns.
    Maps SoundLife-specific columns to standard pipeline format.
    """
    print('Formatting columns...')
    
    # Add standardized columns (keep originals)
    adata.obs['donor_id'] = adata.obs['subject.subjectGuid'].astype(str)
    adata.obs['age'] = pd.to_numeric(adata.obs['sample.subjectAgeAtDraw'], errors='coerce')
    adata.obs['race'] = adata.obs['subject.ethnicity'].astype(str)
    adata.obs['sex'] = adata.obs['subject.biologicalSex'].astype(str)
    adata.obs['visitName'] = adata.obs['sample.visitName'].astype(str)
    
    # Map CMV status to condition
    adata.obs['condition'] = adata.obs['subject.cmv'].apply(
        lambda x: 'healthy' if x == 'Negative' else 'CMV'
    )
    
    # Add dataset identifier
    adata.obs['dataset'] = 'soundlife'
    
    # Create donor_age identifier (donor_id + age)
    adata.obs['donor_age'] = adata.obs['donor_id'].astype(str) + '_' + adata.obs['age'].astype(str)
    
    print(f'Added standardized columns. Total columns: {len(adata.obs.columns)}')
    
    return adata


def map_cell_types(adata):
    """
    Map AIFI_L2 annotations to major cell types and filter out unmapped types.
    Sets cell_type (Major_CT) and Sub_CT columns.
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
    
    # Map cell types
    adata.obs['cell_type'] = adata.obs['AIFI_L2'].map(cell_type_mapping)
    
    # Keep original AIFI_L2 as Sub_CT
    adata.obs['Sub_CT'] = adata.obs['AIFI_L2'].astype(str)
    
    # Also create Major_CT for consistency with other datasets
    adata.obs['Major_CT'] = adata.obs['cell_type']
    
    # Count unmapped cells
    unmapped = adata.obs['cell_type'].isna().sum()
    total = adata.shape[0]
    print(f'Unmapped cells: {unmapped:,} ({unmapped/total*100:.2f}%)')
    
    # Filter out unmapped cell types
    adata = adata[~adata.obs['cell_type'].isna()].copy()
    print(f'Shape after filtering unmapped cell types: {adata.shape}')
    
    # Show cell type distribution
    print('\nCell type distribution:')
    print(adata.obs['cell_type'].value_counts())
    
    return adata


def qc_check(adata):
    """
    Apply basic quality control filtering.
    Adapted from existing QC pipeline.
    """
    print('Applying QC filters...')
    print(f'Shape before filtering: {adata.shape}')
    
    # Calculate QC metrics
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, 
        qc_vars=['mt'], 
        percent_top=None, 
        log1p=False, 
        inplace=True
    )
    
    # Filter cells
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_cells(adata, max_genes=5000)
    
    # Filter genes
    # Consider number of donors for min_cells threshold
    n_donors = adata.obs['donor_id'].nunique()
    min_cells_per_donor = 10
    min_cells = int(n_donors * min_cells_per_donor)
    min_cells = max(min_cells, 10)
    
    print(f'Using min_cells={min_cells} for gene filtering (based on {n_donors} donors)')
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_genes(adata, min_counts=1)
    
    print(f'Shape after filtering: {adata.shape}')
    
    return adata


def process_single_file(file_path, output_path, test_mode=False):
    """
    Process a single SoundLife .h5ad file:
    1. Read data
    2. Format columns
    3. Apply QC
    4. Map cell types
    5. Pseudobulk
    6. Save
    
    Args:
        file_path: Path to input .h5ad file
        output_path: Path to save pseudobulked output
        test_mode: If True, subset data to 2 donors and 2 visits for testing
    """
    import time
    
    print(f'\n{"="*80}')
    print(f'Processing: {os.path.basename(file_path)}')
    print(f'{"="*80}')
    
    # Read data
    start_time = time.time()
    print('Reading data... (this may take several minutes for large files)')
    adata = ad.read_h5ad(file_path)
    elapsed = time.time() - start_time
    print(f'Loaded shape: {adata.shape} (took {elapsed/60:.1f} minutes)')
    
    # Calculate memory usage (handle sparse matrices)
    if hasattr(adata.X, 'data'):  # Sparse matrix
        mem_usage = adata.X.data.nbytes / 1e9
    else:  # Dense matrix
        mem_usage = adata.X.nbytes / 1e9
    print(f'Memory usage: ~{mem_usage:.2f} GB')
    
    # TEST MODE: Subset to small data for quick testing
    if test_mode:
        print('\n' + '='*80)
        print('*** TEST MODE ENABLED: Subsetting data ***')
        print('='*80)
        
        # Get 2 unique donors
        unique_donors = adata.obs['subject.subjectGuid'].unique()[:2]
        print(f'Selected donors: {unique_donors.tolist()}')
        
        # Get 2 unique visits
        unique_visits = adata.obs['sample.visitName'].unique()[:2]
        print(f'Selected visits: {unique_visits.tolist()}')
        
        # Subset to these donors and visits
        mask = (adata.obs['subject.subjectGuid'].isin(unique_donors)) & \
               (adata.obs['sample.visitName'].isin(unique_visits))
        
        adata = adata[mask].copy()
        print(f'Subsetted shape: {adata.shape}')
        print('='*80)
        print('*** END TEST MODE SUBSET ***')
        print('='*80 + '\n')
    
    # Format columns
    adata = format_columns(adata)
    
    # Apply QC
    adata = qc_check(adata)
    
    # Map cell types and filter
    adata = map_cell_types(adata)
    
    # Store raw counts in layers
    print('Storing raw counts in layers...')
    adata.layers['counts'] = adata.X.copy()
    
    # Pseudobulk
    print('Pseudobulking...')
    from task_grn_inference.src.process_data.helper_data import sum_by
    
    # Create sum_by column
    covariates = ['cell_type', 'donor_id', 'visitName']
    adata.obs['sum_by'] = ''
    for covariate in covariates:
        adata.obs['sum_by'] += '_' + adata.obs[covariate].astype(str)
    adata.obs['sum_by'] = adata.obs['sum_by'].astype('category')
    
    # Calculate cell counts BEFORE pseudobulking
    cell_count_df = adata.obs.groupby('sum_by').size().reset_index(name='cell_count')
    
    # Perform pseudobulking
    adata_bulk = sum_by(adata, 'sum_by', unique_mapping=True)
    print(f'After sum_by, bulk shape: {adata_bulk.shape}')
    
    # Check if cell_count is already present (sum_by function adds it with unique_mapping=True)
    if 'cell_count' not in adata_bulk.obs.columns:
        # If not present, merge it
        adata_bulk.obs = adata_bulk.obs.reset_index()
        adata_bulk.obs = adata_bulk.obs.rename(columns={'index': 'sum_by_index'})
        adata_bulk.obs = adata_bulk.obs.merge(cell_count_df, left_on='sum_by', right_on='sum_by', how='left')
    
    # Filter by cell count threshold
    cell_count_t = 10
    print(f'Filtering bulk samples with < {cell_count_t} cells')
    low_cells = adata_bulk.obs['cell_count'] < cell_count_t
    print(f'Dropping {low_cells.sum()} bulk samples with less than {cell_count_t} cells')
    adata_bulk = adata_bulk[~low_cells].copy()
    
    print(f'Pseudobulked shape: {adata_bulk.shape}')
    print(f'Number of bulk samples: {adata_bulk.n_obs}')
    
    # Save individual pseudobulked file
    print(f'Saving to: {output_path}')
    adata_bulk.write(output_path)
    
    # Clean up memory
    del adata
    del adata_bulk
    gc.collect()
    
    print(f'Successfully processed and saved {file_path}')
    
    return output_path
