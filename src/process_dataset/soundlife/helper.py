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
from hiara.src.config import get_config
from hiara.src.process_dataset.preprocess.helper import basic_qc

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
    
    # Map age_group from subject.ageGroup
    # "Sound Life Young Adult" -> "young"
    # "Sound Life Older Adult" -> "old"
    adata.obs['age_group'] = adata.obs['subject.ageGroup'].apply(
        lambda x: 'young' if 'Young' in str(x) else ('old' if 'Older' in str(x) else None)
    )
    
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
    
    # Assign vaccinated column (0 or 1)
    adata.obs['vaccinated'] = parsed['vaccinated']
    
    print(f'Added standardized columns. Total columns: {len(adata.obs.columns)}')
    print(f'Age group distribution:')
    print(adata.obs['age_group'].value_counts(dropna=False))
    print(f'Vaccinated distribution:')
    print(adata.obs['vaccinated'].value_counts(dropna=False))
    print(f'Year distribution:')
    print(adata.obs['year'].value_counts(dropna=False))
    
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


def create_metacells(adata, metacell_size=100):
    """
    Create metacells by splitting each group into chunks of metacell_size cells.
    This increases sample size by creating multiple pseudobulk samples per group.
    
    Args:
        adata: AnnData object with raw counts in layers['counts']
        metacell_size: Number of cells to aggregate per metacell
        
    Returns:
        adata_metacell: AnnData with metacell pseudobulk data
    """
    print(f'Creating metacells with size {metacell_size}...')
    
    from task_grn_inference.src.process_data.helper_data import sum_by
    
    # Create grouping by cell_type, donor_id, visitName (same as regular bulk)
    covariates = ['cell_type', 'donor_id', 'visitName']
    adata.obs['group_id'] = ''
    for covariate in covariates:
        adata.obs['group_id'] += '_' + adata.obs[covariate].astype(str)
    
    # Within each group, assign metacell IDs
    metacell_ids = []
    for group_name, group_df in adata.obs.groupby('group_id'):
        n_cells = len(group_df)
        # Create metacell indices (0, 0, 0, ..., 1, 1, 1, ..., 2, 2, 2, ...)
        metacell_indices = np.arange(n_cells) // metacell_size
        metacell_ids.extend(metacell_indices)
    
    adata.obs['metacell_id'] = metacell_ids
    
    # Create unique identifier for each metacell
    adata.obs['metacell_sum_by'] = adata.obs['group_id'] + '_mc' + adata.obs['metacell_id'].astype(str)
    adata.obs['metacell_sum_by'] = adata.obs['metacell_sum_by'].astype('category')
    
    # Calculate cell counts per metacell
    cell_count_df = adata.obs.groupby('metacell_sum_by').size().reset_index(name='cell_count')
    
    # Perform pseudobulking per metacell
    adata_metacell = sum_by(adata, 'metacell_sum_by', unique_mapping=True)
    print(f'After metacell sum_by, shape: {adata_metacell.shape}')
    
    # Merge cell count if not already present
    if 'cell_count' not in adata_metacell.obs.columns:
        adata_metacell.obs = adata_metacell.obs.reset_index()
        adata_metacell.obs = adata_metacell.obs.rename(columns={'index': 'sum_by_index'})
        adata_metacell.obs = adata_metacell.obs.merge(
            cell_count_df, 
            left_on='metacell_sum_by', 
            right_on='metacell_sum_by', 
            how='left'
        )
    
    # Filter by cell count threshold (keep metacells with at least 100 cells)
    # This allows the last metacell in each group to be kept if it has at least 100 cells
    cell_count_t = 100
    print(f'Filtering metacells with < {cell_count_t} cells')
    low_cells = adata_metacell.obs['cell_count'] < cell_count_t
    print(f'Dropping {low_cells.sum()} metacells with less than {cell_count_t} cells')
    adata_metacell = adata_metacell[~low_cells].copy()
    
    print(f'Final metacell shape: {adata_metacell.shape}')
    print(f'Number of metacells: {adata_metacell.n_obs}')
    
    return adata_metacell


def process_single_file(file_path, output_path, test_mode=False, output_path_metacell=None):
    """
    Process a single SoundLife .h5ad file:
    1. Read data
    2. Format columns
    3. Apply QC
    4. Map cell types
    5. Pseudobulk (regular bulk and metacell)
    6. Save
    
    Args:
        file_path: Path to input .h5ad file
        output_path: Path to save pseudobulked output
        test_mode: If True, subset data to 2 donors and 2 visits for testing
        output_path_metacell: Path to save metacell pseudobulked output (optional)
    """
    import time
    cfg = get_config('soundlife')
    pseudobulk_group = cfg.pseudobulk_group
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
    adata = basic_qc(adata)
    
    # Map cell types and filter
    adata = map_cell_types(adata)
    
    # Store raw counts in layers
    print('Storing raw counts in layers...')
    adata.layers['counts'] = adata.X.copy()
    
    # Pseudobulk
    print('Pseudobulking...')
    from task_grn_inference.src.process_data.helper_data import sum_by
    
    # Create sum_by column
    covariates = pseudobulk_group
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
    print(f'Saving regular bulk to: {output_path}')
    adata_bulk.write(output_path)
    
    # Generate metacell pseudobulk if output path is provided
    if output_path_metacell is not None:
        print('\n' + '='*80)
        print('CREATING METACELLS')
        print('='*80)
        
        # Create a copy of adata for metacell processing (before it's deleted)
        # We need to work with the QC'd and cell-type-mapped data
        adata_for_metacell = adata.copy()
        
        # Create metacells (150 cells per metacell by default)
        adata_metacell = create_metacells(adata_for_metacell, metacell_size=150)
        
        print(f'Saving metacell bulk to: {output_path_metacell}')
        adata_metacell.write(output_path_metacell)
        
        del adata_for_metacell
        del adata_metacell
        gc.collect()
    
    # Clean up memory
    del adata
    del adata_bulk
    gc.collect()
    
    print(f'Successfully processed and saved {file_path}')
    
    return output_path
