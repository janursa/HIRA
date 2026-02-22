
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
from hiara.src.config import get_config, SUB_CT_LABEL, MAJOR_CT_LABEL

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
    
    # print(f'Added standardized columns. Total columns: {len(adata.obs.columns)}')
    # print(f'Age group distribution:')
    # print(adata.obs['age_group'].value_counts(dropna=False))
    # print(f'Vaccinated distribution:')
    # print(adata.obs['vaccinated'].value_counts(dropna=False))
    # print(f'Year distribution:')
    # print(adata.obs['year'].value_counts(dropna=False))
    
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
    
    # Map major cell types
    adata.obs['cell_type'] = adata.obs['AIFI_L2'].map(cell_type_mapping)
    adata.obs[MAJOR_CT_LABEL] = adata.obs['cell_type']
    
    # Map standardized sub cell types
    adata.obs[SUB_CT_LABEL] = adata.obs['AIFI_L2'].map(sub_cell_type_mapping)
    
    # Keep original AIFI_L2 for reference
    adata.obs['AIFI_L2_original'] = adata.obs['AIFI_L2'].astype(str)
    
    # Count unmapped cells
    unmapped_major = adata.obs['cell_type'].isna().sum()
    unmapped_sub = adata.obs[SUB_CT_LABEL].isna().sum()
    total = adata.shape[0]
    
    print(f'Unmapped major cell types: {unmapped_major:,} ({unmapped_major/total*100:.2f}%)')
    print(f'Unmapped sub cell types: {unmapped_sub:,} ({unmapped_sub/total*100:.2f}%)')
    
    # Filter out unmapped cell types
    adata = adata[~adata.obs['cell_type'].isna()].copy()
    print(f'Shape after filtering unmapped cell types: {adata.shape}')
    
    # Show cell type distribution
    print('\nMajor cell type distribution:')
    print(adata.obs['cell_type'].value_counts())
    
    print(f'\nStandardized sub cell type distribution:')
    print(adata.obs[SUB_CT_LABEL].value_counts())
    
    return adata

def remove_attributes(adata):
    for attr in ['uns', 'raw', 'layers', 'obsm', 'varm', 'varp']:
        if hasattr(adata, attr):
            delattr(adata, attr)
    return adata
def format_data(adata, dataset_name):
    config = get_config(dataset_name)
    bulk_group = config.bulk_group
    adata.obs['bulk_group'] = adata.obs[bulk_group].astype(str).agg('_'.join, axis=1)
    # Soundlife-specific formatting
    if dataset_name == 'soundlife':
        adata = format_columns_soundlife(adata)
        # For soundlife, gene names are already in index, just standardize the column
        adata.var.index.name = 'gene_name'
        adata.var = adata.var.reset_index()[['gene_name']].set_index('gene_name')
        adata.obs = adata.obs.astype('str')
        return adata
    
    # ParseBioscience-specific formatting
    if dataset_name == 'parsebioscience':
        adata = format_columns_parsebioscience(adata)
        # For parsebioscience, gene names are already in index
        adata.var.index.name = 'gene_name'
        adata.var = adata.var.reset_index()[['gene_name']].set_index('gene_name')
        adata.obs = adata.obs.astype('str')
        return adata
    
    if dataset_name == 'op':
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
    adata.obs.rename(columns={'perturbation':'condition', 'disease':'condition', 'treatment':'condition'}, inplace=True)
    adata.obs.rename(columns={'orig.ident': 'dataset'}, inplace=True)
    adata.obs = adata.obs.astype('str')
    adata.obs['donor_age'] = adata.obs['age'].astype(str) + '_' + adata.obs['donor_id'].astype(str)
    adata = remove_attributes(adata)
    return adata

### QC Check
def basic_qc(adata):
    print('Shape before filtering:', adata.shape)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    n_donors = adata.obs['donor_id'].nunique()
    min_cells_per_donor = 10 # - consider the number of donors
    min_cells = int(n_donors * min_cells_per_donor)
    min_cells = max(min_cells, 10)
   
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_cells(adata, max_genes=5000)
    # Apply filters
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_genes(adata, min_counts=1)
    print('Shape after filtering:', adata.shape)
    assert adata.shape[0] > 0, "No cells left after QC filtering."
    return adata

def annotate_celltypes(adata, dataset):
    
    # Soundlife uses pre-existing AIFI_L2 annotations instead of CellTypist
    if dataset == 'soundlife': 
        print('Using pre-existing AIFI_L2 annotations for soundlife...')
        adata = map_cell_types_soundlife(adata)
        # Restore raw counts to X (map_cell_types expects and returns raw counts)
        return adata
    
    # ParseBioscience uses pre-existing annotations instead of CellTypist
    if dataset == 'parsebioscience':
        print('Using pre-existing cell type annotations for parsebioscience...')
        adata = map_cell_types_parsebioscience(adata)
        # Restore raw counts to X (map_cell_types expects and returns raw counts)
        return adata
    
    # Standard CellTypist annotation for other datasets
    print('Annotating cell types...')
    adata.layers['counts'] = adata.X.copy()
    ### Celltype annotation via Celltypist:
    import celltypist
    from celltypist import models
    ### Normalization
    # Before normalization, we need to store the raw counts in the layers['counts'] metadata for later use in the differential expression analysis
    gc.collect()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    models.download_models(force_update = True)
    model = models.Model.load(model = 'Immune_All_Low.pkl') #Immune_All_Low good for Major cell types and Immune_All_High for subtypes
    # Please note that the adata.X should be log-normalized data!
    adata_for_celltypist = adata.copy()
    # Annotate cell types using CellTypist
    print('Annotating cell types using CellTypist...')
    print(adata_for_celltypist.shape)
    predictions = celltypist.annotate(
        adata_for_celltypist,
        model=model,
        majority_voting=True,
        use_GPU=False
    )
    print('Cell types annotated successfully!')
    # Update the AnnData object with predictions
    adata_for_celltypist = predictions.to_adata()

    ###Major CT
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
        'MAIT cells': 'CD8T',  # or 'MAIT' if you want to keep it separate
        'Regulatory T cells': 'CD4T',
        'Cycling T cells' : 'CD4T',
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
        'Late erythroid': 'Erythroid'
    }
    ###SubPopulation
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
        'MAIT cells': 'MAIT',  # or 'MAIT' if you want to keep it separate
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
        'Late erythroid': 'Late_Erythroid'
    }
    # Map Major and Sub cell types
    adata_for_celltypist.obs['Major_CT'] = adata_for_celltypist.obs['majority_voting'].apply(lambda x: mapping.get(x, 'Others'))
    adata_for_celltypist.obs[SUB_CT_LABEL] = adata_for_celltypist.obs['majority_voting'].apply(lambda x: mapping_sub.get(x, 'Others'))
    # - post process
    adata.obs = adata.obs.join(adata_for_celltypist.obs[['Major_CT', SUB_CT_LABEL]])
    adata.X = adata.layers["counts"]
    del adata.layers
    adata.obs['cell_type'] = adata.obs['Major_CT']
    major_cell_types = ["MONO", "NK", "B", "CD8T", "CD4T"]
    adata = adata[adata.obs['cell_type'].isin(major_cell_types)]
    
    return adata


# def binarize_age(obs):
#     obs = obs.copy()
#     obs['donor_age'] = obs['age'].astype(str) + '_' + obs['donor_id'].astype(str)
#     obs['age'] = pd.to_numeric(obs['age'], errors='coerce')
#     # min_age = obs.age.min()
#     # bins = [min_age, 35, 45, 55, 65, 75, 100]  
#     # age_groups = ['34-', '35_44', '45_54', '55_64', '65_75', '75+']  
#     # obs['age_group'] = pd.cut(obs['age'], bins=bins, labels=age_groups, right=False)
#     return obs

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
    
    # Map major and sub cell types
    adata.obs[MAJOR_CT_LABEL] = adata.obs['cell_type_original'].map(major_cell_type_map)
    adata.obs[SUB_CT_LABEL] = adata.obs['cell_type_original'].map(sub_cell_type_map)
    adata.obs['cell_type'] = adata.obs[MAJOR_CT_LABEL]
    
    # Count unmapped cells
    unmapped_major = adata.obs[MAJOR_CT_LABEL].isna().sum()
    unmapped_sub = adata.obs[SUB_CT_LABEL].isna().sum()
    total = adata.shape[0]
    
    print(f'Unmapped major cell types: {unmapped_major:,} ({unmapped_major/total*100:.2f}%)')
    print(f'Unmapped sub cell types: {unmapped_sub:,} ({unmapped_sub/total*100:.2f}%)')
    
    # Filter out unmapped cell types (use .copy() to avoid view issues in backed mode)
    adata = adata[~adata.obs['cell_type'].isna(), :].copy()
    print(f'Shape after filtering unmapped cell types: {adata.shape}')
    
    # Show cell type distribution
    print('\nMajor cell type distribution:')
    print(adata.obs[MAJOR_CT_LABEL].value_counts())
    
    print(f'\nStandardized sub cell type distribution:')
    print(adata.obs[SUB_CT_LABEL].value_counts())
    
    return adata