import os
import sys
import argparse
import glob
import anndata as ad
import scanpy as sc
import gc
from pathlib import Path

# Add helper module to path
sys.path.insert(0, os.path.dirname(__file__))
from helper import process_single_file

## VIASH START
parser = argparse.ArgumentParser(description='Process SoundLife datasets')

parser.add_argument(
    '--input_dir',
    type=str,
    default='/home/jnourisa/projs/ongoing/ciim/downloads',
    help="Directory containing SoundLife .h5ad files"
)

parser.add_argument(
    '--output_bulk',
    type=str,
    default='/vol/projects/jnourisa/datasets/bulk/soundlife.h5ad',
    help="Output path for final merged bulk data"
)

parser.add_argument(
    '--output_metacell',
    type=str,
    default='/vol/projects/jnourisa/datasets/bulk/soundlife.h5ad',
    help="Output path for final merged metacell data"
)

parser.add_argument(
    '--temp_dir',
    type=str,
    default='/home/jnourisa/projs/ongoing/ciim/src/process_dataset/soundlife/temp',
    help="Temporary directory for individual pseudobulked files"
)

parser.add_argument(
    '--test',
    action='store_true',
    help="Test mode: subset each file to 2 donors and 2 visits for quick testing"
)

args = parser.parse_args()
## VIASH END


def normalize(adata):
    """
    Normalize data using the same method as other datasets.
    """
    print('Normalizing data...')
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)
    return adata


def merge_pseudobulked_files(file_paths):
    """
    Merge all pseudobulked files using inner join on genes.
    """
    print(f'\n{"="*80}')
    print('Merging pseudobulked files...')
    print(f'{"="*80}')
    
    merged_adata = None
    
    for i, file_path in enumerate(file_paths):
        print(f'\nMerging file {i+1}/{len(file_paths)}: {file_path}')
        adata = ad.read_h5ad(file_path)
        print(f'  Shape: {adata.shape}')
        
        if merged_adata is None:
            merged_adata = adata
        else:
            # Concatenate with inner join (gene intersection)
            merged_adata = ad.concat(
                [merged_adata, adata],
                join='inner',
                merge='same'
            )
            print(f'  Merged shape: {merged_adata.shape}')
        
        del adata
        gc.collect()
    
    print(f'\nFinal merged shape: {merged_adata.shape}')
    print(f'Total bulk samples: {merged_adata.n_obs}')
    
    return merged_adata


if __name__ == '__main__':
    print('='*80)
    print('SoundLife Dataset Processing Pipeline')
    if args.test:
        print('*** RUNNING IN TEST MODE ***')
    print('='*80)
    
    # Create temp directory if it doesn't exist
    os.makedirs(args.temp_dir, exist_ok=True)
    
    # Find all SoundLife .h5ad files
    input_pattern = os.path.join(args.input_dir, 'SoundLife_*.h5ad')
    input_files = sorted(glob.glob(input_pattern))
    
    if not input_files:
        raise ValueError(f"No SoundLife .h5ad files found in {args.input_dir}")
    
    print(f'\nFound {len(input_files)} SoundLife files to process:')
    for f in input_files:
        print(f'  - {os.path.basename(f)}')
    
    # Process each file individually
    print('\n' + '='*80)
    print('STEP 1: Processing individual files and creating pseudobulk data')
    print('='*80)
    
    pseudobulked_files = []
    metacell_files = []
    
    for i, input_file in enumerate(input_files, 1):
        print(f'\n[{i}/{len(input_files)}] Processing file...')
        
        # Create output filename for pseudobulked data
        base_name = os.path.basename(input_file).replace('.h5ad', '_bulk.h5ad')
        output_file = os.path.join(args.temp_dir, base_name)
        
        # Create output filename for metacell data
        base_name_metacell = os.path.basename(input_file).replace('.h5ad', '_metacell.h5ad')
        output_file_metacell = os.path.join(args.temp_dir, base_name_metacell)
        
        # Process and pseudobulk (with test mode flag and metacell output)
        process_single_file(
            input_file, 
            output_file, 
            test_mode=args.test,
            output_path_metacell=output_file_metacell
        )
        pseudobulked_files.append(output_file)
        metacell_files.append(output_file_metacell)
        
        print(f'Completed {i}/{len(input_files)} files')
    
    # Merge all pseudobulked files
    print('\n' + '='*80)
    print('STEP 2: Merging all pseudobulked files')
    print('='*80)
    
    merged_adata = merge_pseudobulked_files(pseudobulked_files)
    
    # Merge all metacell files
    print('\n' + '='*80)
    print('STEP 2b: Merging all metacell files')
    print('='*80)
    
    merged_adata_metacell = merge_pseudobulked_files(metacell_files)
    
    # Normalize the merged data
    print('\n' + '='*80)
    print('STEP 3: Normalizing merged data')
    print('='*80)
    
    print('\nNormalizing regular bulk data...')
    merged_adata = normalize(merged_adata)
    
    print('\nNormalizing metacell data...')
    merged_adata_metacell = normalize(merged_adata_metacell)
    
    # Display summary statistics
    print('\n' + '='*80)
    print('SUMMARY STATISTICS - REGULAR BULK')
    print('='*80)
    print(f'\nFinal bulk data shape: {merged_adata.shape}')
    print(f'  - Samples: {merged_adata.n_obs}')
    print(f'  - Genes: {merged_adata.n_vars}')
    
    print('\nColumn names in .obs:')
    print(merged_adata.obs.columns.tolist())
    
    print('\nCell type distribution:')
    print(merged_adata.obs['cell_type'].value_counts())
    
    print('\nDonor count:')
    print(f"  Unique donors: {merged_adata.obs['donor_id'].nunique()}")
    
    print('\nVisit distribution:')
    print(merged_adata.obs['visitName'].value_counts())
    
    print('\nCondition distribution:')
    print(merged_adata.obs['condition'].value_counts())
    
    print('\nAge statistics:')
    print(merged_adata.obs['age'].describe())
    
    # Display summary statistics for metacell data
    print('\n' + '='*80)
    print('SUMMARY STATISTICS - METACELL')
    print('='*80)
    print(f'\nFinal metacell data shape: {merged_adata_metacell.shape}')
    print(f'  - Samples (metacells): {merged_adata_metacell.n_obs}')
    print(f'  - Genes: {merged_adata_metacell.n_vars}')
    
    print('\nColumn names in .obs:')
    print(merged_adata_metacell.obs.columns.tolist())
    
    print('\nCell type distribution:')
    print(merged_adata_metacell.obs['cell_type'].value_counts())
    
    print('\nDonor count:')
    print(f"  Unique donors: {merged_adata_metacell.obs['donor_id'].nunique()}")
    
    # Save final merged and normalized bulk data
    print('\n' + '='*80)
    print('STEP 4: Saving final bulk and metacell data')
    print('='*80)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_bulk)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f'\nSaving regular bulk to: {args.output_bulk}')
    merged_adata.write(args.output_bulk)
    
    print(f'Saving metacell bulk to: {args.output_metacell}')
    merged_adata_metacell.write(args.output_metacell)
    
    print('\n' + '='*80)
    print('PROCESSING COMPLETE!')
    print('='*80)
    print(f'\nFinal bulk data saved to: {args.output_bulk}')
    print(f'Final metacell data saved to: {args.output_metacell}')
    print(f'Temporary pseudobulked files stored in: {args.temp_dir}')
    print('\nYou can now use this data in your aging analysis pipeline.')
    print(f'\nComparison:')
    print(f'  Regular bulk samples: {merged_adata.n_obs}')
    print(f'  Metacell samples: {merged_adata_metacell.n_obs}')
    print(f'  Sample size increase: {merged_adata_metacell.n_obs / merged_adata.n_obs:.2f}x')
