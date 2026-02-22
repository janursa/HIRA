
import anndata as ad
import gc
from hiara.src.config import DATASET_NAME_MAPPING, get_config, MAJOR_CT_LABEL
import argparse
import glob
import os


## VIASH END
from hiara.src.process_dataset.preprocess.helper import annotate_celltypes, basic_qc, format_data

def subset_to_test(adata):
    print('Test mode: subsetting data', flush=True)
    cell_indices = adata.obs.groupby('bulk_group').apply(lambda x: x.index[0]).values
    adata = adata[cell_indices, :].to_memory()  
    return adata
def load_sc_data(file_name, dataset, run_test):
    if dataset == 'soundlife': # Handle multi-file input for soundlife
        print(f'Soundlife: Processing multiple files from directory: {file_name}', flush=True)
        input_pattern = os.path.join(file_name, 'SoundLife_*.h5ad')
        input_files = sorted(glob.glob(input_pattern))
        if not input_files:
            raise ValueError(f"No SoundLife_*.h5ad files found in {file_name}")
        print(f'Found {len(input_files)} SoundLife files to merge:', flush=True)
        for f in input_files:
            print(f'  - {os.path.basename(f)}', flush=True)
        # Read and merge all files
        merged_adata = None
        if run_test:
            input_files = input_files[:2]  # In test mode, only take first 2 files

        for i, input_file in enumerate(input_files):
            print(f'\nReading file {i+1}/{len(input_files)}: {os.path.basename(input_file)}', flush=True)
            adata_temp = ad.read_h5ad(input_file, backed='r')
            adata_temp = format_data(adata_temp, dataset)
            if run_test:
                # In test mode, take only limited groups
                adata_temp = subset_to_test(adata_temp)
            else:
                adata_temp = adata_temp.to_memory()

            if merged_adata is None:
                merged_adata = adata_temp
            else:
                # Concatenate with inner join on genes
                merged_adata = ad.concat(
                    [merged_adata, adata_temp],
                    join='inner',
                    merge='same'
                )
            
            print(f'Merged shape so far: {merged_adata.shape}', flush=True)
            del adata_temp
            gc.collect()
        
        adata = merged_adata
        print(f'\nFinal merged soundlife shape: {adata.shape}', flush=True)
        
    else:
        adata = ad.read_h5ad(file_name, backed='r')
        adata = format_data(adata, dataset)
        if run_test: # test
            print('Running in test mode, subsetting data...', flush=True)
            adata = subset_to_test(adata)
            print(f'Kept {adata.shape[0]} cells from groups', flush=True)
        else:
            print('Reading to memory...', flush=True)
            adata = adata.to_memory()
    print('Formatting data...', flush=True)
    return adata

def main(par):
    intermediate_save = True
    dataset = par['dataset']
    dataset = DATASET_NAME_MAPPING.get(dataset, dataset)
    par['dataset'] = dataset 
    file_name = par['input_file']
    adata = load_sc_data(file_name, dataset, par['run_test'])
    print('Running QC...', flush=True)
    if not par['run_test']:
        adata = basic_qc(adata)
    if intermediate_save:
        print(f"Saving intermediate QC result to {par['processed_files_dir']}/{dataset}_qc.h5ad", flush=True)
        adata.write_h5ad(f"{par['processed_files_dir']}/{dataset}_qc.h5ad", compression='gzip')
    print('Running cell type annotation...', flush=True)
    adata = annotate_celltypes(adata, dataset)
    print(f"Writing processed data to {par['processed_files_dir']}/{dataset}.h5ad", flush=True)
    adata.write_h5ad(f"{par['processed_files_dir']}/{dataset}.h5ad", compression='gzip')
    print(adata, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--processed_files_dir', 
        type=str,
        required=True,
        help="Processed files dir"
        )
        
    parser.add_argument('--input_file', 
        type=str,
        required=True,
        help="Location of raw input_file"
        )
    parser.add_argument('--run-test', 
        action='store_true',
        help="Whether to run in test mode (subset of data)"
        )

    parser.add_argument('--dataset', help='dataset to process', required=True)

    par = vars(parser.parse_args())
    main(par)
    
    
