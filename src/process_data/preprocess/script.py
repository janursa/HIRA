
import anndata as ad
import gc
from hiara.src.config import DATASET_NAME_MAPPING, get_config, MAJOR_CT_LABEL, PRIOR_DIR
import argparse
import glob
import os
import numpy as np
from hiara.src.process_data.preprocess.helper import annotate_celltypes, basic_qc, format_data

def subset_to_test(adata):
    print('Test mode: subsetting data', flush=True)
    cell_indices = adata.obs.groupby('bulk_group').apply(lambda x: x.index[0]).values
    adata = adata[cell_indices, :].to_memory()  
    return adata
def load_sc_data(file_name, dataset, run_test, gene_names=None):
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
                # In test mode, subset rows first (fast, backed), then load into memory
                adata_temp = subset_to_test(adata_temp)
            elif gene_names is not None:
                # Full run: filter to known genes before loading into memory (saves ~50% memory)
                keep = adata_temp.var_names.isin(gene_names)
                adata_temp = adata_temp[:, keep].to_memory()
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
            adata = subset_to_test(adata)
            print(f'Kept {adata.shape[0]} cells from groups', flush=True)
        else:
            print('Reading to memory...', flush=True)
            adata = adata.to_memory()
        # op stores raw counts in layers['counts']; X is pre-normalized.
        # Swap after to_memory() since backed objects are read-only.
        # Cast to int32: counts are integers stored as float64; explicit cast
        # ensures the integer-values assertion in main() passes cleanly.
        if dataset == 'op' and 'counts' in adata.layers:
            import scipy.sparse as sp
            raw = adata.layers['counts']
            raw = raw.tocsr() if sp.issparse(raw) else sp.csr_matrix(raw)
            adata.X = raw.astype(np.int32)
            del adata.layers['counts']
    return adata



def main(par):
    dataset = par['dataset']
    dataset = DATASET_NAME_MAPPING.get(dataset, dataset)
    par['dataset'] = dataset 
    file_name = par['input_file']

    # Load gene names early so soundlife can filter per-file before to_memory()
    gene_names = np.loadtxt(f'{PRIOR_DIR}/gene_names.txt', dtype=str)
    adata = load_sc_data(file_name, dataset, par['run_test'], gene_names=gene_names)
    adata = basic_qc(adata, par['run_test'])

    # For soundlife: per-file gene filter may leave small gene-set differences across files;
    # enforce the known gene list here to ensure consistency after inner-join concat.
    if dataset == 'soundlife':
        adata = adata[:, adata.var_names.isin(gene_names)].copy()

    adata = annotate_celltypes(adata, dataset)
    print(f"Writing processed data to {par['processed_files_dir']}/{dataset}.h5ad", flush=True)
    # assert that there is no layers before wiring
    assert not adata.layers, "adata should not have any layers before writing"
    # assert that .X is sparse
    assert not isinstance(adata.X, np.ndarray), "adata.X should be sparse before writing"
    # assert that .X is raw counts
    assert np.all(adata.X.data >= 0), "adata.X should contain raw counts (non-negative values)"
    one_value = adata.X.data[0]
    assert np.isclose(one_value % 1, 0), "adata.X should contain raw counts (integer values)"

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
    
    
