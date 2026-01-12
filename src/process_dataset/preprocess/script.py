
import anndata as ad
from hiara.src.config import DATASET_NAME_MAPPING, get_config
import argparse

## VIASH START
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

parser.add_argument('--n_cell_t', help='number of cells threshold', default=10) # optio
parser.add_argument('--dataset', help='dataset to process', required=True)

par = vars(parser.parse_args())
run_test = par['run_test']

## VIASH END
from hiara.src.process_dataset.preprocess.helper import annotate_celltypes, qc_check, format_data, qc_post_annotation

def all_preprocessing_steps(adata):
    print('Running QC...', flush=True)
    adata = qc_check(adata)
    print('Running cell type annotation...', flush=True)
    adata = annotate_celltypes(adata)
    print('Cell type annotation done.', flush=True)
    adata = qc_post_annotation(adata, par)
    return adata

def main(par):
    dataset = par['dataset']
    dataset = DATASET_NAME_MAPPING.get(dataset, dataset)
    par['dataset'] = dataset 
    file_name = par['input_file']
    
    adata = ad.read_h5ad(file_name, backed='r')
    print('Formatting data...', flush=True)
    adata = format_data(adata, dataset)
    if run_test: # test
        print('Running in test mode, subsetting data...', flush=True)
        config = get_config(par['dataset'])
        pseudobulk_group = config.pseudobulk_group
        pseudobulk_group = [col for col in pseudobulk_group if col != 'cell_type']
        print(pseudobulk_group)
        # Get unique groups and keep only first 2
        groups = adata.obs.groupby(pseudobulk_group).size().reset_index()
        groups_to_keep = groups.head(2)
        
        # Create mask to filter adata
        mask = adata.obs.set_index(pseudobulk_group).index.isin(groups_to_keep.set_index(pseudobulk_group).index)
        adata = adata[mask].to_memory()
        print(f'Kept {adata.shape[0]} cells from 2 groups', flush=True)

    else:
        print('Reading to memory...', flush=True)
        adata = adata.to_memory()
    del adata.uns
    del adata.raw
    del adata.layers
    del adata.obsm
    del adata.varm
    del adata.varp

    # if dataset == 'CXCL9':
    #     adata = adata[adata.obs['treatment'].isin(['24 h RPMI', '24 h RPMI + ruxolitinib', '24 h LPS + ruxolitinib', '24 h LPS'])] 
    
    print('Running all preprocessing steps...', flush=True)
    adata = all_preprocessing_steps(adata)
    adata.obs['dataset'] = f"{dataset}"
    print(f"Writing processed data to {par['processed_files_dir']}/{dataset}.h5ad", flush=True)
    
    adata.write_h5ad(f"{par['processed_files_dir']}/{dataset}.h5ad", compression='gzip')

    print(adata, flush=True)

if __name__ == "__main__":
    print(par)
    main(par)
    
    
