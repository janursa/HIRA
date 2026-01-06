
import anndata as ad
from ciim.src.config import DATASET_NAME_MAPPING
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

parser.add_argument('--n_cell_t', help='number of cells threshold', default=200) # optio
parser.add_argument('--dataset', help='dataset to process', required=True)

par = vars(parser.parse_args())


## VIASH END
from ciim.src.process_dataset.preprocess.helper import annotate_celltypes, qc_check, format_data, qc_post_annotation
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
    file_name = par['input_file']
    
    adata = ad.read_h5ad(file_name, backed='r')
    # if dataset == 'CXCL9':
    #     adata = adata[adata.obs['treatment'].isin(['24 h RPMI', '24 h RPMI + ruxolitinib', '24 h LPS + ruxolitinib', '24 h LPS'])] 
    dataset_pretty = DATASET_NAME_MAPPING.get(dataset, dataset)
    if 'race' in adata.obs.columns:
        races = adata.obs['race'].unique()
        for race in races:
            race = race.lower()
            mask = adata.obs['race']==race
            adata = adata[mask].to_memory()
            adata = format_data(adata, dataset)
            adata = all_preprocessing_steps(adata)
            
            adata.obs['dataset'] = f"{dataset_pretty}_{race}"
            adata.write_h5ad(f"{par['processed_files_dir']}/{dataset_pretty}_{race}.h5ad", compression='gzip')
    else:
        adata = adata.to_memory()
        adata = format_data(adata, dataset)
        adata = all_preprocessing_steps(adata)
        adata.obs['dataset'] = f"{dataset_pretty}"
        adata.write_h5ad(f"{par['processed_files_dir']}/{dataset_pretty}.h5ad", compression='gzip')

    del adata.uns
    del adata.raw
    del adata.layers
    del adata.obsm
    del adata.varm
    del adata.varp
    print(adata, flush=True)



if __name__ == "__main__":
    print(par)
    main(par)
    
    
