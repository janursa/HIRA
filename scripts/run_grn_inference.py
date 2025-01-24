
import sys
import os 
from tqdm import tqdm
import argparse

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.join(parent_directory, '../'))
from src.helper_infer_grns import infer_grns_all
from src.helper_dataset import process_dataset

def run_preprocessing_grn_inference(dataset='data1', only_male=True, downsample=False, force=True):
    if dataset=='pbmc_ageing':
        dataset_file = '/vol/projects/jnourisa/adata_pbmc_ageing.h5ad'
    elif dataset=='data1':
        dataset_file  ='/vol/projects/CIIM/Healthy_Single_Cell_Data/count_matrix/data1_CMtx.h5ad'
    elif dataset=='all':
        dataset_file  = "/vol/projects/jnourisa/adata_all.h5ad"
    else:
        raise ValueError('dataset not defined')
    # ----- process dataset
    par_process_dataset={
        'n_cell_t': 200,
        'only_male': only_male,
        'equalize_donor_size': True,
        'downsample': downsample,
        'dataset': dataset,
        'dataset_file': dataset_file,
        'temp_dir': f'output/temp/{dataset}/',
        }
    # - process dataset
    obs = process_dataset(par_process_dataset)

    os.makedirs(par_process_dataset['temp_dir'], exist_ok=True)
    obs.to_csv(f"{par_process_dataset['temp_dir']}/obs_{dataset}.csv")

    # ----- GRN inference
    folder_tag = f"{dataset}_{only_male}_{downsample}"
    par={
        'min_genes_per_cell': 10, 
        'max_genes_per_cell': 5000, 
        'min_cells_per_gene': 2500,
        'save_dir': f'output/grns/{folder_tag}/',
        'weight_t': 0.05,
        'max_workers': 10,
        'force': force,
        'batches': ['batch_1', 'batch_2', 'all_batches'],
        'dataset_file': dataset_file
    }
    os.makedirs(par['save_dir'], exist_ok=True)
    # - run grn inference
    infer_grns_all(par, obs)
if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help="Name of the dataset"
    )
    parser.add_argument(
        '--downsample',
        action='store_true',
        help="Whether to equalize cell counts and donor sizes. Default is False."
    )
    parser.add_argument(
        '--only_male',
        action='store_true',
        help="Whether to subset the data to only males. Default is False."
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Force to rewrite existing files."
    )

    args = parser.parse_args()
    dataset=args.dataset
    downsample=args.downsample
    only_male=args.only_male
    force=args.force

    print(f"dataset: {dataset}, downsample: {downsample}, only_male: {only_male}, force: {force}")

    run_preprocessing_grn_inference(dataset=dataset, only_male=only_male, downsample=downsample, force=force)