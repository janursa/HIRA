from ciim.src.insilico_perturbation.helper import wrapper_run_tf_screening
from ciim.src.common import save_dir

import argparse
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='Run in silico simulation.')

    parser.add_argument('--cell_types', nargs='+', required=True,
                        help='List of cell types (e.g. CD8T MONO)')
    parser.add_argument('--reg_type', type=str, required=True,
                        help='Type of regularization or regression')
    parser.add_argument('--simulation_iteration', type=int, default=0,
                        help='Simulation iteration index')
    parser.add_argument('--n_donors', type=int, default=None,
                        help='Number of donors to sample')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    par = {
        'simulation_iteration': args['simulation_iteration'],
        'n_donors': par['n_donors'], # select only one donor for linear models
        'data_type': 'bulk',
        'version': 'v1.0',
        'reg_type': par['reg_type'],
        'feature_type': 'gene_expression',
        'perturbation_mode': 'overexpression', # natural_aging, overexpression
        'perturbation_type': 'single',
        'tfs': None,  # None means all TFs
        
    }
    datasets = ['data1', 'data7_allTPs_jalil', 'SLE_European', 'data13_Japanese']
    cell_types = par['cell_types'] # ['CD4T', 'CD8T', 'MONO', 'B', 'NK']
    df_all = wrapper_run_tf_screening(par, 
                                    n_jobs=10,
                                    cell_types=cell_types,
                                    datasets=datasets 
                                    )
    save_file=f"{save_dir}/perturbation/tf_screen_results_{perturbation_mode}.csv"
    df_all.to_csv(save_file)