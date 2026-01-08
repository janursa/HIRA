
from ciim.src.clock.train import wrapper_build_model_cell_type
from ciim.src.config import cell_types
import os

import argparse

arg = argparse.ArgumentParser(description='Train a model for a specific cell type')
arg.add_argument('--cell_types', type=str,  nargs='+', default=['MONO'], help='Cell type to train the model for')
arg.add_argument('--datasets_training', type=str, nargs='+', default=['data1', 'data7_allTPs_jalil'], help='Datasets for training')
arg.add_argument('--feature_type', type=str, default='tf_activity', help='Feature type to use for training')
arg.add_argument('--data_type', type=str, default='metacell', help='Data type to use for training')
arg.add_argument('--reg_type', type=str, default='ridge', help='Regularization type to use for training')
arg.add_argument('--tune_model', action='store_true', help='Whether to tune the model or not')
arg.add_argument('--version', type=str, default='v1.0')
arg.add_argument('--temp_dir', type=str, default='tmp/')
arg.add_argument('--age_limit', type=int, default=20)


par = vars(arg.parse_args())

def wrapper_build_model_all(par):
    for cell_type in par['cell_types']:
        print('building model for cell type:', cell_type)
        wrapper_build_model_cell_type(cell_type, par)
    

if __name__ == "__main__":
    print(par)
    os.makedirs(par['temp_dir'], exist_ok=True)
    wrapper_build_model_all(par)