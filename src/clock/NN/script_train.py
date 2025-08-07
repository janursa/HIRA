import seaborn as sns
from typing import Optional
import pandas as pd
import numpy as np
from scipy import sparse
from anndata import AnnData
import scanpy as sc
import torch
torch.set_float32_matmul_precision('medium')
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import cpa
from ciim.src.clock.NN.helper import wrapper_setup_data, format_data, train_datasets, test_datasets, get_params, data_type, cell_type, save_path_train, save_path_test, extend_embedding, batch_key

import argparse 
arg = argparse.ArgumentParser(description='Train CPA model')
arg.add_argument('--mode', type=str, default='train', help='Whether to train or test')
args = arg.parse_args()


if __name__ == '__main__':
    print('Data type:', data_type, 'Cell type:', cell_type)
    model_params, trainer_params = get_params(data_type)

    if args.mode == 'train':
        print("Training mode activated.")
        assert len(train_datasets)>0, "Training mode requires at least one training dataset."
        print('Loading training data ... ')
        adata = format_data(train_datasets, cell_type, data_type)
        wrapper_setup_data(adata, batch_key=batch_key, data_type=data_type, cell_type=cell_type)
        model = cpa.CPA(adata=adata,
                        **model_params,
                    )
        model.train(max_epochs=200,
                use_gpu=True,
                batch_size=128,
                plan_kwargs=trainer_params,
                early_stopping_patience=5,
                check_val_every_n_epoch=100,
                save_path=save_path_train,
            )
    else:
        print("Testing mode activated.")
        print('Loading model ... ')
        model = cpa.CPA.load(dir_path=save_path_train)
        assert len(test_datasets)==1, "Testing mode requires exactly one test dataset."
        print('Loading training data ... ')
        adata_train = model.adata
        print('Loading test data ... ')
        adata_test = format_data(test_datasets, cell_type, data_type)
        from ciim.src.clock.helper import align_feature_space
        adata_test = align_feature_space(adata_test, adata_train.var_names)
        
        model.setup_anndata_test(adata_test, is_count_data=True if data_type == 'sc' else False)
        print('Extending embedding for test dataset ... ')
        extend_embedding(model=model, new_dataset=test_datasets[0], covariate=batch_key)
        print(model.covars_encoder)
        model._register_manager_for_instance(
                        model.adata_manager.transfer_fields(adata_test, extend_categories=True)
                )
        model.train(max_epochs=200,
                initial_training = False,
                use_gpu=True,
                batch_size=128,
                plan_kwargs=trainer_params,
                early_stopping_patience=5,
                check_val_every_n_epoch=100,
                save_path=save_path_test,
            )
    
    