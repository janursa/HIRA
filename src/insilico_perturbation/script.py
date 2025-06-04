from ciim.src.insilico_perturbation.helper import wrapper_run_tf_screening
from ciim.src.common import save_dir


if __name__ == '__main__':
    perturbation_mode = 'overexpression' # 'overexpression' # 'natural_aging' for aging clock, 'overexpression' for overexpression perturbation
    par = {
        'years': 10,
        'n_donors': 1, # select only one donor for linear models
        'data_type': 'bulk',
        'version': 'v1.0',
        'reg_type': 'ridge',
        'feature_type': 'gene_expression',
        'perturbation_mode': perturbation_mode, # natural_aging, overexpression
        'perturbation_type': 'single',
        'tfs': None,  # None means all TFs
        
    }
    datasets = ['data1', 'data7_allTPs_jalil', 'SLE_European', 'data13_Japanese']
    cell_types = ['CD4T', 'CD8T', 'MONO', 'B', 'NK'] # ['CD4T', 'CD8T', 'MONO', 'B', 'NK']
    df_all = wrapper_run_tf_screening(par, 
                                    n_jobs=10,
                                    cell_types=cell_types,
                                    datasets=datasets 
                                    )
    save_file=f"{save_dir}/perturbation/tf_screen_results_{perturbation_mode}.csv"
    df_all.to_csv(save_file, index=False)