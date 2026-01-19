
# from hiara.src.clock.train import wrapper_build_model_cell_type
import os
from pathlib import Path
from grnimmuneclock import train_aging_clock
from hiara import CELL_TYPES, CLOCK_TRAINING_COHORTS, CLOCKS_DIR, clock_version
    

# def wrapper_build_model_all(par):
#     for cell_type in par['cell_types']:
#         print('building model for cell type:', cell_type)
#         wrapper_build_model_cell_type(cell_type, par)
    

if __name__ == "__main__":

    cell_types = CELL_TYPES
    train_datasets = CLOCK_TRAINING_COHORTS
    data_type = 'bulk'
    reg_type = 'ridge'
    tune_model = True
    version = clock_version

    for cell_type in cell_types:
        print(f"\n{'='*60}")
        print(f"Training {cell_type} aging clock")
        print(f"{'='*60}")
        
        model, predictions, adata_train = train_aging_clock(
            cell_type=cell_type,
            datasets=train_datasets,
            data_type=data_type,
            reg_type=reg_type,
            tune_model=tune_model,
            output_dir=CLOCKS_DIR,
            version=version,
            verbose=True,
            scoring='spearman'
        )
        # model_dir = Path(temp_dir) / cell_type
        # model_dir.mkdir(parents=True, exist_ok=True)
        # adata_train.write(f"{model_dir}/adata_train.h5ad")
        
        # print(f"\n✓ {cell_type} model trained and saved")
        # print(f"  Training R²: {adata_train.obs[['age', 'predicted_age']].corr().iloc[0,1]**2:.3f}")