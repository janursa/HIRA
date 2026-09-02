"""Train the per-cell-type aging clocks on CLOCK_TRAINING_COHORTS (bulk, ridge).

Usage: python src/clock/run_train.py
Writes: CLOCKS_DIR/<cell_type>/ (one model per MAJOR_CTS entry, version CLOCK_V)
"""
import anndata as ad
from grnimmuneclock import train_aging_clock
from hira import MAJOR_CTS, CLOCK_TRAINING_COHORTS, CLOCKS_DIR, CLOCK_V, CLOCK_CV_SCORING, TUNE_CLOCK
from hira.src.utils.util import retrieve_adata


if __name__ == "__main__":

    cell_types = MAJOR_CTS
    train_datasets = CLOCK_TRAINING_COHORTS
    data_type = 'bulk'
    reg_type = 'ridge'
    tune_model = TUNE_CLOCK
    version = CLOCK_V

    for cell_type in cell_types:
        print(f"\n{'='*60}")
        print(f"Training {cell_type} aging clock")
        print(f"{'='*60}")

        adata_train = ad.concat([
            retrieve_adata(dataset=dataset, data_type=data_type, cell_type=cell_type, only_net_genes=True)
            for dataset in train_datasets
        ])

        model, predictions, adata_train = train_aging_clock(
            adata=adata_train,
            cell_type=cell_type,
            version=version,
            output_dir=CLOCKS_DIR,
            reg_type=reg_type,
            tune_model=tune_model,
            scoring=CLOCK_CV_SCORING,
            verbose=True
        )
