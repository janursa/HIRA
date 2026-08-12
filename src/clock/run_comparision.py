import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from hira import MAJOR_CTS, CLOCKS_DIR, OUTPUT_DIR, PLOTS_DIR, surrogate_names, colors_blind
from hira import wrapper_clock_predictions
from hira.src.clock.plots import plot_scatter_age_vs_predictedAge
from hira.src.utils.util import retrieve_adata
from grnimmuneclock import evaluate_groupwise_median, train_aging_clock
import anndata as ad


W_train_datasets = ['abf300', 'onek1k'] # we only use these datasets for training for a fair comparision
test_datasets = ['perez_sle', 'hida', 'zhang']
version = 'comparitive'

def train_clocks():
    
    cell_types = MAJOR_CTS
    data_type = 'bulk'
    reg_type = 'ridge'
    tune_model = True

    for cell_type in cell_types:
        print(f"\n{'='*60}")
        print(f"Training {cell_type} aging clock for Wenchao comparison")
        print(f"{'='*60}")
        
        adata_train = ad.concat([
            retrieve_adata(dataset=dataset, data_type=data_type, cell_type=cell_type, only_net_genes=True)
            for dataset in W_train_datasets
        ])

        model, predictions, adata_train = train_aging_clock(
            adata=adata_train,
            cell_type=cell_type,
            reg_type=reg_type,
            tune_model=tune_model,
            output_dir=CLOCKS_DIR,
            version=version,
            verbose=True
        )
def extract_w_results():
    # - read the results of predictions
    df_store = []
    for dataset in ['C4', 'C5']:
        df = pd.read_csv(f'{OUTPUT_DIR}/wenchao/val_data_majorCT_Wenchao_AC_{dataset}_predicted_age_donor.tsv', sep='\t')
        df['dataset'] = 'aida' if dataset=='C4' else 'perez_sle'
        df_store.append(df)
    df = pd.concat(df_store, axis=0)
    df['donor_age'] = df['donor_id'].astype(str) + '_' + df['age'].astype(str)
    # - format 
    df.rename(columns={'Prediction': 'predicted_age', 'CT': 'cell_type'}, inplace=True)
    df['sex'] = df['sex'].map({'M': 'Male', 'F': 'Female'})
    # - median prediction
    median_prediction = df.groupby(['cell_type', 'donor_age', 'dataset'])['predicted_age'].median().reset_index()
    median_prediction = df[['cell_type', 'donor_age', 'age', 'sex', 'dataset']].drop_duplicates().merge(median_prediction, on=['cell_type', 'donor_age', 'dataset'], how='left')
    return median_prediction
if __name__ == "__main__":
    test_datasets = ['perez_sle', 'aida']
    # train_clocks()
    W_median_prediction = extract_w_results()
    predictions_all = wrapper_clock_predictions(MAJOR_CTS, evaluate_datasets=test_datasets, version=version)
    test_predictions = predictions_all[predictions_all['condition']=='healthy'] # only healthy samples?

    print(f"Test predictions shape: {test_predictions.shape}")
    print(f"Unique cell types in test_predictions: {test_predictions['cell_type'].unique()}")
    print(f"Unique datasets in test_predictions: {test_predictions['dataset'].unique()}")
    print(f"Unique cell types in W_median_prediction: {W_median_prediction['cell_type'].unique()}")
    print(f"Unique datasets in W_median_prediction: {W_median_prediction['dataset'].unique()}")

    all_scores = []

    for cell_type in test_predictions['cell_type'].unique():
        for dataset in test_datasets:
            # Your model
            df1 = test_predictions[
                (test_predictions['dataset'] == dataset) & 
                (test_predictions['cell_type'] == cell_type)
            ].copy()
            ss1 = evaluate_groupwise_median(df1)  # returns {'Spearman': ..., 'R2': ...}
            all_scores.append({
                'dataset': dataset,
                'cell_type': cell_type,
                'model': 'GRNdrived',
                'Spearman': ss1['Spearman'],
                'R2': ss1['R2']
            })

            # Wenchao's model
            df2 = W_median_prediction[
                (W_median_prediction['dataset'] == dataset) & 
                (W_median_prediction['cell_type'] == cell_type)
            ].copy()
            ss2 = evaluate_groupwise_median(df2)
            all_scores.append({
                'dataset': dataset,
                'cell_type': cell_type,
                'model': 'Wenchao',
                'Spearman': ss2['Spearman'],
                'R2': ss2['R2']
            })

    # Convert to DataFrame
    score_df = pd.DataFrame(all_scores)
    assert score_df.shape[0] > 0, "Some issues here."
    palette_models = {
        'Wenchao': colors_blind[1], 
        'GRNdrived': colors_blind[0]
    }
    # Clip R2 to [0, 1]score_df['R2'] = score_df['R2'].clip(0, 1)

    # Sort datasets for consistent layout
    datasets = score_df['dataset'].unique()
    n_datasets = len(datasets)

    # Plot each metric in a separate figure
    
    for metric in ['R2', 'Spearman']:
        fig, axes = plt.subplots(1, n_datasets, figsize=(2.5 * n_datasets, 1.7), sharey=True)

        if n_datasets == 1:
            axes = [axes]  # Ensure iterable if only one axis

        for ax, dataset in zip(axes, datasets):
            df_plot = score_df[score_df['dataset'] == dataset]

            sns.barplot(
                data=df_plot,
                x='cell_type',
                y=metric,
                hue='model',
                ax=ax,
                alpha=0.8,
                palette=palette_models,
            )   

            ax.set_ylabel(metric)
            ax.set_xlabel('')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.margins(x=0.1, y=0.1)
            ax.spines[['top', 'right']].set_visible(False)
            ax.get_legend().remove()
            ax.set_title(f'{surrogate_names[dataset]}', pad=15)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title='Model', bbox_to_anchor=(.9, 1), loc='upper left', frameon=False)
        # fig.tight_layout()
        file_name = f'{PLOTS_DIR}/clock_comparison_{metric}.png'
        print(f"Saving figure to {file_name}")
        plt.savefig(file_name, bbox_inches='tight', dpi=300, transparent=True)

    W_median_prediction['model'] = 'Wenchao'
    test_predictions['model'] = 'GRNdrived'

    obs = pd.concat([test_predictions, W_median_prediction], axis=0)
    datasets = obs['dataset'].unique()

    for cell_type in ['CD8T']:
        fig, axes = plt.subplots(1, len(datasets), figsize=(2.8*len(datasets), 2), sharey=True)
        i = 0
        for dataset in datasets:
            ax = axes[i] if len(datasets) > 1 else axes
            obs_d = obs[(obs['cell_type'] == cell_type) & (obs['dataset'] == dataset)].copy()

            plot_scatter_age_vs_predictedAge(obs_d, dataset=dataset, ax=ax, hue='model', palette=palette_models)
            i+=1
            ax.get_legend().remove()
        ax.legend(loc=(1.1, .5), frameon=False, title='Model')
        file_name = f'{PLOTS_DIR}/clock_comparison_{cell_type}.png'
        print(f"Saving figure to {file_name}")
        plt.savefig(file_name, bbox_inches='tight', dpi=300, transparent=True)
        
        # plt.suptitle(cell_type, fontsize=12, weight='bold')
        # plt.tight_layout()