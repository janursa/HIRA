from hiara.src.config import OUTPUT_DIR
import scanpy as sc



data_type = 'sc'
cell_type_train = None
if cell_type_train is not None:
    run_id=f'{cell_type_train}_{data_type}' #'try1'
else:
    run_id=f'{data_type}'
batch_key = 'dataset'

save_path_train = f'{OUTPUT_DIR}/NN/{run_id}_train'
save_path_test = f'{OUTPUT_DIR}/NN/{run_id}_test'


# test_datasets = ['data13_Japanese', 'data12'] #'data12'


def get_params(data_type):
    recon_loss = 'nb' if data_type == 'sc' else 'gauss'
    model_params = {
        "n_latent": 64,
        "recon_loss": recon_loss,
        "doser_type": "linear",
        "n_hidden_encoder": 128,
        "n_layers_encoder": 2,
        "n_hidden_decoder": 128,
        "n_layers_decoder": 2,
        "use_batch_norm_encoder": True,
        "use_layer_norm_encoder": False,
        "use_batch_norm_decoder": False,
        "use_layer_norm_decoder": True,
        "dropout_rate_encoder": 0.0,
        "dropout_rate_decoder": 0.1,
        "variational": False,
        "seed": 6977,
    }

    trainer_params = {
        "n_epochs_kl_warmup": None,
        "n_epochs_pretrain_ae": 20,
        "n_epochs_adv_warmup": 20,
        "n_epochs_pretrain_age": 50,
        "age_lr": 0.0003,
        "n_epochs_mixup_warmup": 0,
        "mixup_alpha": 0.0,
        "adv_steps": None,
        "n_hidden_adv": 64,
        "n_layers_adv": 3,
        "use_batch_norm_adv": True,
        "use_layer_norm_adv": False,
        "dropout_rate_adv": 0.3,
        "reg_adv": 100.0 if data_type=='bulk' else 20.0,
        "pen_adv": 5,
        "lr": 0.0003,
        "wd": 4e-07,
        "adv_lr": 0.0003,
        "adv_wd": 4e-07,
        "adv_loss": "cce",
        "do_clip_grad": True,
        "gradient_clip_value": 1.0,
        "step_size_lr": 10,
    }
    return model_params, trainer_params



def wrapper_umap(ad, cols=['dataset', 'cell_type']):
      sc.pp.neighbors(ad)
      sc.tl.umap(ad)

      sc.pl.umap(ad,
            color=cols,
            frameon=False,
            wspace=0.5)
def plot_history(model):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import math

    df = model.epoch_history
    metric_cols = [col for col in df.columns if col not in ['epoch', 'mode']]
    n_metrics = len(metric_cols)

    n_rows = 2
    n_cols = math.ceil(n_metrics / n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, figsize=(3 * n_cols, 2.5 * n_rows))
    axes = axes.flatten()

    train_df = df[df['mode'] == 'train']
    valid_df = df[df['mode'] == 'valid']

    for i, col in enumerate(metric_cols):
        ax = axes[i]
        ax.plot(train_df['epoch'].values, train_df[col].values, label='train')
        if len(valid_df) > 0:
            ax.plot(valid_df['epoch'].values, valid_df[col].values, label='valid')
        ax.set_title(col)

    # Hide unused subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2)
    plt.tight_layout(rect=[0, 0.05, 1, 1])