from ciim.src.common import save_dir
import cpa
import scanpy as sc


run_id='bulk' #'try1'
data_type = 'bulk'
cell_type = None
batch_key = 'dataset'

save_path_train=f'{save_dir}/NN/{run_id}_train'
save_path_test=f'{save_dir}/NN/{run_id}_test'
train_datasets=['data1',
                'data7_allTPs_jalil',
                'SLE_European',
                'data13_Korean'
                ]
test_datasets=['data13_Japanese'] #'data12'

def wrapper_setup_data(adata, batch_key, data_type, cell_type):
    if cell_type is None:
        cpa.CPA.setup_anndata(adata,
                                # perturbation_key='disease',
                                # control_group='healthy',
                                batch_key=batch_key,  
                                categorical_covariate_keys=['cell_type'],
                                is_count_data=True if data_type == 'sc' else False,
                                max_comb_len=1,
                                )
        
    else:
        cpa.CPA.setup_anndata(adata,
                                # perturbation_key='disease',
                                # control_group='healthy',
                                batch_key=batch_key,  
                                is_count_data=True if data_type == 'sc' else False,
                                max_comb_len=1,
                                )
        

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
        "n_epochs_pretrain_ae": 30,
        "n_epochs_adv_warmup": 50,
        "n_epochs_mixup_warmup": 0,
        "mixup_alpha": 0.0,
        "adv_steps": None,
        "n_hidden_adv": 64,
        "n_layers_adv": 3,
        "use_batch_norm_adv": True,
        "use_layer_norm_adv": False,
        "dropout_rate_adv": 0.3,
        "reg_adv": 20.0,
        "pen_adv": 5.0,
        "lr": 0.0003,
        "wd": 4e-07,
        "adv_lr": 0.0003,
        "adv_wd": 4e-07,
        "adv_loss": "cce",
        "doser_lr": 0.0003,
        "doser_wd": 4e-07,
        "do_clip_grad": True,
        "gradient_clip_value": 1.0,
        "step_size_lr": 10,
    }
    return model_params, trainer_params

def format_data(datasets, cell_type=None, data_type='bulk'):
    from ciim.src.utils.util import retrieve_adata_bulk, get_consensus_net
    import anndata as ad
    
    adata_store = []
    for d in datasets:
        if d == 'SLE_European':
            adata = retrieve_adata_bulk(d, type=data_type, cell_type=cell_type)
            adata.obs['disease'] = adata.obs['disease'].map({'normal': 'healthy', 'systemic lupus erythematosus': 'SLE'})
            adata = adata[adata.obs['disease'].isin(['healthy'])].copy()
        else:
            adata = retrieve_adata_bulk(d, type=data_type)
            adata.obs['disease'] = 'healthy'
        adata_store.append(adata)

    adata_train = ad.concat(adata_store, join='inner', axis=0)
    if cell_type is not None:
        net = get_consensus_net(cell_type=cell_type)
        adata_train = adata_train[adata_train.obs['cell_type'] == cell_type, adata_train.var_names.isin(net['target'].unique())].copy()

    adata_train.obs_names_make_unique()

    return adata_train

def extend_embedding(model, new_dataset, covariate):
    import torch
    import torch.nn as nn

    # Get current covariate encoding and embedding
    covars = model.covars_encoder[covariate]

    if new_dataset not in covars:
        current_embedding = model.module.covars_embeddings[covariate]

        # Add new covariate
        n_covars = len(covars)
        covars[new_dataset] = n_covars

        # Create new embedding layer with 1 extra row
        new_embedding_layer = nn.Embedding(n_covars + 1, model.module.n_latent)

        # Copy old weights into new embedding layer
        with torch.no_grad():
            new_embedding_layer.weight[:n_covars] = current_embedding.weight

        # Freeze old rows using a gradient hook
        def freeze_old_rows(grad):
            grad[:n_covars] = 0
            return grad

        new_embedding_layer.weight.register_hook(freeze_old_rows)

        # Replace the embedding layer in the model
        model.module.covars_embeddings[covariate] = new_embedding_layer

    # Freeze all other model.module parameters
    # for name, param in model.module.named_parameters():
    #     if f'covars_embeddings.{covariate}.weight' in name:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False
    # - check trainable params
    for name, param in model.module.named_parameters():
        if param.requires_grad:
            print(f"{name}: trainable")
        else:
            # print(f"{name}: frozen (requires_grad=False)")
            pass

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