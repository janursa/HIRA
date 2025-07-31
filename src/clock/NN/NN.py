# model.py
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import torch
from anndata import AnnData
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin, EmbeddingMixin
from scvi.module.base import BaseModuleClass, LossOutput, EmbeddingModuleMixin
from scvi.module import VAE
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from torch.distributions import Normal, kl_divergence
from scvi.nn import Embedding
from scvi.data.fields import (
    LayerField,
    CategoricalObsField,
    NumericalObsField,
    CategoricalJointObsField,
    NumericalJointObsField,
)
import torch.nn as nn
import torch.nn.functional as F
from scvi.train import TrainingPlan, TrainRunner
from scvi.dataloaders import DataSplitter
from scvi.train import TrainRunner

class Encoder(nn.Module):
    def __init__(self, n_genes, n_latent, n_hidden=128, dropout=0.2, n_batch_emb=2, latent_type='vanilla'):
        super().__init__()
        self.fc1 = nn.Linear(n_genes + n_batch_emb, n_hidden)
        self.fc_mu = nn.Linear(n_hidden, n_latent)
        self.fc_logvar = nn.Linear(n_hidden, n_latent)
        self.dropout = nn.Dropout(dropout)
        self.latent_type = latent_type

    def forward(self, x, batch_emb):
        x = torch.cat([x, batch_emb], dim=1)
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        mu = self.fc_mu(h)

        if self.latent_type == 'vanilla':
            z = mu
            logvar = torch.zeros_like(mu)
        elif self.latent_type == 'generative':
            logvar = self.fc_logvar(h)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            raise ValueError(f'{self.latent_type} is not defined.')

        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self, n_latent, n_genes, n_batch_emb, n_hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(n_latent + n_batch_emb, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_genes)

    def forward(self, z, batch_emb):
        z = torch.cat([z, batch_emb], dim=1)
        h = F.relu(self.fc1(z))
        return self.fc2(h)

class AgeRegressor(nn.Module):
    def __init__(self, n_latent, n_hidden, n_batch_emb):
        super().__init__()
        self.fc1 = nn.Linear(n_latent + n_batch_emb, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, 1)

    def forward(self, z, batch_emb):
        z = torch.cat([z, batch_emb], dim=1)
        h = F.relu(self.fc2(F.relu(self.fc1(z))))
        return self.fc3(h)

class myModule(BaseModuleClass):
    def __init__(
        self,
        n_genes: int,
        n_batches: int,
        n_latent: int = 100,
        n_hidden: int = 128,
        dropout: float = 0.2,
        n_batch_emb: int = 10,
        age_loss_weight: float = 1.0,
        latent_type: str = "vanilla",
    ):
        super().__init__()
        self.batch_embed = Embedding(n_batches, n_batch_emb)
        self.encoder = Encoder(n_genes, n_latent, n_hidden, dropout, n_batch_emb, latent_type)
        self.decoder = Decoder(n_latent, n_genes, n_batch_emb, n_hidden)
        self.age_head = AgeRegressor(n_latent, n_hidden, n_batch_emb)
        self.age_loss_weight = age_loss_weight
        self.latent_type = latent_type

    def _get_inference_input(self, tensors):
        return {
            "x": tensors[REGISTRY_KEYS.X_KEY],
            "batch_idx": tensors.get(REGISTRY_KEYS.BATCH_KEY, None),
        }

    def _get_generative_input(self, tensors: dict, inference_outputs: dict):
        return {
            "z": inference_outputs["z"],
            "batch_emb": inference_outputs["batch_emb"]
        }

    def inference(self, x, batch_idx=None):
        x = x.to(self.device)
        batch_emb = self.batch_embed(batch_idx.to(self.device).long()) if batch_idx is not None else torch.zeros(x.shape[0], self.batch_embed.embedding_dim, device=self.device)
        batch_emb = batch_emb.reshape(x.shape[0], -1)  # Ensure batch_emb is of shape (batch_size, n_batch_emb)
        z, mu, logvar = self.encoder(x, batch_emb)
        return {"z": z, "qzm": mu, "qzv": torch.exp(logvar), "batch_emb": batch_emb}

    def generative(self, z, batch_emb):
        recon_x = self.decoder(z, batch_emb)
        age_pred = self.age_head(z, batch_emb).squeeze()
        return {"recon_x": recon_x, "age_pred": age_pred}

    def loss(self, tensors, inference_outputs, generative_outputs) -> LossOutput:
        x = tensors[REGISTRY_KEYS.X_KEY]
        age_true = tensors[REGISTRY_KEYS.LABELS_KEY].float()
        recon_x = generative_outputs["recon_x"]
        age_pred = generative_outputs["age_pred"]
        qz_m = inference_outputs["qzm"]
        qz_v = inference_outputs["qzv"]

        recon_loss = F.mse_loss(recon_x, x, reduction="none").sum(dim=1)

        if self.latent_type == "vanilla":
            kl_div = torch.zeros_like(recon_loss)
        else:
            std = torch.sqrt(qz_v)
            post = Normal(qz_m, std)
            prior = Normal(torch.zeros_like(qz_m), torch.ones_like(std))
            kl_div = kl_divergence(post, prior).sum(dim=1)

        age_loss = F.mse_loss(age_pred, age_true.squeeze(), reduction="none")
        loss = (recon_loss + kl_div + self.age_loss_weight * age_loss).mean()

        return LossOutput(
            loss=loss,
            reconstruction_loss=recon_loss,
            kl_local=kl_div,
            kl_global=0.0,
            extra_metrics={"age_loss": age_loss.mean()}
        )

class VAEAgeModel(UnsupervisedTrainingMixin, BaseModelClass, VAEMixin):
    def __init__(self, adata: AnnData, **model_kwargs):
        super().__init__(adata)
        self.module = myModule(
            n_genes=self.summary_stats["n_vars"],
            n_batches=self.summary_stats["n_batch"],
            **model_kwargs,
        )
        self._model_summary_string = f"CustomVAEAge Model"
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    def setup_anndata_test(cls, adata: AnnData):
        batch_name = adata.obs[cls.batch_key].unique()
        assert len(batch_name) == 1, f"Multiple batches found: {batch_name}"
        batch_name = batch_name[0]

        adata.obs[cls.batch_key] = adata.obs[cls.batch_key].astype('category')
        adata.obs[cls.batch_key] = adata.obs[cls.batch_key].cat.set_categories(
            cls._train_batches + [batch_name]
        )
        cls.setup_anndata(adata, is_train=False, batch_key=cls.batch_key)
        

    @classmethod
    def setup_anndata(cls, adata: AnnData, is_train=True, batch_key=None, layer=None, labels_key="age", **kwargs):
        if is_train:
            cls._train_batches = adata.obs[batch_key].unique().tolist()
            cls.batch_key = batch_key
            cls.labels_key = labels_key
        cls._all_batches = adata.obs[batch_key].cat.categories.tolist()
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=False),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            NumericalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
        

    def extend_batch_embedding(self, adata: AnnData):
        batch_name = adata.obs[self.batch_key].unique()[0]

        if batch_name in self._all_batches:
            raise ValueError(f"Batch '{batch_name}' already exists in the model.")

        old_n_batches = self.module.batch_embed.num_embeddings
        print('old_n_batches: ', old_n_batches)
        
        added = 1
        self.module.batch_embed = Embedding.extend(self.module.batch_embed, init=added, freeze_prev=True)

    def predict_age(self, adata):
        self.module.eval()
        if not self.is_trained_:
            raise RuntimeError("Please train the model first.")

        try:
            adata = self._validate_anndata(adata)
        except:
            self._register_manager_for_instance(
                        self.adata_manager.transfer_fields(adata, extend_categories=True)
                )
            self._validate_anndata(adata)
            
        batch = torch.tensor(adata.obs[self.batch_key].cat.codes.values, device=self.module.device)
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        X = torch.from_numpy(X).to(self.module.device).float() if not torch.is_tensor(X) else X.float().to(self.module.device)

        with torch.no_grad():
            rr = self.module.inference(X, batch)
            latent = rr["z"]
            batch_emb = rr['batch_emb']
            age_pred = self.module.age_head(latent, batch_emb).cpu().numpy().squeeze()

        adata.obs["predicted_age"] = age_pred
        return adata
    

    def train_test(
        self,
        adata_test,
        max_epochs: int = 100,
        train_size: float = 0.9,
        validation_size: float = 0.1,
        batch_size: int = 128,
        lr: float = 0.05,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        # Prepare the new anndata for use
        # self._register_manager_for_instance(
        #     self.adata_manager.transfer_fields(adata_test, extend_categories=True)
        # )
        # self._validate_anndata(adata_test)
        # adata_test = adata_test.copy()  # prevent modifying original object

        # self.module.to(device)
        # self.module.train()

        # # Freeze all parameters
        # for param in self.module.parameters():
        #     param.requires_grad = False

        # # Unfreeze only batch embedding weights
        # embed_layer = self.module.batch_embed
        # embed_layer.weight.requires_grad = True
        # new_index = embed_layer.num_embeddings - 1

        # optimizer = torch.optim.Adam([embed_layer.weight], lr=lr)

        data_splitter = DataSplitter(
            self.adata,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
        )
        # defines optimizers, training step, val step, logged metrics
        training_plan = TrainingPlan(
            self.module,
            len(data_splitter.train_idx),
        )
        # creates Trainer, pre and post training procedures (Trainer.fit())
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            **kwargs,
        )
        return runner()

def seed_all(seed=42):
    import random
    import numpy as np
    import torch
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
