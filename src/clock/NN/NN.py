# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

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

import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, n_genes, n_batches, latent_dim, hidden_dim=128, dropout=.1):
        super().__init__()
        self.fc1 = nn.Linear(n_genes, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_idx=None):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        z = self.fc3(x)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim, n_genes, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_genes)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        recon = self.fc3(x)
        return recon

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, batch_dim, latent_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.batch_embed = nn.Embedding(batch_dim, hidden_dim)

    def forward(self, x, batch_idx=None):
        h = F.relu(self.fc1(x))
        if batch_idx is not None:
            h =  h + self.batch_embed(batch_idx)
        h = self.dropout(h)
        mu = self.fc_mu(h)
        if False:  # Use stochastic latent space
            logvar = self.fc_logvar(h)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std  # Reparameterization trick
        else:
            z = mu
            logvar = torch.zeros_like(mu)
        return z, mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        recon_x = self.fc2(h)
        return recon_x

class AgeRegressor(nn.Module):
    def __init__(self, latent_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return self.fc2(h)

class VAEAgeModel(nn.Module):
    def __init__(self, n_genes, n_batches, latent_dim=10, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.encoder = Encoder(n_genes, n_batches, latent_dim, hidden_dim, dropout)
        self.decoder = Decoder(latent_dim, n_genes, hidden_dim)
        self.age_head = AgeRegressor(latent_dim)

    def forward(self, x, batch_idx=None):
        z, mu, logvar = self.encoder(x, batch_idx)
        recon_x = self.decoder(z)
        age_pred = self.age_head(z)
        return recon_x, age_pred.squeeze(), mu, logvar

    def predict(self, X, batch_idx=None):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            recon_x, age_pred, mu, logvar = self.forward(X, batch_idx=batch_idx)
        age_pred = age_pred.detach().numpy()
        return age_pred

    def get_latent(self, X, batch_idx=None, batch_size=256):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        if batch_idx is not None and not isinstance(batch_idx, torch.Tensor):
            batch_idx = torch.tensor(batch_idx, dtype=torch.long)
        
        self.eval()
        latent_means = []

        n = X.shape[0]
        with torch.no_grad():
            for i in range(0, n, batch_size):
                xb = X[i:i+batch_size]
                bb = batch_idx[i:i+batch_size] if batch_idx is not None else None
                _, _, mu, _ = self.forward(xb, batch_idx=bb)
                latent_means.append(mu.cpu().numpy())

        latent_array = np.concatenate(latent_means, axis=0)
        
        return latent_array



# class AgeRegressor(nn.Module):
#     def __init__(self, latent_dim, hidden_dim=64):
#         super().__init__()
#         self.fc1 = nn.Linear(latent_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, 1)

#     def forward(self, z):
#         x = F.relu(self.fc1(z))
#         age = self.fc2(x)
#         return age

# class AgePredictionModel(nn.Module):
#     def __init__(self, n_genes, n_batches, latent_dim=10, hidden_dim=128, dropout=.2):
#         super().__init__()
#         self.encoder = Encoder(n_genes, n_batches, latent_dim, hidden_dim, dropout)
#         self.decoder = Decoder(latent_dim, n_genes, hidden_dim)
#         self.age_head = AgeRegressor(latent_dim)

#     def forward(self, x, batch_idx=None):
#         z = self.encoder(x, batch_idx)
#         recon_x = self.decoder(z)
#         age_pred = self.age_head(z)
#         return recon_x, age_pred.squeeze()