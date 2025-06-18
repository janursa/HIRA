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

class AgeRegressor(nn.Module):
    def __init__(self, latent_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        age = self.fc2(x)
        return age

class AgePredictionModel(nn.Module):
    def __init__(self, n_genes, n_batches, latent_dim=10, hidden_dim=128, dropout=.2):
        super().__init__()
        self.encoder = Encoder(n_genes, n_batches, latent_dim, hidden_dim, dropout)
        self.decoder = Decoder(latent_dim, n_genes, hidden_dim)
        self.age_head = AgeRegressor(latent_dim)

    def forward(self, x, batch_idx=None):
        z = self.encoder(x, batch_idx)
        recon_x = self.decoder(z)
        age_pred = self.age_head(z)
        return recon_x, age_pred.squeeze()

def train(model, X, y, batch_idx, epochs=50, lr=1e-3, batch_size=64, tmp_dir='tmp/', alpha=1.0, beta=1.0):
    import os
    import numpy as np
    from torch.utils.data import TensorDataset, DataLoader

    os.makedirs(tmp_dir, exist_ok=True)
    dataset = TensorDataset(X, y, batch_idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    recon_loss_fn = nn.MSELoss()
    age_loss_fn = nn.MSELoss()

    model.train()
    loss_store = []
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb, bb in loader:
            recon_x, age_pred = model(xb, batch_idx=bb)
            recon_loss = recon_loss_fn(recon_x, xb)
            age_loss = age_loss_fn(age_pred, yb)
            loss = alpha * recon_loss + beta * age_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        epoch_loss = total_loss / len(loader.dataset)
        scheduler.step(epoch_loss)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
        loss_store.append(epoch_loss)

    np.save(os.path.join(tmp_dir, 'loss.npy'), np.array(loss_store))
    return model

def predict(model, X, batch_idx=None):
    model.eval()
    with torch.no_grad():
        _, preds = model(X, batch_idx=batch_idx)
    return preds





# class FiLM(nn.Module):
#     def __init__(self, n_batches, hidden_dim):
#         super().__init__()
#         self.gamma = nn.Embedding(n_batches, hidden_dim)
#         self.beta = nn.Embedding(n_batches, hidden_dim)

#     def forward(self, x, batch_idx=None):
#         if batch_idx is None:
#             gamma = self.gamma.weight.mean(dim=0)
#             beta = self.beta.weight.mean(dim=0)
#             return gamma * x + beta
#         else:
#             gamma = self.gamma(batch_idx)
#             beta = self.beta(batch_idx)
#             return gamma * x + beta

# class Encoder(nn.Module):
#     def __init__(self, n_genes, n_batches, latent_dim, hidden_dim=128, dropout=.1):
#         super().__init__()
#         # self.embedding = FiLM(n_batches, hidden_dim)
#         self.fc1 = nn.Linear(n_genes, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, latent_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, batch_idx=None):
#         x = F.relu(self.fc1(x))   
#         x = self.dropout(x)           
#         # x = self.embedding(x, batch_idx)
#         x = F.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x
# class AgePredictionModel(nn.Module):
#     def __init__(self, n_genes, n_batches, latent_dim=10, hidden_dim=128, dropout=.2):
#         super().__init__()
#         self.encoder = Encoder(n_genes, n_batches, latent_dim, hidden_dim, dropout=dropout)
#         self.decoder = AgeRegressor(latent_dim)

#     def forward(self, x, batch_idx=None):
#         z = self.encoder(x, batch_idx)
#         age_pred = self.decoder(z)
#         return age_pred.squeeze()

# def train(model, X, y, batch_idx, epochs=50, lr=1e-3, batch_size=64, tmp_dir='tmp/'):
#     import os
#     import numpy as np
#     import torch
#     from torch.utils.data import TensorDataset, DataLoader
#     from torch import nn

#     os.makedirs(tmp_dir, exist_ok=True)
#     dataset = TensorDataset(X, y, batch_idx)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
#     loss_fn = nn.MSELoss()

#     model.train()
#     loss_store = []
#     for epoch in range(epochs):
#         total_loss = 0
#         for xb, yb, bb in loader:
#             pred = model(xb, batch_idx=bb)
#             loss = loss_fn(pred, yb)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item() * xb.size(0)
#         epoch_loss = total_loss / len(loader.dataset)
#         scheduler.step(epoch_loss)  # Update scheduler
#         print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

#         loss_store.append(epoch_loss)
    
#     np.save(os.path.join(tmp_dir, 'loss.npy'), np.array(loss_store))
#     return model

# def predict(model, X, batch_idx=None):
#     model.eval()
#     with torch.no_grad():
#         preds = model(X, batch_idx=batch_idx)
#     return preds

# class Encoder(nn.Module):
#     def __init__(self, n_genes, n_batches, latent_dim, hidden_dim=128, dropout=.1):
#         super().__init__()
#         self.n_batches = n_batches
#         self.embedding = FiLM(n_batches, hidden_dim)
#         self.fc1 = nn.Linear(n_genes, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, latent_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, batch_idx):
#         x = F.relu(self.fc1(x))              
#         # if batch_idx is None:
#         #     batch_idx = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
#         if batch_idx is None:
#             # Compute the mean embedding effect across all batches
#             batch_means = []
#             for b in range(self.n_batches):  # assuming FiLM has n_conditions attribute
#                 batch_b = torch.full((x.size(0),), b, dtype=torch.long, device=x.device)
#                 batch_means.append(self.embedding(x, batch_b))
#             x = torch.stack(batch_means).mean(0)
#         else:
#             x = self.embedding(x, batch_idx)    
#         x = self.embedding(x, batch_idx)
#         x = F.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

# class AgeRegressor(nn.Module):
#     def __init__(self, latent_dim, hidden_dim=64):
#         super().__init__()
#         self.fc1 = nn.Linear(latent_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, 1)
#     def forward(self, z):
#         x = F.relu(self.fc1(z))
#         return self.fc2(x)


