
def train(model, X, y, batch_idx, epochs=50, lr=1e-3, batch_size=64, tmp_dir='tmp/', alpha=1.0, beta=1.0, gamma=1.0):
    import os
    import numpy as np
    import torch
    import torch.nn as nn
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
            recon_x, age_pred, mu, logvar = model(xb, batch_idx=bb)

            # Loss components
            recon_loss = recon_loss_fn(recon_x, xb)
            age_loss = age_loss_fn(age_pred, yb)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            # Combined loss
            loss = alpha * recon_loss + beta * age_loss + gamma * kl_loss

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