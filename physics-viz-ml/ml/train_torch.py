import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from models import MLP

FEATURES = ["mass", "cd", "area", "v0", "theta", "phi"]
TARGETS = ["landing_time", "range_x", "range_y", "max_height"]


def train_torch(df: pd.DataFrame, epochs: int = 50, lr: float = 1e-3, batch: int = 128):
    X = torch.tensor(df[FEATURES].values, dtype=torch.float32)
    y = torch.tensor(df[TARGETS].values, dtype=torch.float32)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    model = MLP(in_dim=X.shape[1], out_dim=y.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    for ep in range(epochs):
        tot = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
        print(f"epoch {ep+1}/{epochs} mse={tot/len(ds):.4f}")
    torch.save(model.state_dict(), "mlp.pt")
    return model


if __name__ == "__main__":
    from dataset import generate_projectile_dataset

    df = generate_projectile_dataset(n=5000)
    train_torch(df, epochs=25)