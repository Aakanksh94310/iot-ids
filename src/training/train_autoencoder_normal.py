from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

from src.config import MODELS_DIR  # we only need this


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_CSV = PROJECT_ROOT / "data" / "processed" / "merged_placeholder.csv"


class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        # Same architecture you had before
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def train_autoencoder(
    batch_size: int = 64,
    epochs: int = 20,
    lr: float = 1e-3,
):
    # --------------------------------------------------
    # 1) Load processed merged dataset
    # --------------------------------------------------
    if not PROCESSED_CSV.exists():
        raise FileNotFoundError(f"[train_autoencoder] Processed CSV not found at: {PROCESSED_CSV}")

    df = pd.read_csv(PROCESSED_CSV)

    if "label" not in df.columns:
        raise KeyError(f"[train_autoencoder] Expected 'label' column not found in {PROCESSED_CSV}")

    # Same 8 numeric features as supervised models
    feature_cols = [
        "packet_count",
        "byte_count",
        "flow_duration",
        "avg_packet_size",
        "pkt_rate",
        "byte_rate",
        "tcp_flag_syn",
        "tcp_flag_ack",
    ]
    for col in feature_cols:
        if col not in df.columns:
            raise KeyError(f"[train_autoencoder] Expected feature column '{col}' not found in {PROCESSED_CSV}")

    # Normal-only subset (label == 0)
    normal_df = df[df["label"] == 0]
    X_normal = normal_df[feature_cols].values.astype(np.float32)

    print(f"[train_autoencoder] Normal-only shape: {X_normal.shape}")

    # --------------------------------------------------
    # 2) Train/val split + scaling
    # --------------------------------------------------
    X_train, X_val = train_test_split(
        X_normal,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    input_dim = X_train_scaled.shape[1]
    print(f"[train_autoencoder] Input dim: {input_dim}")

    # --------------------------------------------------
    # 3) PyTorch datasets/dataloaders
    # --------------------------------------------------
    X_train_tensor = torch.from_numpy(X_train_scaled.astype(np.float32))
    X_val_tensor = torch.from_numpy(X_val_scaled.astype(np.float32))

    train_ds = TensorDataset(X_train_tensor)
    val_ds = TensorDataset(X_val_tensor)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # --------------------------------------------------
    # 4) Model, optimizer, loss
    # --------------------------------------------------
    model = SimpleAutoencoder(input_dim=input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"[train_autoencoder] Using device: {DEVICE}")

    # --------------------------------------------------
    # 5) Training loop
    # --------------------------------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for (x_batch,) in train_loader:
            x_batch = x_batch.to(DEVICE)

            optimizer.zero_grad()
            recon = model(x_batch)
            loss = criterion(recon, x_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x_batch,) in val_loader:
                x_batch = x_batch.to(DEVICE)
                recon = model(x_batch)
                loss = criterion(recon, x_batch)
                val_loss += loss.item() * x_batch.size(0)

        val_loss /= len(val_loader.dataset)

        print(
            f"[train_autoencoder] Epoch {epoch:02d}/{epochs} "
            f"- train_loss: {train_loss:.6f}  val_loss: {val_loss:.6f}"
        )

    # --------------------------------------------------
    # 6) Save model + scaler
    # --------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    ae_path = MODELS_DIR / "autoencoder_normal_bot_iot.pth"
    torch.save(model.state_dict(), ae_path)
    print(f"[train_autoencoder] Saved autoencoder weights to: {ae_path}")

    scaler_path = MODELS_DIR / "autoencoder_normal_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"[train_autoencoder] Saved scaler to: {scaler_path}")


if __name__ == "__main__":
    train_autoencoder()
