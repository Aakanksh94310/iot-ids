# src/evaluation/eval_cnn_tcn.py

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

# --------------------------------------------------
# Config
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_CSV = PROJECT_ROOT / "data" / "processed" / "merged_placeholder.csv"
SUPERVISED_METRICS_CSV = PROJECT_ROOT / "data" / "metrics" / "supervised_metrics.csv"

LABEL_COLUMN = "label"
DATASET_SOURCE = "Bot-IoT+N-BaIoT_reduced"

FEATURE_COLS = [
    "packet_count",
    "byte_count",
    "flow_duration",
    "avg_packet_size",
    "pkt_rate",
    "byte_rate",
    "tcp_flag_syn",
    "tcp_flag_ack",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 512
EPOCHS_CNN = 5
EPOCHS_TCN = 5
LR = 1e-3
MAX_TRAIN_SAMPLES = 200_000  # cap for speed


# --------------------------------------------------
# Models
# --------------------------------------------------
class CNN1D_Classifier(nn.Module):
    def __init__(self, input_len: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        # x: [batch, 1, input_len]
        h = self.net(x)          # [batch, 32, 1]
        h = h.squeeze(-1)        # [batch, 32]
        logit = self.fc(h)       # [batch, 1]
        return logit.squeeze(-1) # [batch]


class TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        # x: [batch, channels, length]
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.dropout(self.relu2(out))
        res = self.residual(x)
        return out + res


class TCN_Classifier(nn.Module):
    def __init__(self, input_len: int, channels: int = 32):
        super().__init__()
        self.block1 = TCNBlock(1, channels, dilation=1)
        self.block2 = TCNBlock(channels, channels, dilation=2)
        self.block3 = TCNBlock(channels, channels, dilation=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        # x: [batch, 1, input_len]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1)  # [batch, channels]
        logit = self.fc(x)            # [batch, 1]
        return logit.squeeze(-1)      # [batch]


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def evaluate_and_log(model_name: str,
                     dataset_source: str,
                     y_true,
                     y_pred,
                     rows_list: list):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    rows_list.append({
        "model": model_name,
        "dataset": dataset_source,
        "accuracy": acc,
        "precision_0": prec[0],
        "recall_0": rec[0],
        "f1_0": f1[0],
        "precision_1": prec[1],
        "recall_1": rec[1],
        "f1_1": f1[1],
    })


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        total_samples += xb.size(0)

    return total_loss / max(total_samples, 1)


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    preds = []
    trues = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)
            total_samples += xb.size(0)

            probs = torch.sigmoid(logits)
            preds.append(probs.cpu().numpy())
            trues.append(yb.cpu().numpy())

    avg_loss = total_loss / max(total_samples, 1)
    y_pred = (np.concatenate(preds) >= 0.5).astype(int)
    y_true = np.concatenate(trues).astype(int)

    return avg_loss, y_true, y_pred


def main():
    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    if not PROCESSED_CSV.exists():
        raise FileNotFoundError(f"Processed CSV not found at: {PROCESSED_CSV}")

    df = pd.read_csv(PROCESSED_CSV)

    if LABEL_COLUMN not in df.columns:
        raise KeyError(
            f"Expected label column '{LABEL_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    for col in FEATURE_COLS:
        if col not in df.columns:
            raise KeyError(f"Expected feature column '{col}' not found in {PROCESSED_CSV}")

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[LABEL_COLUMN].astype(int).values

    print(f"[eval_cnn_tcn] Full X shape: {X.shape}, y shape: {y.shape}")
    print("[eval_cnn_tcn] Label distribution:")
    print(pd.Series(y).value_counts())

    # --------------------------------------------------
    # Train/test split
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Optional subsample training for speed
    if len(X_train) > MAX_TRAIN_SAMPLES:
        idx = np.random.RandomState(42).choice(len(X_train), size=MAX_TRAIN_SAMPLES, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
        print(f"[eval_cnn_tcn] Subsampled train to {len(X_train)} samples")

    print("[eval_cnn_tcn] Train label distribution before balancing:")
    print(pd.Series(y_train).value_counts())

    # --------------------------------------------------
    # Balance training data via undersampling
    # --------------------------------------------------
    under = RandomUnderSampler(random_state=42)
    X_train_bal, y_train_bal = under.fit_resample(X_train, y_train)

    print("[eval_cnn_tcn] Train label distribution after undersampling:")
    print(pd.Series(y_train_bal).value_counts())

    # --------------------------------------------------
    # Scale features
    # --------------------------------------------------
    scaler = StandardScaler()
    scaler.fit(X_train_bal)

    X_train_bal = scaler.transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)

    # Reshape for 1D conv: [batch, 1, length]
    X_train_tensor = torch.from_numpy(X_train_bal.astype(np.float32)).unsqueeze(1)
    y_train_tensor = torch.from_numpy(y_train_bal.astype(np.float32))

    X_test_tensor = torch.from_numpy(X_test_scaled.astype(np.float32)).unsqueeze(1)
    y_test_tensor = torch.from_numpy(y_test.astype(np.float32))

    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    test_ds = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    input_len = X_train_tensor.shape[-1]
    print(f"[eval_cnn_tcn] Input length for CNN/TCN: {input_len}")

    metrics_rows = []

    # --------------------------------------------------
    # CNN model
    # --------------------------------------------------
    cnn_model = CNN1D_Classifier(input_len=input_len).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=LR)

    print(f"[eval_cnn_tcn] Training CNN1D_Classifier on {DEVICE}...")
    for epoch in range(1, EPOCHS_CNN + 1):
        train_loss = train_epoch(cnn_model, train_loader, criterion, optimizer)
        val_loss, _, _ = eval_epoch(cnn_model, test_loader, criterion)
        print(
            f"[eval_cnn_tcn] [CNN] Epoch {epoch:02d}/{EPOCHS_CNN} "
            f"- train_loss: {train_loss:.6f}  val_loss: {val_loss:.6f}"
        )

    # Final eval for metrics
    _, y_true_cnn, y_pred_cnn = eval_epoch(cnn_model, test_loader, criterion)
    evaluate_and_log("CNN1D_UNDER", DATASET_SOURCE, y_true_cnn, y_pred_cnn, metrics_rows)

    # --------------------------------------------------
    # TCN model
    # --------------------------------------------------
    tcn_model = TCN_Classifier(input_len=input_len).to(DEVICE)
    criterion_tcn = nn.BCEWithLogitsLoss()
    optimizer_tcn = torch.optim.Adam(tcn_model.parameters(), lr=LR)

    print(f"[eval_cnn_tcn] Training TCN_Classifier on {DEVICE}...")
    for epoch in range(1, EPOCHS_TCN + 1):
        train_loss = train_epoch(tcn_model, train_loader, criterion_tcn, optimizer_tcn)
        val_loss, _, _ = eval_epoch(tcn_model, test_loader, criterion_tcn)
        print(
            f"[eval_cnn_tcn] [TCN] Epoch {epoch:02d}/{EPOCHS_TCN} "
            f"- train_loss: {train_loss:.6f}  val_loss: {val_loss:.6f}"
        )

    _, y_true_tcn, y_pred_tcn = eval_epoch(tcn_model, test_loader, criterion_tcn)
    evaluate_and_log("TCN_UNDER", DATASET_SOURCE, y_true_tcn, y_pred_tcn, metrics_rows)

    # --------------------------------------------------
    # Save / append metrics
    # --------------------------------------------------
    metrics_df = pd.DataFrame(metrics_rows)
    SUPERVISED_METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)

    if SUPERVISED_METRICS_CSV.exists():
        existing = pd.read_csv(SUPERVISED_METRICS_CSV)
        combined = pd.concat([existing, metrics_df], ignore_index=True)
    else:
        combined = metrics_df

    combined.to_csv(SUPERVISED_METRICS_CSV, index=False)

    print(f"[eval_cnn_tcn] Saved updated supervised metrics → {SUPERVISED_METRICS_CSV}")
    print(metrics_df)


if __name__ == "__main__":
    main()
