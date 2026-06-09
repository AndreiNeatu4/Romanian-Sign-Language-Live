"""
Static Sign Language Classifier — Training Script
===================================================
Trains a lightweight MLP on single-frame hand keypoints (126 features).
Input data produced by data_preparation/extract_static_keypoints.py.

Why MLP instead of CNN-LSTM for static signs?
  - Static signs carry no temporal information; a sequence model gains nothing.
  - MLP converges faster and generalises better with limited per-class samples.

Saves to: models/alphabet_static/
  best_model.pth          -- best validation-accuracy checkpoint
  final_model.pth         -- last epoch weights
  class_labels.json       -- class names (copied from data dir)
  training_history.json   -- loss/accuracy per epoch
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR  = Path(__file__).parent.parent / "data" / "alphabet_static"
MODEL_DIR = Path(__file__).parent.parent / "models" / "alphabet_static"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
EPOCHS        = 120
BATCH_SIZE    = 64
LEARNING_RATE = 1e-3
DROPOUT       = 0.3
PATIENCE      = 20          # early-stopping patience


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class StaticSignMLP(nn.Module):
    """3-layer MLP for single-frame hand keypoint classification."""

    def __init__(self, input_size: int = 126, num_classes: int = 30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT / 2),

            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out  = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        correct    += (out.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        out  = model(X_batch)
        loss = criterion(out, y_batch)

        total_loss += loss.item() * len(y_batch)
        correct    += (out.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    X_train = torch.tensor(np.load(DATA_DIR / "X_train.npy"), dtype=torch.float32)
    y_train = torch.tensor(np.load(DATA_DIR / "y_train.npy"), dtype=torch.long)
    X_val   = torch.tensor(np.load(DATA_DIR / "X_val.npy"),   dtype=torch.float32)
    y_val   = torch.tensor(np.load(DATA_DIR / "y_val.npy"),   dtype=torch.long)
    X_test  = torch.tensor(np.load(DATA_DIR / "X_test.npy"),  dtype=torch.float32)
    y_test  = torch.tensor(np.load(DATA_DIR / "y_test.npy"),  dtype=torch.long)

    with open(DATA_DIR / "class_labels.json", encoding="utf-8") as f:
        class_info  = json.load(f)
    class_names = class_info["classes"]
    num_classes = len(class_names)
    feature_size = X_train.shape[1]

    print(f"  Train : {len(X_train)} samples")
    print(f"  Val   : {len(X_val)} samples")
    print(f"  Test  : {len(X_test)} samples")
    print(f"  Classes: {num_classes}  |  Features: {feature_size}")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=BATCH_SIZE)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model     = StaticSignMLP(feature_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8, verbose=True
    )

    best_val_acc   = 0.0
    no_improve     = 0
    history        = []

    print(f"\n{'Epoch':>6}  {'Train L':>8}  {'Train A':>8}  {'Val L':>8}  {'Val A':>8}")
    print("─" * 54)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(vl_acc)

        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 4),
            "train_acc":  round(tr_acc,  4),
            "val_loss":   round(vl_loss, 4),
            "val_acc":    round(vl_acc,  4),
        })

        marker = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")
            marker    = "  ← best"
            no_improve = 0
        else:
            no_improve += 1

        print(f"{epoch:>6}  {tr_loss:>8.4f}  {tr_acc:>7.2%}  {vl_loss:>8.4f}  {vl_acc:>7.2%}{marker}")

        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    # Final save
    torch.save(model.state_dict(), MODEL_DIR / "final_model.pth")

    # Test evaluation
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pth", map_location=device))
    _, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest accuracy (best model): {test_acc:.2%}")

    # Per-class accuracy on test set
    print("\nPer-class test accuracy:")
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            preds = model(X_b.to(device)).argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_true.extend(y_b.tolist())

    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)
    for i, name in enumerate(class_names):
        mask = all_true == i
        if mask.sum() == 0:
            print(f"  {name:3s}: no test samples")
        else:
            acc = (all_preds[mask] == i).mean()
            bar = "█" * int(acc * 20)
            print(f"  {name:3s}: {acc:5.1%}  {bar}")

    # Save artefacts
    with open(MODEL_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    import shutil
    shutil.copy(DATA_DIR / "class_labels.json", MODEL_DIR / "class_labels.json")

    print(f"\nSaved to: {MODEL_DIR}")
    print("  best_model.pth")
    print("  final_model.pth")
    print("  class_labels.json")
    print("  training_history.json")


if __name__ == "__main__":
    main()
