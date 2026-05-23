"""
Train GestureNet on synthetic gesture data.

Outputs:
  models/gesture_net.pt   — PyTorch state-dict checkpoint
  models/training_log.txt — per-epoch loss/accuracy log

Usage:
  python -m trainer.train               # default settings
  python -m trainer.train --epochs 100 --lr 1e-3
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trainer.data_generator  import generate_dataset, GESTURE_NAMES
from trainer.feature_engineer import extract_batch
from trainer.model            import GestureNet

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")


def build_tensors(n_per_class: int = 1200, seed: int = 42):
    print(f"Generating dataset ({n_per_class} samples × 8 classes)…")
    events, labels = generate_dataset(n_per_class=n_per_class, seed=seed)
    X = extract_batch(events)
    y = np.array(labels, dtype=np.int64)
    print(f"  feature matrix: {X.shape}, label distribution: {np.bincount(y)}")
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )


def evaluate(model: GestureNet, loader: DataLoader, criterion: nn.Module) -> tuple:
    model.eval()
    total_loss  = 0.0
    total_right = 0
    total       = 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            loss   = criterion(logits, yb)
            total_loss  += loss.item() * len(yb)
            total_right += (logits.argmax(dim=1) == yb).sum().item()
            total       += len(yb)
    return total_loss / total, total_right / total


def train(
    epochs:      int   = 80,
    lr:          float = 3e-3,
    batch_size:  int   = 128,
    val_split:   float = 0.15,
    n_per_class: int   = 1200,
    seed:        int   = 42,
) -> GestureNet:
    torch.manual_seed(seed)

    X, y = build_tensors(n_per_class=n_per_class, seed=seed)
    ds   = TensorDataset(X, y)

    n_val  = int(len(ds) * val_split)
    n_tr   = len(ds) - n_val
    tr_ds, val_ds = random_split(ds, [n_tr, n_val],
                                 generator=torch.Generator().manual_seed(seed))

    tr_loader  = DataLoader(tr_ds,  batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model     = GestureNet(dropout_p=0.2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(MODELS_DIR, exist_ok=True)
    log_path  = os.path.join(MODELS_DIR, "training_log.txt")

    print(f"\nTraining for {epochs} epochs  (train={n_tr}, val={n_val})\n")
    print(f"{'Epoch':>6}  {'tr_loss':>8}  {'tr_acc':>7}  {'val_loss':>9}  {'val_acc':>8}  {'s':>5}")

    best_val_acc = 0.0
    best_state   = None

    with open(log_path, "w") as log:
        log.write("epoch,tr_loss,tr_acc,val_loss,val_acc\n")

        for epoch in range(1, epochs + 1):
            model.train()
            t0 = time.time()

            for xb, yb in tr_loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

            scheduler.step()
            tr_loss, tr_acc   = evaluate(model, tr_loader,  criterion)
            val_loss, val_acc = evaluate(model, val_loader, criterion)
            elapsed           = time.time() - t0

            print(f"{epoch:>6d}  {tr_loss:>8.4f}  {tr_acc:>6.1%}  "
                  f"{val_loss:>9.4f}  {val_acc:>7.1%}  {elapsed:>5.1f}s")
            log.write(f"{epoch},{tr_loss:.6f},{tr_acc:.6f},{val_loss:.6f},{val_acc:.6f}\n")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"\nBest val accuracy: {best_val_acc:.1%}")

    # Restore best weights
    model.load_state_dict(best_state)  # type: ignore[arg-type]

    ckpt_path = os.path.join(MODELS_DIR, "gesture_net.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved → {ckpt_path}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,   default=80)
    parser.add_argument("--lr",          type=float, default=3e-3)
    parser.add_argument("--batch-size",  type=int,   default=128)
    parser.add_argument("--n-per-class", type=int,   default=1200)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    train(
        epochs      = args.epochs,
        lr          = args.lr,
        batch_size  = args.batch_size,
        n_per_class = args.n_per_class,
        seed        = args.seed,
    )


if __name__ == "__main__":
    main()
