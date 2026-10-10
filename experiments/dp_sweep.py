"""
Step 8 deliverable: the privacy-utility tradeoff, on one client.

Trains the SAME client with plain training and with DP-SGD at several noise
levels, and records epsilon, per-class recall and balanced accuracy for each.
The trained models are saved so Step 10's membership-inference attack can
reuse exactly these models.

Why one client and not the federation: with clip=1 and class-weighted loss,
the heavily imbalanced clients (Hospital, Bank, Manufacturing, University)
collapse to all-benign under DP, and averaging them drags the global model
down too. E-commerce is the balanced client (class weights only ~1.4x apart),
and diagnose_dp.py showed DP keeps ~84% of its web_attack recall there.

Usage:
    python experiments/dp_sweep.py ecommerce
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from client.data_utils import load_client_split, compute_class_weights
from data import schema
from models.mlp import IDSModel

DELTA = 1e-5
# (tag, noise_multiplier). None = plain training, no DP at all.
SETTINGS = [("plain", None), ("dp_noise0.5", 0.5), ("dp_noise1.0", 1.0),
            ("dp_noise2.0", 2.0), ("dp_noise4.0", 4.0)]


def train_local(client: str, noise, clip: float, epochs: int, batch_size: int, seed: int = 42):
    """Returns (trained model, epsilon). epsilon is inf for plain training."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train = load_client_split(client, "train")
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train))
    model = IDSModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(train, batch_size=batch_size, shuffle=True)

    engine = None
    if noise is not None:
        from opacus import PrivacyEngine
        engine = PrivacyEngine(accountant="rdp")
        model, optimizer, loader = engine.make_private(
            module=model, optimizer=optimizer, data_loader=loader,
            noise_multiplier=noise, max_grad_norm=clip)

    model.train()
    for _ in range(epochs):
        for X, y in loader:
            if len(y) == 0:
                continue
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            optimizer.step()

    epsilon = engine.get_epsilon(DELTA) if engine is not None else float("inf")
    return model, epsilon


def evaluate(model, val):
    model.eval()
    with torch.no_grad():
        preds = model(val.X).argmax(dim=1).numpy()
    true = val.y.numpy()
    recalls = {schema.CLASSES[c]: float((preds[true == c] == c).mean())
               for c in range(len(schema.CLASSES)) if (true == c).any()}
    return recalls, float(np.mean(list(recalls.values())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("client", nargs="?", default="ecommerce")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--clip", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    val = load_client_split(args.client, "val")

    print(f"Client: {args.client} | {args.epochs} epochs | batch {args.batch_size} | clip {args.clip}\n")
    rows = []
    for tag, noise in SETTINGS:
        model, eps = train_local(args.client, noise, args.clip, args.epochs, args.batch_size)
        base = getattr(model, "_module", model)  # unwrap Opacus' GradSampleModule
        recalls, balanced = evaluate(base, val)
        torch.save(base.state_dict(), out_dir / f"{args.client}_{tag}.pt")

        rows.append({"setting": tag, "noise_multiplier": noise, "clip": args.clip,
                     "epsilon": eps, "balanced_accuracy": balanced,
                     **{f"recall_{k}": v for k, v in recalls.items()}})
        rec = {k: round(v, 3) for k, v in recalls.items()}
        print(f"{tag:<12} epsilon={eps:8.2f}  balanced_acc={balanced:.3f}  recall={rec}", flush=True)

    csv_path = out_dir / f"{args.client}_dp_sweep.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path} and one .pt model per setting in {out_dir}/")