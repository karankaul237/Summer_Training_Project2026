"""
Find out WHY DP-SGD training collapses -- in minutes, on one small client,
instead of guessing and burning another multi-hour federated run.

Trains ONE client alone (no Flower, no averaging) in four modes and shows
what each predicts on that client's validation set:

  nodp        ordinary training, same loss/optimizer       <- the control
  clip_only   Opacus with clip=1.0 but ZERO noise          <- isolates clipping
  dp          clip=1.0 + noise (what we have been running)
  dp_bigclip  clip=20  + noise (class weights can survive clipping)

How to read the results:
  nodp collapses too             -> not a DP problem at all (imbalance/optimizer)
  nodp fine, clip_only collapses -> clipping is the culprit
  clip_only fine, dp collapses   -> noise is the culprit
  dp_bigclip fixes it            -> clipping was suppressing the class weights

Usage:
    python experiments/diagnose_dp.py hospital
    python experiments/diagnose_dp.py ecommerce --epochs 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from client.data_utils import load_client_split, compute_class_weights
from data import schema
from models.mlp import IDSModel

# name: (use_opacus, noise_multiplier, max_grad_norm)
MODES = {
    "nodp": (False, 0.0, 0.0),
    "clip_only": (True, 0.0, 1.0),
    "dp": (True, 1.0, 1.0),
    "dp_bigclip": (True, 1.0, 20.0),
}


def run_mode(client: str, mode: str, epochs: int, batch_size: int, seed: int = 42):
    use_opacus, noise, clip = MODES[mode]
    torch.manual_seed(seed)
    np.random.seed(seed)

    train = load_client_split(client, "train")
    val = load_client_split(client, "val")
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train))
    model = IDSModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(train, batch_size=batch_size, shuffle=True)

    if use_opacus:
        from opacus import PrivacyEngine
        model, optimizer, loader = PrivacyEngine().make_private(
            module=model, optimizer=optimizer, data_loader=loader,
            noise_multiplier=noise, max_grad_norm=clip)

    model.train()
    for _ in range(epochs):
        for X, y in loader:
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(val.X).argmax(dim=1).numpy()
    return preds, val.y.numpy()


def report(mode: str, preds: np.ndarray, true: np.ndarray) -> None:
    classes = schema.CLASSES
    shares = {classes[c]: round(float((preds == c).mean()), 3)
              for c in range(len(classes)) if (preds == c).any()}
    recalls = {classes[c]: round(float((preds[true == c] == c).mean()), 3)
               for c in range(len(classes)) if (true == c).any()}
    print(f"{mode:<11} predicted share : {shares}")
    print(f"{'':<11} recall per class: {recalls}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("client", nargs="?", default="hospital")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    print(f"Client: {args.client} | {args.epochs} epochs | batch {args.batch_size}\n")
    for mode in MODES:
        print(f"training '{mode}' ...")
        preds, true = run_mode(args.client, mode, args.epochs, args.batch_size)
        report(mode, preds, true)