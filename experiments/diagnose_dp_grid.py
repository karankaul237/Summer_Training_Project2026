"""
Which (class-weighting, clip, noise) combination keeps learning alive under DP?

Findings so far (hospital, noise 0): clip=1 collapses to all-benign because
the class-weighted loss makes minority gradients ~7x larger than benign ones
(median norm 19 vs 2.8 at init), and clipping flattens both to 1.0. Recall
only returns once the clip is above the minority gradient norm (~clip 100).
But noise scales with the clip (std = noise_multiplier * clip), so a big
clip costs a lot of noise. This grid searches for a workable compromise,
now WITH noise on, and reports the epsilon each setting would spend.

"sqrt" weighting = square root of the usual class weights: a milder
imbalance correction, so the minority/benign gradient gap shrinks (roughly
7x -> 2.6x) and a much smaller clip can be used.

Usage:
    python experiments/diagnose_dp_grid.py hospital
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

# Gradient norms scale with the class weights, so the weaker "sqrt"
# weighting needs a much smaller clip than the full weighting.
GRID = {"sqrt": [5.0, 15.0], "full": [30.0, 100.0]}
NOISES = [0.5, 1.0, 2.0]
DELTA = 1e-5


def run(client: str, weighting: str, clip: float, noise: float,
        epochs: int, batch_size: int, seed: int = 42):
    from opacus import PrivacyEngine
    torch.manual_seed(seed)
    np.random.seed(seed)

    train = load_client_split(client, "train")
    val = load_client_split(client, "val")
    weights = compute_class_weights(train)
    if weighting == "sqrt":
        weights = weights.sqrt()
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = IDSModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(train, batch_size=batch_size, shuffle=True)
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

    model.eval()
    with torch.no_grad():
        preds = model(val.X).argmax(dim=1).numpy()
    true = val.y.numpy()
    recalls = {schema.CLASSES[c]: round(float((preds[true == c] == c).mean()), 3)
               for c in range(len(schema.CLASSES)) if (true == c).any()}
    return engine.get_epsilon(DELTA), recalls


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("client", nargs="?", default="hospital")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    print(f"Client: {args.client} | {args.epochs} epochs | batch {args.batch_size}")
    print("Reference: plain non-DP training reached spoofing_mitm recall 0.503\n")
    for weighting, clips in GRID.items():
        for clip in clips:
            for noise in NOISES:
                eps, recalls = run(args.client, weighting, clip, noise,
                                   args.epochs, args.batch_size)
                print(f"{weighting:<5} clip={clip:<6} noise={noise:<4} "
                      f"epsilon={eps:7.2f}  recall={recalls}", flush=True)