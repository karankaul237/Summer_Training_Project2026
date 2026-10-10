"""
Does the CLIP VALUE alone cause the collapse? Every run here uses ZERO noise,
so the only thing that varies is the clipping norm.

Last result (hospital): plain training learned (spoofing recall 0.50), but
clip=1.0 with NO noise collapsed to all-benign. So clipping at 1.0 is enough
by itself. This script finds how large the clip must be before learning
returns, and measures the real per-sample gradient norms to explain why.

Usage:
    python experiments/diagnose_dp_clip.py hospital
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

CLIPS = [1.0, 5.0, 20.0, 100.0]


def per_sample_norms(model) -> torch.Tensor:
    """L2 norm of each sample's full gradient (Opacus stores them as grad_sample)."""
    total = None
    for p in model.parameters():
        gs = getattr(p, "grad_sample", None)
        if gs is None:
            continue
        sq = gs.reshape(gs.shape[0], -1).pow(2).sum(dim=1)
        total = sq if total is None else total + sq
    return total.sqrt()


def run(client: str, clip: float, noise: float, epochs: int, batch_size: int,
        probe: bool, seed: int = 42):
    from opacus import PrivacyEngine
    torch.manual_seed(seed)
    np.random.seed(seed)

    train = load_client_split(client, "train")
    val = load_client_split(client, "val")
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train))
    model = IDSModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    model, optimizer, loader = PrivacyEngine().make_private(
        module=model, optimizer=optimizer, data_loader=loader,
        noise_multiplier=noise, max_grad_norm=clip)

    norms = {"start": {}, "end": {}}  # phase -> class id -> list of norms
    model.train()
    for ep in range(epochs):
        for i, (X, y) in enumerate(loader):
            if len(y) == 0:
                continue
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            if probe and i < 10 and ep in (0, epochs - 1):
                n = per_sample_norms(model)
                phase = "start" if ep == 0 else "end"
                for c in y.unique().tolist():
                    norms[phase].setdefault(c, []).extend(n[y == c].tolist())
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(val.X).argmax(dim=1).numpy()
    return preds, val.y.numpy(), norms


def report(clip: float, preds: np.ndarray, true: np.ndarray) -> None:
    classes = schema.CLASSES
    shares = {classes[c]: round(float((preds == c).mean()), 3)
              for c in range(len(classes)) if (preds == c).any()}
    recalls = {classes[c]: round(float((preds[true == c] == c).mean()), 3)
               for c in range(len(classes)) if (true == c).any()}
    print(f"clip={clip:<6} predicted share : {shares}")
    print(f"{'':<11} recall per class: {recalls}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("client", nargs="?", default="hospital")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    print(f"Client: {args.client} | {args.epochs} epochs | batch {args.batch_size} | noise 0\n")
    last_norms = None
    for clip in CLIPS:
        print(f"training with clip={clip} ...")
        probe = clip == CLIPS[-1]  # the loosest clip shows the true gradient scale
        preds, true, norms = run(args.client, clip, 0.0, args.epochs, args.batch_size, probe)
        report(clip, preds, true)
        if probe:
            last_norms = norms

    print("Per-sample gradient norms (from the clip=100 run, so nothing is clipped):")
    for phase in ("start", "end"):
        for c, vals in sorted(last_norms[phase].items()):
            print(f"  {phase:<5} {schema.CLASSES[c]:<14} median={np.median(vals):7.3f}  "
                  f"p90={np.percentile(vals, 90):7.3f}  (n={len(vals)})")