"""
Diagnostic: what is the global model actually predicting for a given client?

If accuracy is exactly 0.0, this tells us whether the model has collapsed
to predicting one single class for everyone (a real FedAvg-under-non-IID
problem) versus something else being wrong.

Usage:
    python experiments/diagnose_client.py hospital
    python experiments/diagnose_client.py ecommerce
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from client.data_utils import load_client_split
from client.flower_client import set_parameters
from data import schema
from models.mlp import IDSModel


def load_saved_model(path: str = "results/baseline_global_model.npz") -> IDSModel:
    data = np.load(path)
    ndarrays = [data[f"arr_{i}"] for i in range(len(data.files))]
    model = IDSModel()
    set_parameters(model, ndarrays)
    model.eval()
    return model


def diagnose(client: str) -> None:
    model = load_saved_model()
    val = load_client_split(client, "val")

    with torch.no_grad():
        logits = model(val.X)
        preds = logits.argmax(dim=1).numpy()

    true = val.y.numpy()
    classes = schema.CLASSES

    print(f"{'=' * 60}\nDiagnosis for client: {client}\n{'=' * 60}")
    print(f"Val set size: {len(true)}")

    print(f"\nTrue label distribution:")
    for i, c in enumerate(classes):
        n = (true == i).sum()
        if n > 0:
            print(f"  {c}: {n} ({100 * n / len(true):.1f}%)")

    print(f"\nPREDICTED label distribution (what the model actually said):")
    for i, c in enumerate(classes):
        n = (preds == i).sum()
        if n > 0:
            print(f"  {c}: {n} ({100 * n / len(preds):.1f}%)")

    print(f"\nConfusion matrix (rows=true, cols=predicted):")
    header = "              " + "".join(f"{c[:8]:>10}" for c in classes)
    print(header)
    for i, true_class in enumerate(classes):
        row = [(((true == i) & (preds == j)).sum()) for j in range(len(classes))]
        if sum(row) > 0:
            print(f"{true_class[:12]:<14}" + "".join(f"{v:>10}" for v in row))


if __name__ == "__main__":
    client = sys.argv[1] if len(sys.argv) > 1 else "hospital"
    diagnose(client)