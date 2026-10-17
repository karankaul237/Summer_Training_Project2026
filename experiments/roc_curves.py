"""
ROC curves for the dashboard.

1. Detection ROC: how well each trained global model separates attack from
   benign traffic, per client, on that client's held-out TEST split (never
   used for tuning anything). Score = 1 - P(benign); "attack" = any
   non-benign class.
2. Membership ROC: the loss-threshold attack on the saved E-commerce models,
   using the same clean member / non-member sets as mia.py.

Every curve is stored on a fixed grid of 201 false-positive rates, so the
CSVs stay small even for clients with hundreds of thousands of rows.

Writes results/roc_detection.csv and results/<client>_mia_roc.csv.
Then re-run experiments/summarize_results.py to bring them into the dashboard.

Usage:
    python experiments/roc_curves.py
"""

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import schema
from mia import split_members, losses_and_correct
from models.mlp import IDSModel

CLIENTS = ["bank", "ecommerce", "hospital", "manufacturing", "university"]
FPR_GRID = np.linspace(0, 1, 201)

# Federated global models saved by the experiments (label -> file in results/).
GLOBAL_MODELS = {
    "baseline (final)": "baseline_global_model.npz",
    "secure aggregation": "secagg_global_model.npz",
    "client dropout": "dropout_global_model.npz",
    "log weights": "baseline_global_model_logweighted.npz",
    "DP noise 1.0 (federated)": "dp_noise1.0_bs1024_ep5_r10_global_model.npz",
}
MIA_TAGS = ["plain", "dp_noise0.5", "dp_noise1.0", "dp_noise2.0", "dp_noise4.0"]


def roc_on_grid(y_true: np.ndarray, score: np.ndarray):
    """ROC curve resampled onto FPR_GRID. Returns (tpr_on_grid, auc)."""
    fpr, tpr, _ = roc_curve(y_true, score)
    return np.interp(FPR_GRID, fpr, tpr), float(roc_auc_score(y_true, score))


def curve_rows(model: str, tpr: np.ndarray, auc: float, **extra) -> list:
    return [{"model": model, **extra, "fpr": float(f), "tpr": float(t), "auc": auc}
            for f, t in zip(FPR_GRID, tpr)]


def load_global(path: Path) -> IDSModel:
    data = np.load(path)
    arrays = [data[f"arr_{i}"] for i in range(len(data.files))]
    model = IDSModel()
    keys = list(model.state_dict().keys())
    model.load_state_dict(OrderedDict((k, torch.tensor(a)) for k, a in zip(keys, arrays)), strict=True)
    return model.eval()


def attack_score(model: IDSModel, df: pd.DataFrame) -> np.ndarray:
    X = torch.tensor(df[schema.MODEL_INPUT_COLUMNS].values, dtype=torch.float32)
    with torch.no_grad():
        p_benign = torch.softmax(model(X), dim=1)[:, schema.CLASS_TO_ID["benign"]].numpy()
    return 1.0 - p_benign


def detection_roc(results: Path) -> pd.DataFrame:
    rows = []
    benign = schema.CLASS_TO_ID["benign"]
    tests = {c: pd.read_csv(Path("data/partitioned") / c / "test.csv") for c in CLIENTS}
    for label, fname in GLOBAL_MODELS.items():
        path = results / fname
        if not path.exists():
            print(f"  (missing {path}, skipping '{label}')")
            continue
        model = load_global(path)
        for client, test in tests.items():
            y = (test["label_id"].to_numpy() != benign).astype(int)
            if y.min() == y.max():
                print(f"  ({client}: test split has only one class, skipping)")
                continue
            tpr, auc = roc_on_grid(y, attack_score(model, test))
            rows += curve_rows(label, tpr, auc, client=client)
            print(f"  {label:<26} {client:<14} AUC={auc:.3f}", flush=True)
    return pd.DataFrame(rows, columns=["model", "client", "fpr", "tpr", "auc"])


def membership_roc(results: Path, client: str, seed: int = 42) -> pd.DataFrame:
    d = Path("data/partitioned") / client
    train, val, test = (pd.read_csv(d / f"{s}.csv") for s in ("train", "val", "test"))
    members, nonmembers, st = split_members(train, val, test, seed)
    print(f"  clean sets: {st['per_group']:,} member vectors vs {st['per_group']:,} non-member vectors")
    labels = np.concatenate([np.ones(len(members)), np.zeros(len(nonmembers))])
    rows = []
    for tag in MIA_TAGS:
        path = results / f"{client}_{tag}.pt"
        if not path.exists():
            print(f"  (missing {path}, skipping)")
            continue
        model = IDSModel()
        model.load_state_dict(torch.load(path, weights_only=True))
        lm, _ = losses_and_correct(model, members)
        ln, _ = losses_and_correct(model, nonmembers)
        tpr, auc = roc_on_grid(labels, -np.concatenate([lm, ln]))
        rows += curve_rows(tag, tpr, auc)
        print(f"  {tag:<14} AUC={auc:.3f}", flush=True)
    return pd.DataFrame(rows, columns=["model", "fpr", "tpr", "auc"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", default="ecommerce", help="client for the membership ROC")
    args = parser.parse_args()
    results = Path("results")

    print("Detection ROC (attack vs benign, held-out test split)")
    det = detection_roc(results)
    det.to_csv(results / "roc_detection.csv", index=False)

    print(f"\nMembership ROC ({args.client})")
    mia = membership_roc(results, args.client)
    mia.to_csv(results / f"{args.client}_mia_roc.csv", index=False)

    print(f"\nSaved: results/roc_detection.csv ({len(det):,} rows), "
          f"results/{args.client}_mia_roc.csv ({len(mia):,} rows)")
    print("Next: python experiments/summarize_results.py")