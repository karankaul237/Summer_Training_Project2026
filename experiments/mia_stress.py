"""
Step 10b: can we make a model that actually leaks, and does DP then help?

The first run (mia.py) found AUC ~0.50 even for the plain "leaky" model: it
did not memorise (accuracy on training vs unseen vectors: 0.827 vs 0.834).
This script searches, with cheap NON-DP training, for a setting where the
attack works: fewer distinct training vectors, a bigger network, more
epochs. If a 300-vector setting leaks, it re-runs that setting with DP-SGD
at several noise levels.

Note: training on only 300 distinct vectors is an explicit worst-case
stress test (a client with very little distinct data), not a realistic
client. With so little data, DP's epsilon at a given noise is large.

Usage:
    python experiments/mia_stress.py ecommerce
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from client.data_utils import ClientDataset, compute_class_weights
from mia import (FEATURES, DELTA, BigMLP, hash_rows, grouped_split,
                 attack_metrics, losses_and_correct)

GRID_VECTORS = [0, 300]   # 0 = all training vectors
GRID_HIDDEN = [256, 512]
GRID_EPOCHS = [100, 400]
NOISES = [0.5, 1.0, 2.0, 4.0]
LEAK_THRESHOLD = 0.55     # the LOWER end of the AUC interval must clear this
DP_VECTORS = 300          # DP is only run on this (cheap) size


def subset(train_rows, members, nonmembers, n_vectors, seed):
    """Keep n_vectors distinct training vectors (with all their copies) and n_vectors unseen ones."""
    if n_vectors and n_vectors < len(members):
        m = members.sample(n_vectors, random_state=seed)
        keep = np.isin(hash_rows(train_rows), m["_h"].to_numpy())
        return train_rows[keep].reset_index(drop=True), m, nonmembers.sample(n_vectors, random_state=seed)
    return train_rows, members, nonmembers


def choose_best(rows: list):
    """Best DP-affordable setting whose AUC interval clearly clears the threshold, else None."""
    ok = [r for r in rows if r["vectors"] == DP_VECTORS and r["auc_lo"] > LEAK_THRESHOLD]
    return max(ok, key=lambda r: r["auc"]) if ok else None


def train(rows, hidden, epochs, noise, batch_size, clip, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = ClientDataset(rows[FEATURES].values.astype(np.float32),
                       rows["label_id"].values.astype(np.int64))
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(ds))
    model = BigMLP(hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

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
    return getattr(model, "_module", model), epsilon


def attack(model, members, nonmembers, seed):
    lm, cm = losses_and_correct(model, members)
    ln, cn = losses_and_correct(model, nonmembers)
    return attack_metrics(lm, ln, cm, cn, seed)


def fmt(res):
    return (f"AUC={res['auc']:.3f} [{res['auc_lo']:.3f},{res['auc_hi']:.3f}]  "
            f"acc train/unseen={res['acc_members']:.3f}/{res['acc_nonmembers']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("client", nargs="?", default="ecommerce")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path("data/partitioned") / args.client
    all_rows = pd.concat([pd.read_csv(data_dir / f"{s}.csv") for s in ("train", "val", "test")],
                         ignore_index=True)
    train_rows, members, nonmembers, st = grouped_split(all_rows, args.seed)
    print(f"Client: {args.client} | {st['train_group_distinct']:,} training vectors, "
          f"{st['held_group_distinct']:,} held-out vectors\n")

    print("Stage A: plain (non-DP) training -- where does the attack start to work?")
    results = []
    for n_vec in GRID_VECTORS:
        rows_n, mem_n, non_n = subset(train_rows, members, nonmembers, n_vec, args.seed)
        for hidden in GRID_HIDDEN:
            for epochs in GRID_EPOCHS:
                model, _ = train(rows_n, hidden, epochs, None, args.batch_size, args.clip, args.seed)
                res = attack(model, mem_n, non_n, args.seed)
                results.append({"vectors": n_vec, "hidden": hidden, "epochs": epochs,
                                "noise_multiplier": None, "epsilon": float("inf"), **res})
                label = n_vec if n_vec else "all"
                print(f"  vectors={label!s:<4} hidden={hidden:<4} epochs={epochs:<4} {fmt(res)}", flush=True)
    out = Path("results") / f"{args.client}_mia_stress.csv"
    pd.DataFrame(results).to_csv(out, index=False)

    best = choose_best(results)
    if best is None:
        print(f"\nNo 300-vector setting leaked (needed AUC interval above {LEAK_THRESHOLD}).")
        print("Conclusion: the loss-threshold attack found no membership leakage here, even "
              "under stress. DP cannot be shown to reduce it.")
        print(f"Saved: {out}")
        sys.exit(0)

    print(f"\nStage B: DP on the setting that leaked most: hidden={best['hidden']}, "
          f"epochs={best['epochs']}, {DP_VECTORS} training vectors")
    print(f"  plain: {fmt(best)}")
    rows_n, mem_n, non_n = subset(train_rows, members, nonmembers, DP_VECTORS, args.seed)
    for noise in NOISES:
        model, eps = train(rows_n, best["hidden"], best["epochs"], noise,
                           args.batch_size, args.clip, args.seed)
        res = attack(model, mem_n, non_n, args.seed)
        results.append({"vectors": DP_VECTORS, "hidden": best["hidden"], "epochs": best["epochs"],
                        "noise_multiplier": noise, "epsilon": eps, **res})
        print(f"  DP noise={noise:<4} eps={eps:9.2f}  {fmt(res)}", flush=True)
        pd.DataFrame(results).to_csv(out, index=False)
    print(f"\nSaved: {out}")