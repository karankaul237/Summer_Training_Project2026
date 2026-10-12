"""
Step 10: membership inference attack (loss-threshold attack, Yeom et al. 2018).

Question: given a trained model and a record, can an attacker tell whether
that record was in the training set? The attacker only needs the model's
loss on the record: training records tend to have LOWER loss than records
the model has never seen. We score every record by -loss and measure how
well that separates members from non-members.

Reading the numbers:
  AUC 0.50 = attacker does no better than guessing (no leakage)
  AUC 1.00 = attacker is always right
  advantage = best (true-positive rate - false-positive rate) over thresholds

IMPORTANT -- duplicates. E-commerce has about 10 copies of each distinct
record. A validation row that is an exact copy of a training row gets exactly
the same loss, so calling it a "non-member" is wrong and drags the attack
towards 0.5. So membership is defined per FEATURE VECTOR, never per row.

Part 1 attacks the small models saved by dp_sweep.py, using the original
train/val/test files:
  members     = vectors in train and never in val/test
  non-members = vectors in val/test and never in train
Because of the duplicates, very few non-member vectors may exist there; the
script prints how many. The "naive" AUC (train rows vs val rows) is shown
alongside to show how much the duplicates hide.

Part 2 trains "leaky" models (bigger, no dropout, no weight decay), plain and
with DP, on a split made by distinct vector: half the distinct vectors are
the training set, the other half are guaranteed-unseen non-members. A defence
can only be shown to help against a model that actually leaks.

Usage:
    python experiments/mia.py ecommerce
    python experiments/mia.py ecommerce --skip-leaky
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from client.data_utils import ClientDataset, compute_class_weights
from data import schema
from models.mlp import IDSModel

FEATURES = schema.MODEL_INPUT_COLUMNS
DELTA = 1e-5
SAVED_TAGS = ["plain", "dp_noise0.5", "dp_noise1.0", "dp_noise2.0", "dp_noise4.0"]
LEAKY_NOISES = [None, 0.5, 1.0, 2.0, 4.0]  # None = plain training


# ----------------------------------------------------------------------
# Building clean member / non-member sets
# ----------------------------------------------------------------------

def hash_rows(df: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(df[FEATURES], index=False).to_numpy()


def _distinct(df: pd.DataFrame, h: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    out = df[mask].copy()
    out["_h"] = h[mask]
    return out.drop_duplicates("_h")


def _balance(members: pd.DataFrame, nonmembers: pd.DataFrame, seed: int):
    n = min(len(members), len(nonmembers))
    if n == 0:
        return members, nonmembers, 0
    return members.sample(n, random_state=seed), nonmembers.sample(n, random_state=seed), n


def split_members(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, seed: int = 42):
    """Part 1 sets, from the original files. Returns (members, nonmembers, stats)."""
    held = pd.concat([val, test], ignore_index=True)
    train_h, held_h = hash_rows(train), hash_rows(held)

    members = _distinct(train, train_h, ~np.isin(train_h, held_h))
    nonmembers = _distinct(held, held_h, ~np.isin(held_h, train_h))
    stats = {
        "train_rows": len(train),
        "train_distinct": len(np.unique(train_h)),
        "held_rows": len(held),
        "held_rows_copied_from_train_pct": 100 * float(np.isin(held_h, train_h).mean()),
        "members_available": len(members),
        "nonmembers_available": len(nonmembers),
    }
    members, nonmembers, stats["per_group"] = _balance(members, nonmembers, seed)
    return members, nonmembers, stats


def grouped_split(all_rows: pd.DataFrame, seed: int = 42, frac_train: float = 0.5):
    """
    Part 2 sets: split by DISTINCT vector, so no vector is on both sides.
    Returns (train_rows, members, nonmembers, stats). train_rows keeps every
    copy of every training vector (that is what the model is trained on).
    """
    h = hash_rows(all_rows)
    distinct = np.unique(h)
    rng = np.random.default_rng(seed)
    rng.shuffle(distinct)
    cut = int(len(distinct) * frac_train)
    is_train = np.isin(h, distinct[:cut])

    train_rows = all_rows[is_train].reset_index(drop=True)
    members = _distinct(all_rows, h, is_train)
    nonmembers = _distinct(all_rows, h, ~is_train)
    stats = {"distinct_total": len(distinct), "train_group_distinct": len(members),
             "held_group_distinct": len(nonmembers), "train_rows": len(train_rows)}
    members, nonmembers, stats["per_group"] = _balance(members, nonmembers, seed)
    return train_rows, members, nonmembers, stats


# ----------------------------------------------------------------------
# The attack
# ----------------------------------------------------------------------

def attack_metrics(loss_members: np.ndarray, loss_nonmembers: np.ndarray,
                   correct_members: np.ndarray, correct_nonmembers: np.ndarray,
                   seed: int = 0, n_boot: int = 200) -> dict:
    scores = -np.concatenate([loss_members, loss_nonmembers])  # higher = more member-like
    labels = np.concatenate([np.ones(len(loss_members)), np.zeros(len(loss_nonmembers))])
    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)

    rng = np.random.default_rng(seed)
    idx_m, idx_n = np.where(labels == 1)[0], np.where(labels == 0)[0]
    boot = []
    for _ in range(n_boot):
        i = np.concatenate([rng.choice(idx_m, len(idx_m)), rng.choice(idx_n, len(idx_n))])
        boot.append(roc_auc_score(labels[i], scores[i]))
    lo, hi = np.percentile(boot, [2.5, 97.5])

    return {"auc": auc, "auc_lo": float(lo), "auc_hi": float(hi),
            "advantage": float(np.max(tpr - fpr)),
            "tpr_at_5pct_fpr": float(np.interp(0.05, fpr, tpr)),
            "acc_members": float(np.mean(correct_members)),
            "acc_nonmembers": float(np.mean(correct_nonmembers))}


def losses_and_correct(model: nn.Module, df: pd.DataFrame):
    X = torch.tensor(df[FEATURES].values, dtype=torch.float32)
    y = torch.tensor(df["label_id"].values, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss = F.cross_entropy(logits, y, reduction="none").numpy()  # unweighted on purpose
        correct = (logits.argmax(dim=1) == y).numpy()
    return loss, correct


def evaluate(name: str, kind: str, noise, epsilon: float, model: nn.Module,
             sets: dict, seed: int) -> dict:
    lm, cm = losses_and_correct(model, sets["members"])
    ln, cn = losses_and_correct(model, sets["nonmembers"])
    res = attack_metrics(lm, ln, cm, cn, seed)

    naive_auc, naive_txt = float("nan"), ""
    if "naive_members" in sets:
        lm2, cm2 = losses_and_correct(model, sets["naive_members"])
        ln2, cn2 = losses_and_correct(model, sets["naive_nonmembers"])
        naive_auc = attack_metrics(lm2, ln2, cm2, cn2, seed, n_boot=2)["auc"]
        naive_txt = f"  (naive {naive_auc:.3f})"

    row = {"model": name, "kind": kind, "noise_multiplier": noise, "epsilon": epsilon,
           **res, "naive_auc": naive_auc, "n_per_group": len(sets["members"])}
    eps_txt = "inf" if np.isinf(epsilon) else f"{epsilon:.2f}"
    print(f"{name:<22} eps={eps_txt:>7}  AUC={res['auc']:.3f} [{res['auc_lo']:.3f},{res['auc_hi']:.3f}]"
          f"{naive_txt}  advantage={res['advantage']:.3f}  "
          f"acc members/non-members={res['acc_members']:.3f}/{res['acc_nonmembers']:.3f}", flush=True)
    return row


# ----------------------------------------------------------------------
# Part 2: models built to leak
# ----------------------------------------------------------------------

class BigMLP(nn.Module):
    """Bigger than IDSModel, no dropout: free to memorise its training set."""

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(schema.INPUT_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, len(schema.CLASSES)))

    def forward(self, x):
        return self.net(x)


def train_leaky(train_rows: pd.DataFrame, noise, epochs: int, batch_size: int,
                clip: float, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = ClientDataset(train_rows[FEATURES].values.astype(np.float32),
                          train_rows["label_id"].values.astype(np.int64))
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train))
    model = BigMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
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
    return getattr(model, "_module", model), epsilon


# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("client", nargs="?", default="ecommerce")
    parser.add_argument("--skip-leaky", action="store_true")
    parser.add_argument("--leaky-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path("data/partitioned") / args.client
    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "val.csv")
    test = pd.read_csv(data_dir / "test.csv")
    results_dir = Path("results")
    out = results_dir / f"{args.client}_mia_results.csv"

    members, nonmembers, st = split_members(train, val, test, args.seed)
    print(f"Client: {args.client}")
    print(f"  train rows {st['train_rows']:,} but only {st['train_distinct']:,} distinct feature vectors")
    print(f"  {st['held_rows_copied_from_train_pct']:.1f}% of val/test rows are exact copies of a training row")
    print(f"  Part 1 clean sets: {st['members_available']:,} member vectors, "
          f"{st['nonmembers_available']:,} non-member vectors -> using {st['per_group']:,} of each")
    if st["per_group"] < 100:
        print("  WARNING: very few non-member vectors, so Part 1 intervals will be wide "
              "(Part 2 does not have this problem).")
    print()

    n_naive = min(len(val), len(train))
    sets = {"members": members, "nonmembers": nonmembers,
            "naive_members": train.sample(n_naive, random_state=args.seed), "naive_nonmembers": val}

    sweep_path = results_dir / f"{args.client}_dp_sweep.csv"
    eps_lookup = (pd.read_csv(sweep_path).set_index("setting")["epsilon"].to_dict()
                  if sweep_path.exists() else {})

    rows = []
    print("Part 1: the small models saved by dp_sweep.py")
    for tag in SAVED_TAGS:
        path = results_dir / f"{args.client}_{tag}.pt"
        if not path.exists():
            print(f"  (missing {path}, skipping)")
            continue
        model = IDSModel()
        model.load_state_dict(torch.load(path))
        eps = float(eps_lookup.get(tag, float("inf") if tag == "plain" else float("nan")))
        noise = None if tag == "plain" else float(tag.replace("dp_noise", ""))
        rows.append(evaluate(f"saved_{tag}", "saved", noise, eps, model, sets, args.seed))
    pd.DataFrame(rows).to_csv(out, index=False)

    if not args.skip_leaky:
        all_rows = pd.concat([train, val, test], ignore_index=True)
        train_rows, m2, n2, st2 = grouped_split(all_rows, args.seed)
        print(f"\nPart 2: models built to leak (bigger, no dropout, no weight decay, "
              f"{args.leaky_epochs} epochs)")
        print(f"  split by distinct vector: {st2['distinct_total']:,} vectors -> "
              f"{st2['train_group_distinct']:,} train / {st2['held_group_distinct']:,} held out; "
              f"training on {st2['train_rows']:,} rows, attacking {st2['per_group']:,} members "
              f"vs {st2['per_group']:,} non-members")
        sets2 = {"members": m2, "nonmembers": n2}
        for noise in LEAKY_NOISES:
            tag = "plain" if noise is None else f"dp_noise{noise}"
            model, eps = train_leaky(train_rows, noise, args.leaky_epochs,
                                     args.batch_size, args.clip, args.seed)
            rows.append(evaluate(f"leaky_{tag}", "leaky", noise, eps, model, sets2, args.seed))
            pd.DataFrame(rows).to_csv(out, index=False)

    print(f"\nSaved: {out}")