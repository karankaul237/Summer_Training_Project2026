"""
Why can't the models memorise? Check whether identical feature vectors carry
DIFFERENT labels (benign vs attack), which no model can fit.

For each distinct feature vector we count its labels. If a vector appears as
both benign and attack, the best any model can do on it is pick the majority
label, so accuracy is capped no matter how long it trains.

Usage:
    python experiments/check_label_conflicts.py ecommerce
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import schema

FEATURES = schema.MODEL_INPUT_COLUMNS


def conflict_stats(df: pd.DataFrame) -> dict:
    h = pd.util.hash_pandas_object(df[FEATURES], index=False)
    counts = pd.crosstab(h.to_numpy(), df["label_id"].to_numpy())  # rows: vectors, cols: labels
    conflicted = (counts > 0).sum(axis=1) > 1
    return {
        "rows": int(counts.values.sum()),
        "distinct_vectors": len(counts),
        "conflicted_vectors": int(conflicted.sum()),
        "conflicted_vector_pct": 100 * float(conflicted.mean()),
        "rows_on_conflicted_vectors_pct": 100 * float(counts[conflicted].values.sum() / counts.values.sum()),
        # best possible accuracy for ANY model on these exact feature vectors
        "accuracy_ceiling": float(counts.max(axis=1).sum() / counts.values.sum()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("client", nargs="?", default="ecommerce")
    args = parser.parse_args()

    data_dir = Path("data/partitioned") / args.client
    df = pd.concat([pd.read_csv(data_dir / f"{s}.csv") for s in ("train", "val", "test")],
                   ignore_index=True)
    s = conflict_stats(df)
    print(f"Client: {args.client}")
    print(f"  {s['rows']:,} rows, {s['distinct_vectors']:,} distinct feature vectors")
    print(f"  {s['conflicted_vectors']:,} vectors ({s['conflicted_vector_pct']:.1f}%) appear with BOTH labels")
    print(f"  those vectors cover {s['rows_on_conflicted_vectors_pct']:.1f}% of all rows")
    print(f"  best possible accuracy on these features (any model, even memorising): "
          f"{s['accuracy_ceiling']:.3f}")