"""Validate generated partitions for row/label preservation and feature leakage.

Run from the repository root after `python partition/split.py`:
    python partition/validate_splits.py

The audit uses the canonical model-input columns, not labels or metadata, when
checking whether feature vectors cross partition boundaries.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from data.schema import MODEL_INPUT_COLUMNS

SPLIT_NAMES = ("train", "val", "test")


def expected_counts(combined_path: str) -> tuple[Counter, dict[str, Counter]]:
    """Count original rows and labels without loading all feature columns."""
    row_counts = Counter()
    label_counts: dict[str, Counter] = {}

    for chunk in pd.read_csv(
        combined_path, usecols=["client", "label_name"], chunksize=250_000
    ):
        for client, client_df in chunk.groupby("client", sort=False):
            client = str(client)
            row_counts[client] += len(client_df)
            label_counts.setdefault(client, Counter()).update(
                client_df["label_name"].astype(str).value_counts().to_dict()
            )

    return row_counts, label_counts


def audit_splits(
    combined_path: str = "data/harmonized/combined.csv",
    split_dir: str = "data/partitioned",
) -> bool:
    combined = Path(combined_path)
    base = Path(split_dir)
    if not combined.is_file():
        raise FileNotFoundError(f"Combined dataset not found: {combined}")

    expected_rows, expected_labels = expected_counts(str(combined))
    all_passed = True

    for client in sorted(expected_rows):
        client_dir = base / client
        paths = {name: client_dir / f"{name}.csv" for name in SPLIT_NAMES}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            print(f"FAIL {client}: missing split file(s): {', '.join(missing)}")
            all_passed = False
            continue

        hashes = {}
        actual_rows = 0
        actual_labels = Counter()
        sizes = {}

        print(f"\n{client}")
        for name, path in paths.items():
            frame = pd.read_csv(path, usecols=list(MODEL_INPUT_COLUMNS) + ["label_name"])
            sizes[name] = len(frame)
            actual_rows += len(frame)
            actual_labels.update(frame["label_name"].astype(str).value_counts().to_dict())

            hashes[name] = np.unique(
                pd.util.hash_pandas_object(
                    frame[list(MODEL_INPUT_COLUMNS)], index=False
                ).to_numpy(dtype="uint64")
            )
            del frame
            print(f"  {name}: {sizes[name]:,} rows ({sizes[name] / max(expected_rows[client], 1):.1%})")

        if actual_rows != expected_rows[client]:
            print(f"  FAIL row preservation: expected {expected_rows[client]:,}, got {actual_rows:,}")
            all_passed = False
        else:
            print(f"  PASS row preservation: {actual_rows:,} rows")

        if actual_labels != expected_labels[client]:
            print("  FAIL label preservation: per-class row counts differ from combined.csv")
            all_passed = False
        else:
            print("  PASS label preservation: per-class counts match combined.csv")

        client_overlap = False
        for i, left in enumerate(SPLIT_NAMES):
            for right in SPLIT_NAMES[i + 1:]:
                overlap = np.intersect1d(hashes[left], hashes[right], assume_unique=True).size
                print(f"  Shared unique feature hashes ({left}/{right}): {overlap:,}")
                if overlap:
                    client_overlap = True

        if client_overlap:
            print("  FAIL duplicate feature leakage detected")
            all_passed = False
        else:
            print("  PASS no shared model-input feature vectors")

        print("  Class distribution:")
        for name in SPLIT_NAMES:
            frame = pd.read_csv(paths[name], usecols=["label_name"])
            distribution = frame["label_name"].value_counts(normalize=True).round(3).to_dict()
            print(f"    {name}: {distribution}")
            del frame

    print("\nOVERALL:", "PASS" if all_passed else "FAIL")
    return all_passed


if __name__ == "__main__":
    raise SystemExit(0 if audit_splits() else 1)
