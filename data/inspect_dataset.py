"""
Step 2: Dataset inspection utility.

Run this against each raw dataset AFTER you've downloaded it into
data/raw/<dataset_name>/. It reports the structure we need before we can
design the feature harmonization schema in Step 3: shape, column names,
dtypes, missing values, and label distribution.

Usage:
    python data/inspect_dataset.py data/raw/cic_ids2017/some_file.csv
    python data/inspect_dataset.py data/raw/wustl_ehms/wustl_ehms_2020.csv --label Label
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def inspect(path: str, label_col: str | None, nrows: int | None) -> None:
    file_path = Path(path)
    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"Inspecting: {file_path}")
    print(f"{'=' * 70}")

    df = pd.read_csv(file_path, nrows=nrows, low_memory=False)

    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    print(f"\nColumns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  - {col!r}  (dtype: {df[col].dtype})")

    print("\nMissing values per column (top 10 worst):")
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        print("  None")
    else:
        for col, count in missing.head(10).items():
            pct = 100 * count / len(df)
            print(f"  - {col}: {count:,} ({pct:.2f}%)")

    # Try to auto-detect the label column if not given
    if label_col is None:
        candidates = [c for c in df.columns if c.strip().lower()
                      in ("label", "labels", "attack", "attack_type", "class", "target")]
        label_col = candidates[0] if candidates else None

    if label_col and label_col in df.columns:
        print(f"\nLabel distribution (column: {label_col!r}):")
        counts = df[label_col].value_counts()
        for label, count in counts.items():
            pct = 100 * count / len(df)
            print(f"  - {label!r}: {count:,} ({pct:.2f}%)")
    else:
        print("\nNo label column detected/specified. "
              "Pass --label <column_name> explicitly and re-run.")

    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a raw dataset CSV.")
    parser.add_argument("path", help="Path to the CSV file")
    parser.add_argument("--label", default=None, help="Name of the label column")
    parser.add_argument("--nrows", type=int, default=None,
                         help="Only read the first N rows (useful for huge files)")
    args = parser.parse_args()
    inspect(args.path, args.label, args.nrows)
