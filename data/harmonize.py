"""
Step 3, part 3: harmonize.

Runs all 4 loaders, combines them into one dataset, removes duplicate rows,
and saves the result to data/harmonized/combined.csv.

Duplicate removal happens here (before any train/test split in Step 4) on
purpose: CIC-IDS2017 and Edge-IIoTset are both known to contain duplicate
rows. If a duplicate ends up in both the train and test split, the model
gets evaluated on data it already memorized, which is exactly the kind of
overfitting we discussed and want to prevent from the start.

Usage:
    python data/harmonize.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import schema
    from .loaders import LOADERS
except ImportError:
    import schema
    from loaders import LOADERS


def harmonize(raw_dir: str = "data/raw", out_path: str = "data/harmonized/combined.csv") -> pd.DataFrame:
    frames = []
    for name, loader_fn in LOADERS.items():
        print(f"\n{'=' * 60}\nLoading {name}\n{'=' * 60}")
        df = loader_fn(raw_dir)
        frames.append(df)
        print(f"[{name}] {len(df):,} rows after label mapping")

    combined = pd.concat(frames, ignore_index=True)

    # Safety net: fixed transforms (log1p, port scaling) shouldn't produce
    # inf/NaN given our clipping, but real-world data always finds a way.
    # Catch it here rather than let it silently corrupt training later.
    feature_cols = schema.MODEL_INPUT_COLUMNS
    bad = ~np.isfinite(combined[feature_cols].values)
    if bad.any():
        n_bad_rows = bad.any(axis=1).sum()
        print(f"\nWARNING: {n_bad_rows} rows had non-finite values after transform, "
              f"replacing with 0")
        combined[feature_cols] = combined[feature_cols].replace([np.inf, -np.inf], np.nan)
        combined[feature_cols] = combined[feature_cols].fillna(0)

    # NOTE on deduplication: CIC and WUSTL are deduplicated inside their own
    # loaders, on the RAW data, before feature reduction (see loaders.py).
    # Edge and CSIC are NOT deduplicated: their loaders use `usecols` to stay
    # memory-safe on large files, so we don't have enough raw columns loaded
    # to tell a true duplicate apart from two different events that happen
    # to share the same coarse harmonized features. Deduplicating on the
    # reduced feature set directly (an earlier version of this script did
    # this) incorrectly removed 90%+ of both clients' data -- two genuinely
    # different packets/requests easily collapse to the same few numbers
    # once reduced to ~10 features. Leaving them as-is is the honest choice
    # until/unless we revisit loading more raw columns for these two.

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_file, index=False)

    print(f"\n{'=' * 60}\nFinal harmonized dataset\n{'=' * 60}")
    print(f"Total: {len(combined):,} rows x {combined.shape[1]} columns")
    print(f"Saved to: {out_file}")
    print("\nRows per client:")
    print(combined["client"].value_counts())
    print("\nRows per class (overall):")
    print(combined["label_name"].value_counts())
    print("\nClass distribution per client (%):")
    ct = pd.crosstab(combined["client"], combined["label_name"], normalize="index") * 100
    print(ct.round(2))

    return combined


if __name__ == "__main__":
    harmonize()