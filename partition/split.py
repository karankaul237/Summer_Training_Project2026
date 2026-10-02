"""
Step 4, part 1: per-client train/val/test split.

Split happens BEFORE anything is fit (no scaler, no model) -- this is the
overfitting safeguard from our design discussion. Our transforms (log1p,
port scaling) are fixed constants, not fitted statistics, so there's no
leakage risk there regardless, but the split still needs to happen first
so Step 5 never accidentally trains on a row that ends up in testing.

Split is stratified by label_id WITHIN each client separately (not pooled
across clients) -- each of the 5 clients gets its own 70/15/15 split, which
matches how federated learning actually works: each client only ever sees
its own data.

Usage:
    python partition/split.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42


def stratified_split(df: pd.DataFrame, label_col: str = "label_id",
                      train_frac: float = 0.70, val_frac: float = 0.15,
                      seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split one client's data into train/val/test, preserving each class's
    proportion in every split. A class with too few rows to split 3 ways
    (fewer than 3 rows) goes entirely into train rather than being dropped --
    losing a whole rare class is worse than a slightly optimistic train set.
    """
    rng = np.random.default_rng(seed)
    train_parts, val_parts, test_parts = [], [], []

    for _, group in df.groupby(label_col):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)

        if n < 3:
            train_parts.append(group.loc[idx])
            continue

        n_train = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac))
        n_train = min(n_train, n - 2)  # leave at least 1 for val and 1 for test

        train_parts.append(group.loc[idx[:n_train]])
        val_parts.append(group.loc[idx[n_train:n_train + n_val]])
        test_parts.append(group.loc[idx[n_train + n_val:]])

    def _concat(parts):
        if not parts:
            return df.iloc[0:0]
        return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

    return _concat(train_parts), _concat(val_parts), _concat(test_parts)


def split_all_clients(combined_path: str = "data/harmonized/combined.csv",
                       out_dir: str = "data/partitioned") -> None:
    df = pd.read_csv(combined_path)
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}\nPer-client stratified split (70/15/15)\n{'=' * 60}")
    for client, client_df in df.groupby("client"):
        train, val, test = stratified_split(client_df)

        client_dir = out_base / client
        client_dir.mkdir(exist_ok=True)
        train.to_csv(client_dir / "train.csv", index=False)
        val.to_csv(client_dir / "val.csv", index=False)
        test.to_csv(client_dir / "test.csv", index=False)

        print(f"\n{client}: {len(client_df):,} total "
              f"-> train {len(train):,} / val {len(val):,} / test {len(test):,}")
        # Sanity check: class proportions should look near-identical across splits
        for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
            top_class = split_df["label_name"].value_counts(normalize=True).round(3).to_dict()
            print(f"    {split_name}: {top_class}")


if __name__ == "__main__":
    split_all_clients()