"""
Step 4, part 1: per-client train/val/test split.

Splitting happens before model fitting. Each client's rows are split
independently. Identical model-input feature vectors are kept in the same
partition to prevent exact duplicate rows leaking between train/validation/
test, even when identical features have conflicting labels.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
_NON_FEATURE_COLUMNS = {"label_id", "label_name", "client"}


def stratified_split(df: pd.DataFrame, label_col: str = "label_id",
                      train_frac: float = 0.70, val_frac: float = 0.15,
                      seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split one client's data into train/val/test while keeping identical feature
    vectors together. Stratification is performed over unique feature groups,
    using each group's most frequent label as its representative label.

    A class with fewer than 3 unique feature groups is kept in train rather
    than being dropped. If one feature vector has conflicting labels, all of
    its rows still go to the same partition; its representative label is used
    only to approximate stratification.
    """
    if df.empty:
        empty = df.reset_index(drop=True).copy()
        return empty.copy(), empty.copy(), empty.copy()
    if label_col not in df.columns:
        raise ValueError(f"label column {label_col!r} not found")

    work = df.reset_index(drop=True).copy()
    feature_cols = [
        c for c in work.columns
        if c not in (_NON_FEATURE_COLUMNS | {label_col})
    ]
    if not feature_cols:
        raise ValueError("No feature columns available for grouping")

    # ngroup assigns the same ID to rows with identical feature values,
    # treating NaNs in the same feature positions as equal.
    work["_split_group"] = work.groupby(
        feature_cols, dropna=False, sort=False
    ).ngroup()

    group_labels = (
        work.groupby("_split_group", sort=False)[label_col]
        .agg(lambda labels: labels.mode(dropna=False).iloc[0])
        .rename("_representative_label")
        .reset_index()
    )

    rng = np.random.default_rng(seed)
    train_ids, val_ids, test_ids = [], [], []

    for _, label_groups in group_labels.groupby("_representative_label", dropna=False):
        ids = label_groups["_split_group"].to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)

        if n < 3:
            train_ids.extend(ids.tolist())
            continue

        n_train = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac))
        n_train = min(n_train, n - 2)
        n_val = min(n_val, n - n_train - 1)

        train_ids.extend(ids[:n_train].tolist())
        val_ids.extend(ids[n_train:n_train + n_val].tolist())
        test_ids.extend(ids[n_train + n_val:].tolist())

    def select(group_ids):
        return (
            work.loc[work["_split_group"].isin(group_ids)]
            .drop(columns="_split_group")
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )

    return select(train_ids), select(val_ids), select(test_ids)


def split_all_clients(combined_path: str = "data/harmonized/combined.csv",
                       out_dir: str = "data/partitioned") -> None:
    df = pd.read_csv(combined_path)
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}\nPer-client group-aware stratified split (70/15/15)\n{'=' * 60}")
    for client, client_df in df.groupby("client"):
        train, val, test = stratified_split(client_df)

        client_dir = out_base / client
        client_dir.mkdir(exist_ok=True)
        train.to_csv(client_dir / "train.csv", index=False)
        val.to_csv(client_dir / "val.csv", index=False)
        test.to_csv(client_dir / "test.csv", index=False)

        print(f"\n{client}: {len(client_df):,} total "
              f"-> train {len(train):,} / val {len(val):,} / test {len(test):,}")
        for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
            top_class = split_df["label_name"].value_counts(normalize=True).round(3).to_dict()
            print(f"    {split_name}: {top_class}")


if __name__ == "__main__":
    split_all_clients()
