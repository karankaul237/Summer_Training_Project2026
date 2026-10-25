"""
Per-client train/validation/test split with duplicate-group isolation.

Identical model-input feature vectors stay in the same partition, even if
those rows have conflicting labels. Assignment is row-aware: large feature
groups are placed first, then each group is assigned to the partition that
best reduces deviations from target row counts and per-class row counts.

When a class occurs in fewer than three distinct feature groups, all groups
containing that class are kept in training so the model can learn that class.
Large indivisible groups can still make exact 70/15/15 proportions impossible.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
_NON_FEATURE_COLUMNS = {"label_id", "label_name", "client"}


def stratified_split(
    df: pd.DataFrame,
    label_col: str = "label_id",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows while keeping identical feature vectors together.

    Groups are assigned using a deterministic largest-first greedy strategy.
    Its objective balances row counts and actual label counts, rather than
    treating every unique feature group as if it had the same size.
    """
    if df.empty:
        empty = df.reset_index(drop=True).copy()
        return empty.copy(), empty.copy(), empty.copy()
    if label_col not in df.columns:
        raise ValueError(f"label column {label_col!r} not found")
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("train_frac and val_frac must be positive and sum to less than 1")

    work = df.reset_index(drop=True).copy()
    feature_cols = [
        c for c in work.columns
        if c not in (_NON_FEATURE_COLUMNS | {label_col})
    ]
    if not feature_cols:
        raise ValueError("No feature columns available for grouping")

    work["_split_group"] = work.groupby(
        feature_cols, dropna=False, sort=False
    ).ngroup()

    group_sizes = work.groupby("_split_group", sort=False).size()
    label_codes, _ = pd.factorize(work[label_col], sort=False)
    group_class_counts = pd.crosstab(
        work["_split_group"],
        pd.Series(label_codes, index=work.index, name="_label_code"),
    ).reindex(group_sizes.index, fill_value=0)

    fractions = np.array([train_frac, val_frac, 1.0 - train_frac - val_frac])
    target_rows = len(work) * fractions
    total_class_counts = group_class_counts.sum(axis=0).to_numpy(dtype=float)
    target_class_counts = fractions[:, None] * total_class_counts[None, :]

    # Keep all groups containing a class found in fewer than three groups in
    # training; otherwise a rare class could be absent from model fitting.
    group_presence = (group_class_counts > 0).sum(axis=0)
    rare_codes = set(group_presence[group_presence < 3].index.tolist())
    rare_group_ids = set(
        group_class_counts.index[
            group_class_counts.reindex(columns=list(rare_codes), fill_value=0).gt(0).any(axis=1)
        ].tolist()
    ) if rare_codes else set()

    rng = np.random.default_rng(seed)
    group_ids = group_sizes.index.to_numpy()
    # Randomized tie-breaks keep equal-size groups deterministic for a seed.
    order = np.lexsort((rng.random(len(group_ids)), -group_sizes.to_numpy()))
    ordered_ids = group_ids[order]

    assigned = [[], [], []]
    current_rows = np.zeros(3, dtype=float)
    current_class_counts = np.zeros_like(target_class_counts)

    for group_id in ordered_ids:
        size = float(group_sizes.loc[group_id])
        class_counts = group_class_counts.loc[group_id].to_numpy(dtype=float)

        if group_id in rare_group_ids:
            chosen = 0
        else:
            row_delta = (
                (current_rows + size - target_rows) ** 2
                - (current_rows - target_rows) ** 2
            ) / np.maximum(target_rows, 1.0) ** 2
            class_delta = (
                (current_class_counts + class_counts[None, :] - target_class_counts) ** 2
                - (current_class_counts - target_class_counts) ** 2
            ) / np.maximum(target_class_counts, 1.0) ** 2
            scores = row_delta + class_delta.mean(axis=1)
            chosen = int(np.argmin(scores))

        assigned[chosen].append(group_id)
        current_rows[chosen] += size
        current_class_counts[chosen] += class_counts

    def select(group_ids_for_split):
        return (
            work.loc[work["_split_group"].isin(group_ids_for_split)]
            .drop(columns="_split_group")
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )

    return tuple(select(ids) for ids in assigned)


def split_all_clients(
    combined_path: str = "data/harmonized/combined.csv",
    out_dir: str = "data/partitioned",
) -> None:
    df = pd.read_csv(combined_path)
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}\nPer-client row-aware group split (70/15/15)\n{'=' * 60}")
    for client, client_df in df.groupby("client"):
        train, val, test = stratified_split(client_df)

        client_dir = out_base / client
        client_dir.mkdir(exist_ok=True)
        train.to_csv(client_dir / "train.csv", index=False)
        val.to_csv(client_dir / "val.csv", index=False)
        test.to_csv(client_dir / "test.csv", index=False)

        print(
            f"\n{client}: {len(client_df):,} total "
            f"-> train {len(train):,} / val {len(val):,} / test {len(test):,}"
        )
        for split_name, split_df in [
            ("train", train), ("val", val), ("test", test)
        ]:
            top_class = (
                split_df["label_name"].value_counts(normalize=True).round(3).to_dict()
            )
            print(f"    {split_name}: {top_class}")


if __name__ == "__main__":
    split_all_clients()
