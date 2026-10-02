"""
Step 5, part 1: per-client data loading for PyTorch/Flower.

Class weights are computed PER CLIENT, from that client's own train split
only -- never pooled across clients. Pooling would mean the loss function
implicitly "knows" the global distribution, which no real client could
know without seeing everyone else's data. This keeps the federated setup
honest: each client only ever uses information it actually has.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from ..data import schema
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
    import schema


class ClientDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def load_client_split(client: str, split: str,
                       partitioned_dir: str = "data/partitioned") -> ClientDataset:
    """split is one of: 'train', 'val', 'test'."""
    path = Path(partitioned_dir) / client / f"{split}.csv"
    df = pd.read_csv(path)
    X = df[schema.MODEL_INPUT_COLUMNS].values.astype(np.float32)
    y = df["label_id"].values.astype(np.int64)
    return ClientDataset(X, y)


def compute_class_weights(train_dataset: ClientDataset,
                           num_classes: int = len(schema.CLASSES),
                           cap: float = 20.0) -> torch.Tensor:
    """
    Inverse-frequency class weights from this client's OWN train split.
    Classes absent from this client get weight 0 (never predicted-for,
    since there's nothing to learn and no way to evaluate it locally).
    `cap` prevents a near-zero-count class from producing an enormous,
    training-destabilizing weight.
    """
    counts = torch.bincount(train_dataset.y, minlength=num_classes).float()
    weights = torch.zeros(num_classes)
    present = counts > 0
    weights[present] = counts.sum() / (counts[present] * present.sum())
    weights = torch.clamp(weights, max=cap)
    return weights


if __name__ == "__main__":
    for client in ["bank", "ecommerce", "hospital", "manufacturing", "university"]:
        train = load_client_split(client, "train")
        weights = compute_class_weights(train)
        print(f"\n{client}: {len(train)} train rows")
        print(f"  class weights: {weights.round(decimals=2).tolist()}")