"""
Step 5, part 2: Flower client.

One instance of this class represents one organization (bank, hospital,
etc.) in the federation. Flower calls fit() when this client should train
locally for a round, and evaluate() when it should report metrics on its
own held-out data. Flower never calls anything that would expose this
client's raw data -- only get_parameters/set_parameters (model weights)
cross the client/server boundary.
"""

from collections import OrderedDict
from pathlib import Path

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from ..models.mlp import IDSModel
    from .data_utils import load_client_split, compute_class_weights
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from models.mlp import IDSModel
    from client.data_utils import load_client_split, compute_class_weights

DEVICE = torch.device("cpu")  # CPU-only by design -- see Step 1 hardware discussion


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


# Small clients get many more local epochs per round -- their datasets are
# tiny, so extra epochs are cheap, but without this their single-epoch
# update is too weak to survive being averaged with 4 other clients each
# round (confirmed via diagnose_client.py: Hospital collapsed to a single
# predicted class under equal per-round epochs). Big clients stay at 1
# epoch -- they already see huge amounts of data per epoch, and more
# epochs here would mostly just blow up runtime.
CLIENT_CONFIG = {
    "hospital":      {"local_epochs": 10, "mu": 0.01},
    "ecommerce":     {"local_epochs": 8, "mu": 0.01},
    "bank":          {"local_epochs": 1, "mu": 0.01},
    "university":    {"local_epochs": 1, "mu": 0.01},
    "manufacturing": {"local_epochs": 1, "mu": 0.01},
}


class FedShieldClient(fl.client.NumPyClient):
    def __init__(self, client_name: str, local_epochs: int = None, batch_size: int = 256,
                 lr: float = 1e-3, weight_decay: float = 1e-4, mu: float = None):
        cfg = CLIENT_CONFIG.get(client_name, {"local_epochs": 1, "mu": 0.01})
        self.client_name = client_name
        self.local_epochs = local_epochs if local_epochs is not None else cfg["local_epochs"]
        self.mu = mu if mu is not None else cfg["mu"]  # FedProx proximal term strength
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay  # L2 regularization: part of our overfitting defense

        self.model = IDSModel().to(DEVICE)
        self.train_ds = load_client_split(client_name, "train")
        self.val_ds = load_client_split(client_name, "val")
        self.class_weights = compute_class_weights(self.train_ds).to(DEVICE)

    def get_parameters(self, config) -> list[np.ndarray]:
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        # Snapshot the global model's weights BEFORE local training starts --
        # FedProx penalizes drifting away from this point, which keeps a
        # client's local update (especially Hospital's 10 epochs) from
        # swinging so far that it destabilizes the shared global model.
        global_params = [p.clone().detach() for p in self.model.parameters()]
        self.model.train()

        loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        for _ in range(self.local_epochs):
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(self.model(X_batch), y_batch)
                if self.mu > 0:
                    prox_term = sum((p - gp).pow(2).sum()
                                     for p, gp in zip(self.model.parameters(), global_params))
                    loss = loss + (self.mu / 2) * prox_term
                loss.backward()
                optimizer.step()

        # Equal weighting: every client gets identical influence on the
        # global average, regardless of row count. log-weighting (tried
        # previously) still let Manufacturing/University dominate enough to
        # erase Hospital's signal entirely (see diagnose_client.py history).
        # Combined with more local epochs for small clients above, this is
        # the strongest fairness correction we apply while keeping one
        # shared global model (required for SecAgg/DP/MIA in later steps).
        fedavg_weight = 1000
        return get_parameters(self.model), fedavg_weight, {
            "client": self.client_name, "true_n_examples": len(self.train_ds),
        }

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.eval()

        loader = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                logits = self.model(X_batch)
                total_loss += criterion(logits, y_batch).item() * len(y_batch)
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)
                all_preds.extend(preds.tolist())
                all_labels.extend(y_batch.tolist())

        accuracy = correct / total if total else 0.0
        macro_f1 = _macro_f1(all_labels, all_preds, num_classes=7)

        return total_loss / total, total, {
            "client": self.client_name, "accuracy": accuracy, "macro_f1": macro_f1,
        }


def _macro_f1(labels: list[int], preds: list[int], num_classes: int) -> float:
    """
    Macro-F1: unweighted average of per-class F1. Used instead of plain
    accuracy because a model that always predicts "benign" would score
    80-98% accuracy on every client here (see Step 4's class distributions)
    while being useless -- macro-F1 punishes that.
    """
    f1_scores = []
    for c in range(num_classes):
        tp = sum(1 for l, p in zip(labels, preds) if l == c and p == c)
        fp = sum(1 for l, p in zip(labels, preds) if l != c and p == c)
        fn = sum(1 for l, p in zip(labels, preds) if l == c and p != c)
        support = tp + fn
        if support == 0:
            continue  # class not present in this client's val set -- skip, don't penalize
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def client_fn(client_name: str) -> FedShieldClient:
    """Factory Flower's simulation calls to create each client."""
    return FedShieldClient(client_name)


if __name__ == "__main__":
    # Quick sanity check without Flower's simulation machinery: one client,
    # one local round, confirm fit() and evaluate() both run without error.
    client = FedShieldClient("hospital")
    params = client.get_parameters(config={})
    new_params, n_examples, fit_metrics = client.fit(params, config={})
    print(f"fit() ok: trained on {n_examples} examples, metrics={fit_metrics}")
    loss, n_val, eval_metrics = client.evaluate(new_params, config={})
    print(f"evaluate() ok: val_loss={loss:.4f}, n={n_val}, metrics={eval_metrics}")