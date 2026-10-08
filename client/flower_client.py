"""
Flower client (Steps 5-8).

One instance of this class represents one organization (bank, hospital,
etc.) in the federation. Flower calls fit() when this client should train
locally for a round, and evaluate() when it should report metrics on its
own held-out data. Flower never calls anything that would expose this
client's raw data -- only get_parameters/set_parameters (model weights)
cross the client/server boundary.

Optional behaviours, all off by default so the plain baseline is unchanged:
    use_secagg  Step 6: mask the update before returning it
    malicious   Step 7: simulate a compromised client (sign-flip attack)
    use_dp      Step 8: train with DP-SGD via Opacus
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
                 lr: float = 1e-3, weight_decay: float = 1e-4, mu: float = None,
                 use_secagg: bool = False, malicious: bool = False, attack_scale: float = 20.0,
                 use_dp: bool = False, noise_multiplier: float = 1.0, max_grad_norm: float = 1.0):
        self.use_dp = use_dp
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.total_dp_steps = 0  # informational only; authoritative count computed post-hoc

        if use_dp:
            # Uniform settings across all clients, FedProx off -- isolates
            # the DP noise as the only variable under study for this
            # experiment. Mixing in Step 5/7's adaptive epochs and FedProx
            # would make it impossible to attribute an accuracy change to
            # noise specifically versus one of those other mechanisms.
            cfg = {"local_epochs": 2, "mu": 0.0}
        else:
            cfg = CLIENT_CONFIG.get(client_name, {"local_epochs": 1, "mu": 0.01})
        self.client_name = client_name
        self.local_epochs = local_epochs if local_epochs is not None else cfg["local_epochs"]
        self.mu = mu if mu is not None else cfg["mu"]  # FedProx proximal term strength
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay  # L2 regularization: part of our overfitting defense
        self.use_secagg = use_secagg      # Step 6: mask updates before returning them
        self.malicious = malicious        # Step 7: simulate a compromised/attacking client
        self.attack_scale = attack_scale

        self.model = IDSModel().to(DEVICE)
        self.train_ds = load_client_split(client_name, "train")
        self.val_ds = load_client_split(client_name, "val")
        self.class_weights = compute_class_weights(self.train_ds).to(DEVICE)

    def get_parameters(self, config) -> list[np.ndarray]:
        return get_parameters(self.model)

    def fit(self, parameters, config):
        # NumPy copy of the global weights as received -- used by the Step 7
        # attack below (final_params is also NumPy, so the two can be mixed).
        global_params_np = [p.copy() for p in parameters]

        set_parameters(self.model, parameters)
        # Separate torch-tensor snapshot for FedProx: its loss term does
        # arithmetic against the LIVE, autograd-tracked model parameters,
        # which only works with tensors, not NumPy arrays.
        global_params = [p.clone().detach() for p in self.model.parameters()]
        self.model.train()

        loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        if self.use_dp:
            # Opacus wraps model/optimizer/loader to do per-sample gradient
            # clipping + calibrated noise injection (DP-SGD). A fresh
            # PrivacyEngine each fit() call is intentional: the client object
            # itself is fresh each round anyway (Flower recreates it), so
            # there's no hook-accumulation risk.
            # NOTE: make_private() rebuilds the loader from the dataset with
            # uniform Poisson sampling -- any custom sampler passed in is
            # ignored, so class imbalance cannot be handled via sampling here.
            from opacus import PrivacyEngine
            privacy_engine = PrivacyEngine()
            dp_model, dp_optimizer, dp_loader = privacy_engine.make_private(
                module=self.model, optimizer=optimizer, data_loader=loader,
                noise_multiplier=self.noise_multiplier, max_grad_norm=self.max_grad_norm,
            )
            steps_this_round = 0
            for _ in range(self.local_epochs):
                for X_batch, y_batch in dp_loader:
                    X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                    dp_optimizer.zero_grad()
                    loss = criterion(dp_model(X_batch), y_batch)
                    loss.backward()
                    dp_optimizer.step()
                    steps_this_round += 1
            self.total_dp_steps += steps_this_round
            # dp_model wraps self.model (same underlying parameters), so the
            # trained weights are read back from self.model below unchanged.
        else:
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
        # global average, regardless of row count. Required for SecAgg's
        # masks to cancel (Step 6) and for fairness to small clients (Step 5).
        fedavg_weight = 1000
        final_params = get_parameters(self.model)

        if self.malicious:
            # Sign-flipping / model-replacement attack: invert and amplify the
            # honestly-learned direction, actively pushing the global model
            # AWAY from what honest clients are learning.
            delta = [fp - gp for fp, gp in zip(final_params, global_params_np)]
            final_params = [gp - self.attack_scale * d
                             for gp, d in zip(global_params_np, delta)]

        if self.use_secagg:
            from privacy.secure_aggregation import mask_parameters
            server_round = config.get("server_round", 1)
            final_params = mask_parameters(self.client_name, final_params, server_round)

        return final_params, fedavg_weight, {
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