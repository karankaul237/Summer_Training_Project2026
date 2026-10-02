"""
Step 5, part 3: FedAvg strategy.

Flower's built-in FedAvg already does weighted parameter averaging (each
client's model update is weighted by how many examples it trained on --
Manufacturing's 1.5M rows count for more than Hospital's 11K, same as
real-world FedAvg). What we add here is metric aggregation: combining each
client's local accuracy/macro-F1 into a round-level picture, AND keeping
the per-client breakdown (not just the average) so we can see whether one
client is being left behind by the global model -- which plain accuracy
would hide.
"""

from typing import Optional

import flwr as fl
from flwr.common import Metrics


def make_metric_aggregators(history: list[dict]):
    """
    Returns (fit_metrics_fn, evaluate_metrics_fn) for use in FedAvg.
    Both append to `history` (passed by reference) so the caller can save
    it after the simulation ends, in addition to Flower's own console log.
    """

    def fit_metrics_aggregation_fn(metrics: list[tuple[int, Metrics]]) -> Metrics:
        # fit() only returns {"client": name} -- nothing numeric to average,
        # we just confirm which clients participated this round.
        clients = [m.get("client", "?") for _, m in metrics]
        return {"participating_clients": ",".join(clients)}

    def evaluate_metrics_aggregation_fn(metrics: list[tuple[int, Metrics]]) -> Metrics:
        total_examples = sum(n for n, _ in metrics)

        for n, m in metrics:
            history.append({
                "client": m.get("client", "?"),
                "n_examples": n,
                "accuracy": m.get("accuracy", 0.0),
                "macro_f1": m.get("macro_f1", 0.0),
            })

        weighted_acc = sum(n * m.get("accuracy", 0.0) for n, m in metrics) / total_examples
        weighted_f1 = sum(n * m.get("macro_f1", 0.0) for n, m in metrics) / total_examples
        return {"weighted_accuracy": weighted_acc, "weighted_macro_f1": weighted_f1}

    return fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn


class SavingFedAvg(fl.server.strategy.FedAvg):
    """
    Identical to FedAvg, except it keeps the latest aggregated global model
    in memory (self.latest_parameters) and writes it to disk after every
    round. Needed for: (a) diagnosing per-client behavior after the fact,
    (b) Step 10's Membership Inference Attack, which needs an actual saved
    baseline model to attack.
    """

    def __init__(self, *args, save_path: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = save_path
        self.latest_parameters: Optional[fl.common.Parameters] = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            self.latest_parameters = aggregated_parameters
            if self.save_path:
                import numpy as np
                ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
                np.savez(self.save_path, *ndarrays)
        return aggregated_parameters, metrics


def build_fedavg_strategy(history: list[dict],
                           initial_parameters: Optional[fl.common.Parameters] = None,
                           num_clients: int = 5,
                           save_path: Optional[str] = None) -> SavingFedAvg:
    fit_agg, eval_agg = make_metric_aggregators(history)

    return SavingFedAvg(
        fraction_fit=1.0,        # all 5 clients train every round -- small federation, no sampling needed
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        fit_metrics_aggregation_fn=fit_agg,
        evaluate_metrics_aggregation_fn=eval_agg,
        initial_parameters=initial_parameters,
        save_path=save_path,
    )