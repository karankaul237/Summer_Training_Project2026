"""
Step 9b: FedAvg with client dropout.

Each round, every client independently fails to take part with probability
`dropout_prob` (at least `min_clients` always remain). Dropped clients are
simply not asked to train that round, so the server averages only the
clients that did. All 5 clients are still EVALUATED every round, so the
metrics show how well the global model serves a client even in rounds
where that client sat out.

Participation is logged by client NAME (read from each client's fit
metrics), not by Flower's client id: in recent Flower versions the proxy's
`cid` is a random large number, not the 0-4 index.
"""

import random
from typing import Optional

from server.strategy import SavingFedAvg, make_metric_aggregators


class DropoutFedAvg(SavingFedAvg):
    def __init__(self, *args, dropout_prob: float = 0.3, min_clients: int = 2,
                 seed: int = 42, participation_log: Optional[list] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout_prob = dropout_prob
        self.min_clients = min_clients
        self.seed = seed
        self.participation_log = participation_log if participation_log is not None else []

    def configure_fit(self, server_round, parameters, client_manager):
        all_clients = super().configure_fit(server_round, parameters, client_manager)
        # Deterministic per round, so a run can be reproduced exactly.
        rng = random.Random(self.seed * 1000 + server_round)
        kept = [pair for pair in all_clients if rng.random() >= self.dropout_prob]
        if len(kept) < self.min_clients:
            kept = rng.sample(all_clients, self.min_clients)
        return kept

    def aggregate_fit(self, server_round, results, failures):
        names = [fit_res.metrics.get("client", "?") for _, fit_res in results]
        self.participation_log.append({"round": server_round, "clients": names})
        return super().aggregate_fit(server_round, results, failures)


def build_dropout_strategy(history: list, participation_log: list, initial_parameters=None,
                            num_clients: int = 5, dropout_prob: float = 0.3,
                            save_path: Optional[str] = None) -> DropoutFedAvg:
    fit_agg, eval_agg = make_metric_aggregators(history)
    return DropoutFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        fit_metrics_aggregation_fn=fit_agg,
        evaluate_metrics_aggregation_fn=eval_agg,
        initial_parameters=initial_parameters,
        save_path=save_path,
        on_fit_config_fn=lambda server_round: {"server_round": server_round},
        dropout_prob=dropout_prob,
        participation_log=participation_log,
    )