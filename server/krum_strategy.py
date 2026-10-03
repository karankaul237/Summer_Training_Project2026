"""
Step 7: Krum -- Byzantine-robust aggregation.

Unlike FedAvg, which blends all clients' updates together (letting one
poisoned update corrupt the result), Krum SELECTS a single client's update
each round: whichever one is most "consistent" with the others. For each
client, Krum sums squared distances to its n-f-2 closest neighbors (f =
assumed max malicious clients); the client with the SMALLEST such score
wins, since an honest update should look similar to most other honest
updates, while a poisoned update tends to stand apart from the pack.

With n=5 clients, Krum's requirement n > 2f+2 means f can be at most 1 --
i.e. this defends against up to 1 malicious client out of 5, not more.

NOTE: this strategy requires seeing each client's individual plaintext
update to compute distances -- it is fundamentally incompatible with
SecAgg (Step 6) run simultaneously, since SecAgg's entire purpose is to
hide individual updates from the server. See README for discussion.
"""

from pathlib import Path
from typing import Optional

import flwr as fl
import numpy as np
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy


def _flatten(ndarrays: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([a.flatten() for a in ndarrays])


class KrumStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, num_malicious: int = 1, save_path: Optional[str] = None,
                 score_log: Optional[list] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_malicious = num_malicious
        self.save_path = save_path
        self.score_log = score_log if score_log is not None else []
        self.latest_parameters: Optional[Parameters] = None

    def aggregate_fit(self, server_round: int,
                       results: list[tuple[ClientProxy, FitRes]],
                       failures: list) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        if not results:
            return None, {}

        client_names = [fit_res.metrics.get("client", f"client_{i}")
                         for i, (_, fit_res) in enumerate(results)]
        flat_updates = [
            _flatten(fl.common.parameters_to_ndarrays(fit_res.parameters))
            for _, fit_res in results
        ]

        n = len(flat_updates)
        k = max(1, n - self.num_malicious - 2)  # neighbors considered per client

        scores = []
        for i in range(n):
            dists = sorted(
                np.sum((flat_updates[i] - flat_updates[j]) ** 2)
                for j in range(n) if j != i
            )
            scores.append(sum(dists[:k]))

        winner_idx = int(np.argmin(scores))
        winner_name = client_names[winner_idx]

        self.score_log.append({
            "round": server_round,
            **{f"score_{name}": s for name, s in zip(client_names, scores)},
            "selected": winner_name,
        })
        print(f"[Krum] Round {server_round}: scores={dict(zip(client_names, [round(s, 2) for s in scores]))}"
              f" -> selected '{winner_name}'")

        selected_parameters = results[winner_idx][1].parameters
        self.latest_parameters = selected_parameters

        if self.save_path:
            ndarrays = fl.common.parameters_to_ndarrays(selected_parameters)
            np.savez(self.save_path, *ndarrays)

        return selected_parameters, {"selected_client": winner_name}


def build_krum_strategy(score_log: list, initial_parameters: Optional[Parameters] = None,
                         num_clients: int = 5, num_malicious: int = 1,
                         save_path: Optional[str] = None) -> KrumStrategy:
    return KrumStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
        num_malicious=num_malicious,
        save_path=save_path,
        score_log=score_log,
        on_fit_config_fn=lambda server_round: {"server_round": server_round},
    )