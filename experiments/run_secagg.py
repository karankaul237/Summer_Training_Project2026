"""
Step 6: FedAvg + Secure Aggregation.

Identical setup to run_baseline.py (same model, same clients, same
hyperparameters) with one difference: use_secagg=True, so each client
masks its update before sending it. The server never sees an individual
client's real update -- only the unmasked sum survives aggregation.

Expect final accuracy/macro-F1 to be in the same ballpark as the baseline
(masking doesn't change WHAT the model learns, only what the server can
see along the way) -- not bit-identical, since local training still
involves random minibatch shuffling that differs run to run.

Usage:
    python experiments/run_secagg.py --rounds 15
"""

import argparse
import sys
from pathlib import Path

import flwr as fl
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from client.flower_client import FedShieldClient, get_parameters
from models.mlp import IDSModel
from server.strategy import build_fedavg_strategy

CLIENTS = ["bank", "ecommerce", "hospital", "manufacturing", "university"]
SEED = 42


def client_fn(cid: str) -> fl.client.Client:
    client_name = CLIENTS[int(cid)]
    return FedShieldClient(client_name, use_secagg=True).to_client()


def run(num_rounds: int, out_path: str) -> None:
    torch.manual_seed(SEED)
    initial_model = IDSModel()
    initial_parameters = fl.common.ndarrays_to_parameters(get_parameters(initial_model))

    model_save_path = "results/secagg_global_model.npz"
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    strategy = build_fedavg_strategy(history, initial_parameters, num_clients=len(CLIENTS),
                                      save_path=model_save_path)

    print(f"{'=' * 70}\nFedShield-IDS FedAvg + SecAgg: {len(CLIENTS)} clients, "
          f"{num_rounds} rounds\n{'=' * 70}")

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(CLIENTS),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )

    for i, row in enumerate(history):
        row["round"] = i // len(CLIENTS) + 1

    df = pd.DataFrame(history)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)

    print(f"\n{'=' * 70}\nFinal round per-client results (SecAgg)\n{'=' * 70}")
    print(df[df["round"] == num_rounds].to_string(index=False))
    print(f"\nFull history saved to: {out_file}")
    print("\nCompare against results/baseline_history.csv -- numbers should be in the")
    print("same ballpark (not identical, due to training randomness), confirming SecAgg")
    print("doesn't change what the model learns, only what the server can observe.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--out", default="results/secagg_history.csv")
    args = parser.parse_args()
    run(args.rounds, args.out)