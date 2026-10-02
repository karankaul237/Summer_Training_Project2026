"""
Step 5, part 3: baseline FedAvg -- the first full federated training run.

No privacy yet (that's Steps 6-8). This is deliberately the "undefended"
version: Step 10's Membership Inference Attack runs against this exact
baseline, so we have a clean before/after once SecAgg and DP are added.

Usage:
    python experiments/run_baseline.py
    python experiments/run_baseline.py --rounds 15
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
    """Flower calls this once per client; cid is '0'..'4', mapped to our names."""
    client_name = CLIENTS[int(cid)]
    return FedShieldClient(client_name).to_client()


def run(num_rounds: int, out_path: str) -> None:
    torch.manual_seed(SEED)  # reproducible initial global model
    initial_model = IDSModel()
    initial_parameters = fl.common.ndarrays_to_parameters(get_parameters(initial_model))

    model_save_path = "results/baseline_global_model.npz"
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    strategy = build_fedavg_strategy(history, initial_parameters, num_clients=len(CLIENTS),
                                      save_path=model_save_path)

    print(f"{'=' * 70}\nFedShield-IDS baseline FedAvg: {len(CLIENTS)} clients, "
          f"{num_rounds} rounds\n{'=' * 70}")

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(CLIENTS),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1},  # CPU-only, see Step 1 hardware discussion
    )

    # history currently has one row per (round, client) -- recover round
    # numbers from row order (evaluate_metrics_aggregation_fn appends in
    # round order, len(CLIENTS) rows per round).
    for i, row in enumerate(history):
        row["round"] = i // len(CLIENTS) + 1

    df = pd.DataFrame(history)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)

    print(f"\n{'=' * 70}\nFinal round per-client results\n{'=' * 70}")
    print(df[df["round"] == num_rounds].to_string(index=False))
    print(f"\nFull history saved to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--out", default="results/baseline_history.csv")
    args = parser.parse_args()
    run(args.rounds, args.out)