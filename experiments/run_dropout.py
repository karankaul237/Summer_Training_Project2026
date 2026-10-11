"""
Step 9b: plain FedAvg when clients drop out of rounds.

Same model, clients and hyperparameters as the final baseline
(run_baseline.py); the only change is that each round some clients don't
take part. Checks that training still works with partial participation.

Usage:
    python experiments/run_dropout.py --rounds 15 --dropout 0.3
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
from server.dropout_strategy import build_dropout_strategy

CLIENTS = ["bank", "ecommerce", "hospital", "manufacturing", "university"]
SEED = 42


def client_fn(cid: str) -> fl.client.Client:
    return FedShieldClient(CLIENTS[int(cid)]).to_client()


def last3(df: pd.DataFrame) -> pd.DataFrame:
    """Per-client mean over the final 3 rounds -- steadier than one round."""
    last = df["round"].max()
    return df[df["round"] > last - 3].groupby("client")[["accuracy", "macro_f1"]].mean().round(3)


def run(num_rounds: int, dropout: float) -> None:
    torch.manual_seed(SEED)
    initial = fl.common.ndarrays_to_parameters(get_parameters(IDSModel()))
    Path("results").mkdir(exist_ok=True)

    history: list[dict] = []
    participation: list[dict] = []
    strategy = build_dropout_strategy(history, participation, initial, len(CLIENTS), dropout,
                                       save_path="results/dropout_global_model.npz")

    print(f"{'=' * 70}\nFedAvg with client dropout: p={dropout}, {num_rounds} rounds\n{'=' * 70}")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(CLIENTS),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )

    for i, row in enumerate(history):
        row["round"] = i // len(CLIENTS) + 1  # all 5 clients are evaluated every round
    df = pd.DataFrame(history)
    df.to_csv("results/dropout_history.csv", index=False)

    part = pd.DataFrame([{"round": p["round"], "n_participants": len(p["clients"]),
                          "participants": ",".join(p["clients"])}
                         for p in participation])
    part.to_csv("results/dropout_participation.csv", index=False)

    print(f"\n{'=' * 70}\nWho took part each round\n{'=' * 70}")
    print(part.to_string(index=False))
    print(f"\nAverage participants per round: {part['n_participants'].mean():.2f} of {len(CLIENTS)}")

    print(f"\n{'=' * 70}\nMean of the last 3 rounds, per client\n{'=' * 70}")
    result = last3(df).add_prefix("dropout_")
    baseline_path = Path("results/baseline_history.csv")
    if baseline_path.exists():
        result = result.join(last3(pd.read_csv(baseline_path)).add_prefix("baseline_"))
    print(result.to_string())
    print("\nSaved: results/dropout_history.csv, results/dropout_participation.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()
    run(args.rounds, args.dropout)