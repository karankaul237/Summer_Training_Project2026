"""
Step 7: Krum vs. plain FedAvg, with one malicious client present.

Runs the SAME setup twice: once with plain FedAvg (expect visible damage
from the attacker), once with Krum (expect the attacker's updates to be
consistently rejected). Both runs exclude SecAgg -- see krum_strategy.py
docstring for why.

Usage:
    python experiments/run_krum_attack.py --rounds 10
    python experiments/run_krum_attack.py --rounds 10 --attacker university
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
from server.krum_strategy import build_krum_strategy

CLIENTS = ["bank", "ecommerce", "hospital", "manufacturing", "university"]
SEED = 42


def make_client_fn(attacker: str):
    def client_fn(cid: str) -> fl.client.Client:
        name = CLIENTS[int(cid)]
        return FedShieldClient(name, malicious=(name == attacker)).to_client()
    return client_fn


def run_fedavg(num_rounds: int, attacker: str) -> pd.DataFrame:
    torch.manual_seed(SEED)
    initial_parameters = fl.common.ndarrays_to_parameters(get_parameters(IDSModel()))
    history: list[dict] = []
    strategy = build_fedavg_strategy(history, initial_parameters, num_clients=len(CLIENTS))

    print(f"\n{'=' * 70}\nPlain FedAvg WITH attacker ('{attacker}')\n{'=' * 70}")
    fl.simulation.start_simulation(
        client_fn=make_client_fn(attacker), num_clients=len(CLIENTS),
        config=fl.server.ServerConfig(num_rounds=num_rounds), strategy=strategy,
        client_resources={"num_cpus": 1},
    )
    for i, row in enumerate(history):
        row["round"] = i // len(CLIENTS) + 1
    return pd.DataFrame(history)


def run_krum(num_rounds: int, attacker: str) -> tuple[pd.DataFrame, list[dict]]:
    torch.manual_seed(SEED)
    initial_parameters = fl.common.ndarrays_to_parameters(get_parameters(IDSModel()))
    history: list[dict] = []
    score_log: list[dict] = []
    strategy = build_krum_strategy(score_log, initial_parameters, num_clients=len(CLIENTS))
    # Krum has no evaluate_metrics_aggregation_fn set up by default -- reuse
    # the same aggregator pattern as FedAvg for evaluate (fit() aggregation
    # is overridden entirely by KrumStrategy itself).
    from server.strategy import make_metric_aggregators
    _, eval_agg = make_metric_aggregators(history)
    strategy.evaluate_metrics_aggregation_fn = eval_agg

    print(f"\n{'=' * 70}\nKrum WITH attacker ('{attacker}')\n{'=' * 70}")
    fl.simulation.start_simulation(
        client_fn=make_client_fn(attacker), num_clients=len(CLIENTS),
        config=fl.server.ServerConfig(num_rounds=num_rounds), strategy=strategy,
        client_resources={"num_cpus": 1},
    )
    for i, row in enumerate(history):
        row["round"] = i // len(CLIENTS) + 1
    return pd.DataFrame(history), score_log


def run(num_rounds: int, attacker: str) -> None:
    fedavg_df = run_fedavg(num_rounds, attacker)
    krum_df, score_log = run_krum(num_rounds, attacker)

    Path("results").mkdir(exist_ok=True)
    fedavg_df.to_csv("results/krum_comparison_fedavg_attacked.csv", index=False)
    krum_df.to_csv("results/krum_comparison_krum_defended.csv", index=False)
    pd.DataFrame(score_log).to_csv("results/krum_selection_scores.csv", index=False)

    print(f"\n{'=' * 70}\nFinal round comparison\n{'=' * 70}")
    print(f"\nPlain FedAvg (attacked by '{attacker}'), round {num_rounds}:")
    print(fedavg_df[fedavg_df["round"] == num_rounds][
        ["client", "accuracy", "macro_f1"]].to_string(index=False))

    print(f"\nKrum (attacked by '{attacker}'), round {num_rounds}:")
    print(krum_df[krum_df["round"] == num_rounds][
        ["client", "accuracy", "macro_f1"]].to_string(index=False))

    n_selected_attacker = sum(1 for row in score_log if row["selected"] == attacker)
    print(f"\nKrum selected the attacker's update in {n_selected_attacker}/{num_rounds} rounds.")
    print("\nSaved: results/krum_comparison_fedavg_attacked.csv")
    print("Saved: results/krum_comparison_krum_defended.csv")
    print("Saved: results/krum_selection_scores.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--attacker", default="university",
                         help="Which client simulates a compromised/malicious node")
    args = parser.parse_args()
    run(args.rounds, args.attacker)