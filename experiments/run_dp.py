"""
Step 8: FedAvg + DP-SGD at a chosen noise level.

Run this multiple times with different --noise-multiplier values to build
the privacy-utility tradeoff: e.g. 0.5 (weak privacy), 1.0 (medium), 2.0
(strong). Each run saves its own labeled result files so they don't
overwrite each other.

Usage:
    python experiments/run_dp.py --noise-multiplier 1.0 --rounds 10
"""

import argparse
import sys
from pathlib import Path

import flwr as fl
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from client.flower_client import FedShieldClient, get_parameters, CLIENT_CONFIG
from client.data_utils import load_client_split
from models.mlp import IDSModel
from server.strategy import build_fedavg_strategy
from privacy.epsilon_accounting import compute_epsilon, steps_for_client, sample_rate_for_client

CLIENTS = ["bank", "ecommerce", "hospital", "manufacturing", "university"]
SEED = 42
BATCH_SIZE = 256
LOCAL_EPOCHS = 2  # must match the uniform value FedShieldClient uses when use_dp=True


def client_fn(cid: str, noise_multiplier: float, max_grad_norm: float):
    name = CLIENTS[int(cid)]
    return FedShieldClient(name, use_dp=True, noise_multiplier=noise_multiplier,
                            max_grad_norm=max_grad_norm).to_client()


def run(num_rounds: int, noise_multiplier: float, max_grad_norm: float) -> None:
    torch.manual_seed(SEED)
    initial_parameters = fl.common.ndarrays_to_parameters(get_parameters(IDSModel()))

    model_save_path = f"results/dp_noise{noise_multiplier}_global_model.npz"
    Path("results").mkdir(exist_ok=True)
    history: list[dict] = []
    strategy = build_fedavg_strategy(history, initial_parameters, num_clients=len(CLIENTS),
                                      save_path=model_save_path)

    print(f"{'=' * 70}\nFedAvg + DP-SGD: noise_multiplier={noise_multiplier}, "
          f"{num_rounds} rounds\n{'=' * 70}")

    fl.simulation.start_simulation(
        client_fn=lambda cid: client_fn(cid, noise_multiplier, max_grad_norm),
        num_clients=len(CLIENTS),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )

    for i, row in enumerate(history):
        row["round"] = i // len(CLIENTS) + 1
    df = pd.DataFrame(history)
    out_file = Path(f"results/dp_noise{noise_multiplier}_history.csv")
    df.to_csv(out_file, index=False)

    # Compute each client's final cumulative epsilon post-hoc (see
    # epsilon_accounting.py docstring for why this is computed this way
    # rather than read from a live accountant object).
    print(f"\n{'=' * 70}\nFinal round results + privacy budget (noise={noise_multiplier})\n{'=' * 70}")
    epsilon_rows = []
    for client in CLIENTS:
        train = load_client_split(client, "train")
        n = len(train)
        steps = steps_for_client(n, BATCH_SIZE, LOCAL_EPOCHS, num_rounds)
        rate = sample_rate_for_client(n, BATCH_SIZE)
        eps = compute_epsilon(noise_multiplier, rate, steps)

        final_row = df[(df["round"] == num_rounds) & (df["client"] == client)]
        acc = final_row["accuracy"].values[0] if len(final_row) else None
        f1 = final_row["macro_f1"].values[0] if len(final_row) else None

        epsilon_rows.append({"client": client, "train_size": n, "total_steps": steps,
                              "epsilon": eps, "accuracy": acc, "macro_f1": f1})
        print(f"  {client:<15} train_size={n:>8,}  steps={steps:>5}  "
              f"epsilon={eps:>7.2f}  accuracy={acc:.3f}  macro_f1={f1:.3f}")

    eps_df = pd.DataFrame(epsilon_rows)
    eps_out = Path(f"results/dp_noise{noise_multiplier}_epsilon_summary.csv")
    eps_df.to_csv(eps_out, index=False)
    print(f"\nSaved: {out_file}")
    print(f"Saved: {eps_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--noise-multiplier", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    args = parser.parse_args()
    run(args.rounds, args.noise_multiplier, args.max_grad_norm)