"""
Step 8: cumulative epsilon accounting.

Computed POST-HOC, independent of any live object persisting across Flower
rounds (we can't rely on that -- see run_dp.py). Given the noise multiplier
used throughout training and the total number of gradient steps taken
across every round, this reconstructs the same cumulative privacy budget
Opacus's own accountant would track if it ran continuously in one process.

delta=1e-5 is used for every client: standard convention is delta < 1/N
(N = client's dataset size); even our smallest client (Hospital, 11,422
train rows) has 1/N ~= 8.8e-5, safely above 1e-5.
"""

import math

from opacus.accountants import RDPAccountant


def compute_epsilon(noise_multiplier: float, sample_rate: float,
                     total_steps: int, delta: float = 1e-5) -> float:
    accountant = RDPAccountant()
    for _ in range(total_steps):
        accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    return accountant.get_epsilon(delta=delta)


def steps_for_client(train_size: int, batch_size: int, local_epochs: int,
                      num_rounds: int) -> int:
    steps_per_epoch = math.ceil(train_size / batch_size)
    return steps_per_epoch * local_epochs * num_rounds


def sample_rate_for_client(train_size: int, batch_size: int) -> float:
    return batch_size / train_size


if __name__ == "__main__":
    # Quick sanity check: more steps / higher noise multiplier should
    # increase epsilon (weaker privacy guarantee needs MORE noise to
    # compensate, so at fixed noise, more steps = more epsilon spent).
    for noise_mult in [0.5, 1.0, 2.0]:
        eps = compute_epsilon(noise_multiplier=noise_mult, sample_rate=0.01, total_steps=500)
        print(f"noise_multiplier={noise_mult}: epsilon={eps:.2f} (500 steps, sample_rate=0.01)")