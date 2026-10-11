"""
Step 9a: what happens to secure aggregation when a client drops out?

No training needed -- pure NumPy, runs in seconds. Four checks:

  1. all 5 clients present        -> masks cancel (the Step 6 result)
  2. one client drops AFTER masking -> its masks are left behind in the sum
  3. dropout known BEFORE masking   -> mask only among survivors, cancels again
  4. how well do the masks actually hide an individual update?

Usage:
    python experiments/secagg_dropout_demo.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from privacy.secure_aggregation import mask_parameters, CLIENTS

# Real layer shapes of our IDSModel (3,527 parameters). Values are random
# with a typical trained-weight scale; only the scale matters for this demo.
SHAPES = [(64, 18), (64,), (32, 64), (32,), (7, 32), (7,)]
WEIGHT_SCALE = 0.3
ROUND = 1


def rms(arrays) -> float:
    flat = np.concatenate([a.ravel() for a in arrays])
    return float(np.sqrt(np.mean(flat ** 2)))


def sum_updates(updates: dict, clients: list) -> list:
    return [sum(updates[c][i] for c in clients) for i in range(len(SHAPES))]


def sum_error(masked: dict, true: dict, clients: list) -> tuple[float, float]:
    """RMS gap between the sum of masked updates and the sum of true updates."""
    err = [m - t for m, t in zip(sum_updates(masked, clients), sum_updates(true, clients))]
    return rms(err), rms(sum_updates(true, clients))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    true = {c: [rng.normal(scale=WEIGHT_SCALE, size=s).astype(np.float32) for s in SHAPES]
            for c in CLIENTS}
    dropped = "hospital"
    survivors = [c for c in CLIENTS if c != dropped]

    # 1. everyone present
    masked_all = {c: mask_parameters(c, true[c], ROUND) for c in CLIENTS}
    e, scale = sum_error(masked_all, true, CLIENTS)
    print(f"1. All 5 clients present:      RMS error in the sum = {e:.2e}   (sum RMS = {scale:.2f})  -> masks cancel")

    # 2. one client vanishes after everyone masked for all 5
    e, scale = sum_error(masked_all, true, survivors)
    print(f"2. '{dropped}' drops after masking: RMS error in the sum = {e:.2e}   (sum RMS = {scale:.2f})  "
          f"-> {100 * e / scale:.1f}% corruption left by the missing client's masks")

    # 3. dropout known up front, masks built only among survivors
    masked_surv = {c: mask_parameters(c, true[c], ROUND, participants=survivors) for c in survivors}
    e, scale = sum_error(masked_surv, true, survivors)
    print(f"3. Dropout known beforehand:   RMS error in the sum = {e:.2e}   (sum RMS = {scale:.2f})  -> masks cancel again")

    # 4. how much does masking actually disguise one client's update?
    rel = [np.linalg.norm(np.concatenate([(m - t).ravel() for m, t in zip(masked_all[c], true[c])]))
           / np.linalg.norm(np.concatenate([t.ravel() for t in true[c]])) for c in CLIENTS]
    print(f"\n4. A masked update differs from the true one by {100 * np.mean(rel):,.1f}% on average")
    print("   (small % = a server sees nearly the real update; large % = the update is hidden)")