"""
Step 4, part 2: quantify non-IID heterogeneity across clients.

Computes Jensen-Shannon divergence between each client's class distribution
and the pooled global distribution. JS divergence (not KL) is used on
purpose: it's symmetric, bounded in [0, 1], and well-defined even when a
client is missing classes entirely (KL divergence blows up to infinity on a
zero-probability class, which every client here has several of).

Usage:
    python partition/non_iid_metrics.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ..data import schema
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
    import schema


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p, q = p + eps, q + eps
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return 0.5 * kl_pm + 0.5 * kl_qm


def compute_divergence(combined_path: str = "data/harmonized/combined.csv") -> pd.DataFrame:
    df = pd.read_csv(combined_path)
    classes = schema.CLASSES

    global_dist = df["label_name"].value_counts(normalize=True).reindex(classes, fill_value=0.0)

    rows = []
    for client, group in df.groupby("client"):
        local_dist = group["label_name"].value_counts(normalize=True).reindex(classes, fill_value=0.0)
        jsd = js_divergence(local_dist.values, global_dist.values)
        rows.append({"client": client, "n_rows": len(group),
                     "js_divergence": jsd, "js_distance": np.sqrt(jsd),
                     "n_classes_present": int((local_dist > 0).sum())})

    result = pd.DataFrame(rows).sort_values("js_divergence", ascending=False).reset_index(drop=True)
    return result, global_dist


if __name__ == "__main__":
    result, global_dist = compute_divergence()

    print(f"{'=' * 70}\nGlobal class distribution (pooled across all clients)\n{'=' * 70}")
    print(global_dist.round(4))

    print(f"\n{'=' * 70}\nPer-client non-IID divergence from global (sorted, most divergent first)\n{'=' * 70}")
    print(result.to_string(index=False))

    print(f"\nNote: higher js_divergence = more different from the global average.\n"
          f"n_classes_present out of {len(global_dist)} total classes shows how many\n"
          f"attack types each client actually has any data for.")

    # Save to results/ so this doesn't just disappear when the terminal closes --
    # Step 12's dashboard will read these same files directly.
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    result.to_csv(out_dir / "non_iid_divergence.csv", index=False)
    global_dist.rename("proportion").to_csv(out_dir / "global_class_distribution.csv")
    print(f"\nSaved: {out_dir / 'non_iid_divergence.csv'}")
    print(f"Saved: {out_dir / 'global_class_distribution.csv'}")