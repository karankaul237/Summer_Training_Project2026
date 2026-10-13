"""
Step 11: collect every experiment's raw CSV in results/ into clean summary
tables, one per experiment, plus a readable markdown report.

Reads only files that already exist; anything missing is skipped and listed
at the end, so one absent file never stops the rest. Writes to
results/summary/ (new folder, so raw results stay untouched).

Windows used for per-client numbers:
  - training histories: mean of the LAST 3 rounds (steadier than one round,
    since quality jumps around from round to round)
  - Krum comparison: the FINAL round only (10 rounds, and its behaviour is
    about which client got locked in at the end)

Usage:
    python experiments/summarize_results.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ATTACKER = "university"  # the client run_krum_attack.py made malicious (its default)

# Wall-clock seconds copied by hand from the console logs ("Run finished 15
# round(s) in ...s"); no CSV stores them. Edit here if you re-run either.
RUNTIMES_S = {"baseline": 2415.11, "secagg": 2434.39}

# The three federated DP runs from run_dp.py (first version) plus the
# large-batch retry. (noise multiplier, batch size, local epochs) -- rounds
# are read from each run's history file.
DP_RUNS = {
    "dp_noise0.5": (0.5, 256, 2),
    "dp_noise1.0": (1.0, 256, 2),
    "dp_noise2.0": (2.0, 256, 2),
    "dp_noise1.0_bs1024_ep5_r10": (1.0, 1024, 5),
}


def read(res: Path, name: str):
    path = res / name
    return pd.read_csv(path) if path.exists() else None


def tail_mean(df: pd.DataFrame, k: int = 3, prefix: str = "") -> pd.DataFrame:
    """Per-client mean accuracy / macro-F1 over the final k rounds."""
    last = df["round"].max()
    out = df[df["round"] > last - k].groupby("client")[["accuracy", "macro_f1"]].mean()
    return out.add_prefix(prefix)


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    def cell(v):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return ""
            return "inf" if np.isinf(v) else f"{v:.{digits}f}"
        return str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |",
             "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def build_all(res: Path):
    sections, skipped = [], []

    def add(stem, title, df, notes=""):
        sections.append((stem, title, df, notes))

    def need(*names):
        frames = [read(res, n) for n in names]
        missing = [n for n, f in zip(names, frames) if f is None]
        skipped.extend(missing)
        return None if missing else frames

    # 0/1. Non-IID heterogeneity ------------------------------------------------
    got = need("global_class_distribution.csv")
    if got:
        d = got[0]
        d.columns = ["class", "proportion"]
        add("00_class_distribution", "Pooled class distribution across all clients", d)

    got = need("non_iid_divergence.csv")
    if got:
        d = got[0][["client", "n_rows", "n_classes_present", "js_divergence"]]
        add("01_noniid_divergence", "How different each client's data is (Jensen-Shannon divergence)", d,
            "0 = same class mix as the pooled data; higher = more different. "
            "n_classes_present is out of 7.")

    # 2. Weighting ablation -----------------------------------------------------
    runs = {"sample-count weights": "baseline_history_samplecount_weighted.csv",
            "log weights": "baseline_history_logweighted.csv",
            "equal weights + FedProx (final)": "baseline_history.csv"}
    rows = []
    for label, fname in runs.items():
        h = read(res, fname)
        if h is None:
            skipped.append(fname)
            continue
        t = tail_mean(h, 3).reset_index()
        t.insert(0, "run", label)
        t.insert(1, "rounds", int(h["round"].max()))
        rows.append(t)
    if rows:
        d = pd.concat(rows, ignore_index=True).sort_values(["client", "run"])
        add("02_weighting_ablation", "Baseline FedAvg: how client weighting changed the result", d,
            "Runs differ in length (see rounds): the sample-count run was only a short first test. "
            "Per-client numbers are the mean of the last 3 rounds (or of all rounds if fewer).")

    # 3. Secure aggregation -----------------------------------------------------
    got = need("baseline_history.csv", "secagg_history.csv")
    if got:
        d = tail_mean(got[0], 3, "baseline_").join(tail_mean(got[1], 3, "secagg_")).reset_index()
        add("03_secagg", "Secure aggregation vs plain FedAvg (same settings)", d,
            "This run used mask scale 0.01, which a later check showed hides an update only weakly "
            "(a masked update was ~7% off the real one). Mask size changes only what a server can see, "
            "not the aggregated model (aggregation error ~1e-5 even at scale 100), so these accuracy "
            "numbers should still hold.")
        b, s = RUNTIMES_S["baseline"], RUNTIMES_S["secagg"]
        add("03b_secagg_overhead", "Secure aggregation runtime overhead",
            pd.DataFrame({"run": ["baseline", "secagg"], "seconds": [b, s],
                          "overhead_pct": [0.0, 100 * (s - b) / b]}),
            "Run times are recorded by hand from the console logs (see RUNTIMES_S in this script).")

    # 4. Krum -------------------------------------------------------------------
    got = need("krum_comparison_fedavg_attacked.csv", "krum_comparison_krum_defended.csv")
    if got:
        d = tail_mean(got[0], 1, "fedavg_attacked_").join(
            tail_mean(got[1], 1, "krum_defended_")).reset_index()
        add("04_krum_comparison", f"One malicious client ('{ATTACKER}'): plain FedAvg vs Krum", d,
            "Final round only. Both runs exclude secure aggregation, because Krum needs to see "
            "individual updates and SecAgg hides them.")
    got = need("krum_selection_scores.csv")
    if got:
        scores = got[0]
        picked = scores["selected"].value_counts()
        everyone = sorted(set(picked.index) | {c[len("score_"):] for c in scores.columns
                                                if c.startswith("score_")})
        d = pd.DataFrame({"client": everyone,
                          "rounds_selected": [int(picked.get(c, 0)) for c in everyone],
                          "rounds_total": len(scores)})
        d["is_attacker"] = d["client"] == ATTACKER
        d = d.sort_values("rounds_selected", ascending=False).reset_index(drop=True)
        add("04b_krum_selection", "Which client's update Krum picked each round", d,
            "Krum keeps ONE client's update per round. If the same client keeps winning, the global "
            "model drifts toward that client (the lock-in effect).")

    # 5. Federated DP -----------------------------------------------------------
    frames = []
    for prefix, (noise, bs, ep) in DP_RUNS.items():
        eps = read(res, f"{prefix}_epsilon_summary.csv")
        hist = read(res, f"{prefix}_history.csv")
        if eps is None:
            skipped.append(f"{prefix}_epsilon_summary.csv")
            continue
        eps = eps.copy()
        eps.insert(0, "noise_multiplier", noise)
        eps.insert(1, "batch_size", bs)
        eps.insert(2, "local_epochs", ep)
        eps.insert(3, "rounds", int(hist["round"].max()) if hist is not None else np.nan)
        frames.append(eps)
    if frames:
        add("05_dp_federated", "Differential privacy across all five clients (final round)",
            pd.concat(frames, ignore_index=True),
            "These federated DP runs collapsed toward predicting the majority class. The likely cause, "
            "measured on Hospital, is that clipping at 1.0 cancels the class weights (minority "
            "gradients were ~7x larger before clipping). Treat the accuracy columns as evidence of "
            "the collapse, not as a privacy-utility curve. The epsilon values are "
            "correct accounting for each run. Epsilon differs by client because smaller clients "
            "sample a larger fraction of their data per step.")

    # 6. E-commerce DP sweep ----------------------------------------------------
    got = need("ecommerce_dp_sweep.csv")
    if got:
        d = got[0].copy()
        plain = d.loc[d["setting"] == "plain", "balanced_accuracy"]
        if len(plain):
            d["utility_vs_plain"] = d["balanced_accuracy"] / float(plain.iloc[0])
        add("06_dp_ecommerce", "Privacy-utility tradeoff on E-commerce (the client where DP works)", d,
            "Single client, trained alone for 20 epochs. Lower epsilon = stronger privacy. "
            "Epsilon is per row; each record has ~10 copies, so protection for a repeated pattern "
            "is weaker than the nominal number.")

    # 7. Dropout ----------------------------------------------------------------
    got = need("baseline_history.csv", "dropout_history.csv", "dropout_participation.csv")
    if got:
        base, drop, part = got
        d = tail_mean(base, 3, "baseline_").join(tail_mean(drop, 3, "dropout_"))
        counts = pd.Series(",".join(part["participants"].astype(str)).split(",")).value_counts()
        d["rounds_trained"] = counts.reindex(d.index).fillna(0).astype(int)
        d["rounds_total"] = len(part)
        add("07_dropout", "Training with clients dropping out of rounds", d.reset_index(),
            f"Average participants per round: {part['n_participants'].mean():.2f} of 5. "
            "This is the SECOND of two dropout runs: the first run's per-client numbers were "
            "overwritten by the re-run.")

    # 8. Membership inference ---------------------------------------------------
    def mia_flags(m):
        m = m.copy()
        m["ci_includes_chance"] = (m["auc_lo"] <= 0.5) & (m["auc_hi"] >= 0.5)
        m["acc_gap"] = m["acc_members"] - m["acc_nonmembers"]
        return m

    got = need("ecommerce_mia_results.csv")
    if got:
        cols = ["model", "kind", "noise_multiplier", "epsilon", "auc", "auc_lo", "auc_hi",
                "advantage", "acc_members", "acc_nonmembers"]
        d = mia_flags(got[0])
        add("08_mia", "Membership inference attack on E-commerce models",
            d[cols + ["acc_gap", "ci_includes_chance"]],
            "AUC 0.5 = attacker no better than guessing. 'saved' = the small models from the DP sweep "
            "(their AUC below 0.5 is probably an artifact of how non-members were chosen; not confirmed); "
            "'leaky' = bigger "
            "models built to overfit. The leaky run was interrupted, so it has no noise 2.0/4.0 rows.")
    got = need("ecommerce_mia_stress.csv")
    if got:
        cols = ["vectors", "hidden", "epochs", "noise_multiplier", "epsilon", "auc", "auc_lo",
                "auc_hi", "advantage", "acc_members", "acc_nonmembers"]
        d = mia_flags(got[0])
        add("09_mia_stress", "Membership inference under stress (few training vectors, big models)",
            d[cols + ["acc_gap", "ci_includes_chance"]],
            "vectors = number of distinct training records (0 means all of them). No setting leaked, "
            "so the DP stage never ran.")

    return sections, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    res = Path(args.results_dir)
    out = Path(args.out_dir) if args.out_dir else res / "summary"
    out.mkdir(parents=True, exist_ok=True)

    sections, skipped = build_all(res)

    report = ["# FedShield-IDS: results summary", ""]
    for stem, title, df, notes in sections:
        df.to_csv(out / f"{stem}.csv", index=False)
        report += [f"## {title}", ""]
        if notes:
            report += [f"_{notes}_", ""]
        report += [md_table(df), ""]
    (out / "summary_report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"Wrote {len(sections)} tables + summary_report.md to {out}/")
    for stem, title, df, _ in sections:
        print(f"  {stem}.csv  ({len(df)} rows)  {title}")
    if skipped:
        print("\nSkipped (input file not found):")
        for name in dict.fromkeys(skipped):
            print(f"  {name}")


if __name__ == "__main__":
    main()