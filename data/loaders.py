"""
Step 3, part 2: per-dataset loaders.

Each load_*() function reads a client's raw file(s) and returns a DataFrame
with exactly schema.MODEL_INPUT_COLUMNS + label_name + label_id + client.
Nothing downstream needs to know these datasets were ever different shapes.

Run this file directly for a quick sanity check of every loader:
    python data/loaders.py
"""

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import schema
except ImportError:
    import schema


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _apply_labels(dataset: str, raw_labels: pd.Series) -> pd.Series:
    """Map raw labels to the unified taxonomy; unknowns are logged and dropped."""
    mapped, unknown_counts = [], {}
    for raw in raw_labels:
        try:
            mapped.append(schema.map_label(dataset, raw))
        except ValueError:
            key = str(raw)
            unknown_counts[key] = unknown_counts.get(key, 0) + 1
            mapped.append(None)
    if unknown_counts:
        total = sum(unknown_counts.values())
        print(f"[{dataset}] dropping {total} rows with unrecognized/excluded labels:")
        for lbl, cnt in sorted(unknown_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"    {lbl!r}: {cnt}")
    return pd.Series(mapped, index=raw_labels.index)


_SPECIAL_RE = "[" + re.escape(schema.SPECIAL_CHARS) + "]"


def _text_features(text: pd.Series) -> dict[str, pd.Series]:
    """special_char_count / sql_keyword_hits / script_pattern_hits from raw text."""
    text = text.fillna("").astype(str)
    lower = text.str.lower()
    special = text.str.count(_SPECIAL_RE)
    sql_hits = pd.Series(0, index=text.index)
    for kw in schema.SQL_KEYWORDS:
        sql_hits = sql_hits + lower.str.count(re.escape(kw))
    script_hits = pd.Series(0, index=text.index)
    for kw in schema.SCRIPT_PATTERNS:
        script_hits = script_hits + lower.str.count(re.escape(kw))
    return {"special_char_count": special, "sql_keyword_hits": sql_hits,
            "script_pattern_hits": script_hits}


def _finalize(dataset: str, raw_features: pd.DataFrame, raw_labels: pd.Series) -> pd.DataFrame:
    """
    raw_features: only the columns this dataset can supply (untransformed).
    raw_labels:   the dataset's raw label column, unmapped.
    Returns a DataFrame in the exact shared schema, transforms applied,
    missing groups zero-filled with their flag set, bad-label rows dropped.
    """
    label_name = _apply_labels(dataset, raw_labels)
    keep = label_name.notna()
    raw_features = raw_features.loc[keep].reset_index(drop=True)
    label_name = label_name.loc[keep].reset_index(drop=True)

    available = schema.GROUPS_AVAILABLE[dataset]
    display_name = schema.CLIENT_DISPLAY_NAME.get(dataset, dataset)
    out = pd.DataFrame(index=raw_features.index)
    for group, cols in schema.FEATURE_GROUPS.items():
        if group in available:
            for c in cols:
                out[c] = raw_features[c].astype(float)
            if group not in schema.ALWAYS_PRESENT:
                out[f"{group}_missing"] = 0
        else:
            for c in cols:
                out[c] = 0.0
            if group not in schema.ALWAYS_PRESENT:
                out[f"{group}_missing"] = 1

    for feat, kind in schema.TRANSFORM.items():
        if kind == "log1p":
            out[feat] = np.log1p(out[feat].clip(lower=0))
        elif kind == "scale_65535":
            out[feat] = out[feat].clip(lower=0, upper=65535) / 65535.0

    out = out[schema.MODEL_INPUT_COLUMNS].copy()
    out["label_name"] = label_name.values
    out["label_id"] = out["label_name"].map(schema.CLASS_TO_ID)
    out["client"] = display_name
    return out


def _raw_path(raw_dir: str, dataset: str) -> str:
    return str(Path(raw_dir) / schema.RAW_GLOBS[dataset])


# ---------------------------------------------------------------------------
# CIC-IDS2017 -- Bank + University (8 files, one per day/scenario)
# ---------------------------------------------------------------------------

def load_cic(raw_dir: str = "data/raw") -> pd.DataFrame:
    files = sorted(glob.glob(_raw_path(raw_dir, "cic")))
    if not files:
        raise FileNotFoundError(f"No CIC-IDS2017 files found under {raw_dir}/bank_university/")

    src = schema.SOURCES["cic"]
    frames = []
    for fp in files:
        df = pd.read_csv(fp, low_memory=False, on_bad_lines="warn")
        df.columns = df.columns.str.strip()

        # Deduplicate on the RAW data, excluding true identifiers (flow ID,
        # IPs, timestamp). Doing this on the full ~78 raw columns -- not
        # our reduced harmonized feature set -- avoids falsely merging rows
        # that only look alike after we've thrown away most of their detail.
        id_cols = [c for c in ("Flow ID", "Source IP", "Source Port",
                                "Destination IP", "Timestamp") if c in df.columns]
        dedup_cols = [c for c in df.columns if c not in id_cols]
        before = len(df)
        df = df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
        if before != len(df):
            print(f"[cic] {Path(fp).name}: {before:,} -> {len(df):,} rows "
                  f"({before - len(df):,} raw duplicates removed)")

        feats = pd.DataFrame(index=df.index)
        feats["total_packets"] = df[src["total_packets"]].sum(axis=1)
        feats["total_bytes"] = df[src["total_bytes"]].sum(axis=1)
        feats["avg_bytes_per_packet"] = feats["total_bytes"] / feats["total_packets"].replace(0, np.nan)
        feats["avg_bytes_per_packet"] = feats["avg_bytes_per_packet"].fillna(0)
        feats["dst_port"] = df[src["dst_port"][0]]
        feats["duration_s"] = df[src["duration_s"][0]] * schema.UNIT_SCALE["cic"]["duration_s"]
        feats["syn_count"] = df[src["syn_count"][0]]
        feats["rst_count"] = df[src["rst_count"][0]]
        feats["fin_count"] = df[src["fin_count"][0]]
        feats["ack_count"] = df[src["ack_count"][0]]

        out = _finalize("cic", feats, df[src["label"]])
        out["client"] = schema.CIC_FILE_TO_CLIENT[Path(fp).name]
        frames.append(out)
        print(f"[cic -> {out['client'].iloc[0]}] {Path(fp).name}: {len(df):,} rows loaded")

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# WUSTL-EHMS-2020 -- Hospital
# ---------------------------------------------------------------------------

def load_wustl(raw_dir: str = "data/raw") -> pd.DataFrame:
    fp = _raw_path(raw_dir, "wustl")
    src = schema.SOURCES["wustl"]
    df = pd.read_csv(fp, low_memory=False, on_bad_lines="warn")
    df.columns = df.columns.str.strip()

    id_cols = [c for c in ("SrcAddr", "DstAddr", "SrcMac", "DstMac",
                            "Packet_num", "Sport") if c in df.columns]
    dedup_cols = [c for c in df.columns if c not in id_cols]
    before = len(df)
    df = df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
    if before != len(df):
        print(f"[wustl] {before:,} -> {len(df):,} rows "
              f"({before - len(df):,} raw duplicates removed)")

    feats = pd.DataFrame(index=df.index)
    feats["total_packets"] = df[src["total_packets"][0]]
    feats["total_bytes"] = df[src["total_bytes"][0]]
    feats["avg_bytes_per_packet"] = feats["total_bytes"] / feats["total_packets"].replace(0, np.nan)
    feats["avg_bytes_per_packet"] = feats["avg_bytes_per_packet"].fillna(0)
    feats["dst_port"] = df[src["dst_port"][0]]
    feats["duration_s"] = df[src["duration_s"][0]]  # already seconds, no scaling

    return _finalize("wustl", feats, df[src["label"]])


# ---------------------------------------------------------------------------
# Edge-IIoTset -- Manufacturing/IoT (2.2M rows: use usecols to keep RAM low)
# ---------------------------------------------------------------------------

def load_edge(raw_dir: str = "data/raw") -> pd.DataFrame:
    fp = _raw_path(raw_dir, "edge")
    src = schema.SOURCES["edge"]
    needed = schema.columns_needed("edge")

    df = pd.read_csv(fp, usecols=list(needed), low_memory=False, on_bad_lines="warn")

    feats = pd.DataFrame(index=df.index)
    feats["total_packets"] = 1.0  # each row is one packet
    feats["total_bytes"] = df[src["total_bytes"][0]].fillna(0)
    feats["avg_bytes_per_packet"] = feats["total_bytes"]  # total_packets == 1
    tcp_port = df["tcp.dstport"].fillna(0)
    udp_port = df["udp.port"].fillna(0)
    feats["dst_port"] = np.maximum(tcp_port, udp_port)
    feats["syn_count"] = df[src["syn_count"][0]].fillna(0)
    feats["rst_count"] = df[src["rst_count"][0]].fillna(0)
    feats["fin_count"] = df[src["fin_count"][0]].fillna(0)
    feats["ack_count"] = df[src["ack_count"][0]].fillna(0)

    http_cols = src["http_text"]
    http_text = df[http_cols].fillna("").astype(str).agg(" ".join, axis=1)
    feats["uri_length"] = df[http_cols[0]].fillna("").astype(str).str.len()
    feats["payload_length"] = df[http_cols[2]].fillna("").astype(str).str.len()
    for name, series in _text_features(http_text).items():
        feats[name] = series

    return _finalize("edge", feats, df[src["label"]])


# ---------------------------------------------------------------------------
# CSIC-2010 -- E-commerce (raw HTTP requests, needs real feature engineering)
# ---------------------------------------------------------------------------

def load_csic(raw_dir: str = "data/raw") -> pd.DataFrame:
    fp = _raw_path(raw_dir, "csic")
    src = schema.SOURCES["csic"]
    needed = list({src["label"], *src["http_text"], *src["size_text"]})

    df = pd.read_csv(fp, usecols=needed, low_memory=False, on_bad_lines="warn")

    text_cols = [c for c in src["size_text"] if c in df.columns]
    lengths = df[text_cols].fillna("").astype(str).apply(lambda s: s.str.len())
    feats = pd.DataFrame(index=df.index)
    feats["total_bytes"] = lengths.sum(axis=1)
    feats["total_packets"] = 1.0
    feats["avg_bytes_per_packet"] = feats["total_bytes"]

    http_text = df[src["http_text"]].fillna("").astype(str).agg(" ".join, axis=1)
    feats["uri_length"] = df["URI"].fillna("").astype(str).str.len()
    feats["payload_length"] = df["POST-Data"].fillna("").astype(str).str.len()
    for name, series in _text_features(http_text).items():
        feats[name] = series

    return _finalize("csic", feats, df[src["label"]])


LOADERS = {"cic": load_cic, "wustl": load_wustl, "edge": load_edge, "csic": load_csic}


if __name__ == "__main__":
    for name, fn in LOADERS.items():
        print(f"\n{'=' * 60}\nLoading {name}\n{'=' * 60}")
        out = fn()
        print(f"\n{name}: {out.shape[0]:,} rows x {out.shape[1]} columns")
        print(out["label_name"].value_counts())