"""
Unified schema for FedShield-IDS (Option A: universal core features).

This file is pure configuration: it loads no data. Every later step
(loaders, partitioning, model) imports its constants from here, so the
definition of a "feature" or a "class" lives in exactly one place.

Dataset keys used everywhere:
    cic    CIC-IDS2017        (Bank + University clients)
    wustl  WUSTL-EHMS-2020    (Hospital client)
    edge   Edge-IIoTset       (Manufacturing/IoT client)
    csic   CSIC-2010          (E-commerce client)
"""

# ---------------------------------------------------------------------------
# 1. Where the raw files live (relative to data/raw/)
# ---------------------------------------------------------------------------
RAW_GLOBS = {
    "cic": "bank_university/MachineLearningCVE/*.csv",
    "wustl": "hospital/wustl-ehms-2020_with_attacks_categories.csv",
    "edge": "manufacturing/Edge-IIoTset_dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv",
    "csic": "e-commerce/Web-Application-Attack-Datasets-master/CSVData/csic_final.csv",
}

# ---------------------------------------------------------------------------
# 2. Unified label taxonomy
# ---------------------------------------------------------------------------
CLASSES = [
    "benign",
    "dos_ddos",           # all DoS / DDoS variants
    "recon",              # port scans, vulnerability scans, fingerprinting
    "web_attack",         # SQL injection, XSS, web brute force, malicious uploads
    "credential_attack",  # FTP / SSH / password guessing
    "malware",            # botnet, backdoor, ransomware, infiltration
    "spoofing_mitm",      # spoofing, data alteration, man-in-the-middle
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

_LABEL_MAPS = {
    "wustl": {
        "normal": "benign",
        "spoofing": "spoofing_mitm",
        "data alteration": "spoofing_mitm",
    },
    "edge": {
        "normal": "benign",
        "ddos_udp": "dos_ddos", "ddos_icmp": "dos_ddos",
        "ddos_tcp": "dos_ddos", "ddos_http": "dos_ddos",
        "port_scanning": "recon", "vulnerability_scanner": "recon",
        "fingerprinting": "recon",
        "sql_injection": "web_attack", "xss": "web_attack", "uploading": "web_attack",
        "password": "credential_attack",
        "backdoor": "malware", "ransomware": "malware",
        "mitm": "spoofing_mitm",
    },
    # CSIC only says Valid / Anomalous. Its anomalous requests are all web
    # application attacks, so they map to web_attack (no finer detail exists).
    "csic": {"valid": "benign", "anomalous": "web_attack"},
}


def map_label(dataset: str, raw) -> str | None:
    """
    Map a dataset-specific label to a unified class name.

    Returns None for labels we deliberately drop. Raises ValueError for
    anything unrecognised, so no row is ever silently lost or mislabelled.
    """
    s = str(raw).strip().lower()

    if dataset == "cic":
        # Keyword matching on purpose: CIC's web-attack labels contain a
        # corrupted dash character, so exact string equality would fail.
        if s == "benign":
            return "benign"
        if s.startswith("dos") or s == "ddos":
            return "dos_ddos"
        if s == "portscan":
            return "recon"
        if s in ("ftp-patator", "ssh-patator"):
            return "credential_attack"
        if s.startswith("web attack"):
            return "web_attack"
        if s in ("bot", "infiltration"):
            return "malware"
        if s == "heartbleed":
            return None  # 11 rows in the whole dataset: too few to learn or test
    elif dataset in _LABEL_MAPS and s in _LABEL_MAPS[dataset]:
        return _LABEL_MAPS[dataset][s]

    raise ValueError(f"Unknown {dataset} label: {raw!r}")


# ---------------------------------------------------------------------------
# 3. Feature groups
# ---------------------------------------------------------------------------
# A group is either fully available for a client or fully missing. Missing
# groups are filled with 0 and announced to the model by a "<group>_missing"
# flag, so the model can tell "zero" apart from "not measured".
FEATURE_GROUPS = {
    "size":      ["total_packets", "total_bytes", "avg_bytes_per_packet"],
    "port":      ["dst_port"],
    "timing":    ["duration_s"],
    "tcp_flags": ["syn_count", "rst_count", "fin_count", "ack_count"],
    "http":      ["uri_length", "payload_length", "special_char_count",
                  "sql_keyword_hits", "script_pattern_hits"],
}
ALWAYS_PRESENT = {"size"}  # every client can provide these

GROUPS_AVAILABLE = {
    "cic":   {"size", "port", "timing", "tcp_flags"},
    "wustl": {"size", "port", "timing"},
    "edge":  {"size", "port", "tcp_flags", "http"},
    "csic":  {"size", "http"},
}

FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]
FLAG_COLUMNS = [f"{g}_missing" for g in FEATURE_GROUPS if g not in ALWAYS_PRESENT]
MODEL_INPUT_COLUMNS = FEATURES + FLAG_COLUMNS
INPUT_DIM = len(MODEL_INPUT_COLUMNS)

# Fixed transforms (no statistics fitted from data, so nothing is pooled
# across clients). Heavy-tailed counts get log1p; ports are scaled to 0-1.
TRANSFORM = {f: "log1p" for f in FEATURES}
TRANSFORM["dst_port"] = "scale_65535"

# Edge and CSIC rows are single packets / single HTTP requests, not flows.
ROW_IS_ONE_PACKET = {"edge", "csic"}

# CIC reports flow duration in microseconds; WUSTL (Argus) already uses seconds.
UNIT_SCALE = {"cic": {"duration_s": 1e-6}}

# ---------------------------------------------------------------------------
# 4. Text-feature definitions (shared, so CSIC and Edge are measured identically)
# ---------------------------------------------------------------------------
SPECIAL_CHARS = "'\"<>;%&|`$(){}[]\\"
SQL_KEYWORDS = ("select", "union", "insert", "drop table", "or 1=1",
                "' or ", "--", "sleep(", "waitfor delay")
SCRIPT_PATTERNS = ("<script", "javascript:", "onerror=", "onload=",
                   "alert(", "<img", "<iframe")

# ---------------------------------------------------------------------------
# 5. Source columns per dataset (names as they appear AFTER whitespace strip)
#    Identifiers (IPs, MACs, timestamps, raw payload, patient vitals) are
#    never listed here, so loaders never even read them.
# ---------------------------------------------------------------------------
SOURCES = {
    "cic": {
        "label": "Label",
        "dst_port": ["Destination Port"],
        "total_packets": ["Total Fwd Packets", "Total Backward Packets"],       # summed
        "total_bytes": ["Total Length of Fwd Packets", "Total Length of Bwd Packets"],  # summed
        "duration_s": ["Flow Duration"],
        "syn_count": ["SYN Flag Count"],
        "rst_count": ["RST Flag Count"],
        "fin_count": ["FIN Flag Count"],
        "ack_count": ["ACK Flag Count"],
    },
    "wustl": {
        "label": "Attack Category",   # NOT "Label", which is only 0/1
        "dst_port": ["Dport"],
        "total_packets": ["TotPkts"],
        "total_bytes": ["TotBytes"],
        "duration_s": ["Dur"],
    },
    "edge": {
        "label": "Attack_type",
        "dst_port": ["tcp.dstport", "udp.port"],   # first non-zero wins
        "total_bytes": ["tcp.len"],                # TCP payload only; no frame length exists
        "syn_count": ["tcp.connection.syn"],
        "rst_count": ["tcp.connection.rst"],
        "fin_count": ["tcp.connection.fin"],
        "ack_count": ["tcp.flags.ack"],
        "http_text": ["http.request.full_uri", "http.request.uri.query", "http.file_data"],
    },
    "csic": {
        "label": "Class",
        "http_text": ["URI", "GET-Query", "POST-Data"],
        # request size is approximated as the sum of these text lengths
        "size_text": ["URI", "GET-Query", "POST-Data", "Cookie", "User-Agent"],
    },
}


def columns_needed(dataset: str) -> set[str]:
    """Every raw column a loader must read for this dataset (use with usecols)."""
    cols: set[str] = set()
    for key, value in SOURCES[dataset].items():
        if key == "label":
            cols.add(value)
        else:
            cols.update(value)
    return cols
