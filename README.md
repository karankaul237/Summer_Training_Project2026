# FedShield-IDS

Privacy-preserving federated intrusion detection across 5 real-world client
organizations, each holding a genuinely different, sector-native dataset.

## Why this exists

Network intrusion data cannot be centralized across organizations (hospitals,
banks, universities, e-commerce platforms, industrial/IoT operators) without
violating privacy law and competitive boundaries. FedShield-IDS trains one
shared intrusion/malware/phishing detection model across these clients using
federated learning, without any client ever sharing its raw traffic data.

The project also proves its privacy claims empirically, rather than just
stating them: it runs a Membership Inference Attack (MIA) against the trained
model under three settings (no privacy, secure aggregation only, secure
aggregation + differential privacy) and reports how attack success drops as
privacy protections are added.

## Clients and datasets

| Client              | Dataset                          |
|---------------------|-----------------------------------|
| Hospital            | WUSTL-EHMS-2020                   |
| Bank / Finance       | CIC-IDS2017 (brute force + web attack subset) |
| University           | CIC-IDS2017 (general subset)      |
| E-commerce           | CSIC-2010                         |
| Manufacturing / IoT  | Edge-IIoTset / TON_IoT            |

All five are harmonized into one shared flow-based feature schema so a single
global model architecture can be trained across all of them.

## Repo layout

```
fedshield-ids/
├── data/
│   ├── raw/            # downloaded, untouched datasets
│   ├── harmonized/      # after feature-schema mapping
│   └── partitioned/     # per-client non-IID splits
├── partition/           # non-IID partitioning logic
├── client/              # Flower client: local training, DP-SGD
├── server/               # Flower server: FedAvg / SecAgg / robust aggregation
├── privacy/              # DP accountant, SecAgg config, MIA attack script
├── models/                # model architecture definitions
├── experiments/           # run configs (baseline, secagg, secagg+dp)
├── results/               # logged metrics, plots
└── requirements.txt
```

## Build status

- [x] Step 1 — environment & repo scaffold
- [ ] Step 2 — download & inspect datasets
- [ ] Step 3 — feature harmonization module
- [ ] Step 4 — non-IID partitioning
- [ ] Step 5 — baseline FedAvg
- [ ] Step 6 — secure aggregation
- [ ] Step 7 — robust aggregation (Krum)
- [ ] Step 8 — DP-SGD + epsilon tracking
- [ ] Step 9 — client dropout simulation
- [ ] Step 10 — membership inference attack
- [ ] Step 11 — metrics/results logging
- [ ] Step 12 — Streamlit dashboard

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
