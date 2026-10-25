# FedShield-IDS

Federated intrusion detection across five simulated organizations (a hospital, a bank, a
university, an e-commerce site and an industrial-IoT plant). One shared model is trained
with federated learning, so raw traffic data never leaves a client. The project then adds
several protections and measures each one: secure aggregation, robust aggregation (Krum),
differential privacy, client dropout, and a membership-inference attack. It reports where
they worked and where they did not.

> **Research prototype.** Everything here is a simulation on public datasets. It is not a
> deployment, not evidence of regulatory compliance, and the secure-aggregation component
> is a simulation of the protocol, not a secure implementation. See [Limitations](#limitations).

## Previously reported findings (pre-correction; rerun required)

> **Important evaluation warning:** The metrics and conclusions in this README were produced
> using the previous row-level split. That split allowed identical model-input feature vectors
> to occur in both training and held-out data for some clients. The group-aware split now prevents
> exact model-input vectors from crossing partitions, but the experiments have **not yet been
> rerun on these corrected partitions**. Treat the results below as historical, not as validated
> performance for the corrected split. Regenerate the partitions, rerun the experiments, and
> regenerate `results/summary/` before reporting current model performance.


1. **Federated training works for most clients, not all.** The final baseline (15 rounds,
   equal client weights, FedProx, extra local epochs for the two smallest clients) gives
   usable detectors for Bank, Manufacturing and University. Hospital's only attack class
   (spoofing/MITM) is detected poorly at the default decision cutoff: about 2% recall in the
   final diagnosis, with accuracy close to its benign share. Its ROC AUC is about 0.77 in every
   federated model scored, so a ranking signal exists even though the default cutoff hides it.
2. **Client weighting decided whether small clients were served.** Weighting by row count made the
   global model predict `dos_ddos` for 100% of Hospital's validation rows and nearly all of
   E-commerce's (0% accuracy for both). Log weights helped E-commerce. Equal weights, FedProx
   and extra local epochs was the best of the three but did not fix Hospital.
3. **Secure aggregation cost 0.8% extra runtime** (2,434 s vs 2,415 s over 15 rounds) and the
   masks cancel to floating-point precision. Caveats: it is a simulation (a shared constant
   stands in for key exchange), and the 15-round run used masks too small to hide an update
   (a masked update was only about 7% away from the real one). The code now uses masks 10,000x
   larger (see [Secure aggregation](#secure-aggregation)), but the run was not repeated.
4. **Secure aggregation needs a fixed set of participants.** If a client disappears after
   masking, the sum is corrupted (33,333% error with strong masks). Masking only among the known
   survivors restores exact cancellation. Recovering from a client lost mid-round needs secret
   sharing, which is not implemented.
5. **Krum stopped one sign-flipping attacker, at a price.** With `university` malicious over
   10 rounds, plain FedAvg fell to about 0% accuracy on four of five clients (Hospital 12.5%).
   Krum never selected the attacker (0 of 10 rounds). But Krum keeps one client's update per
   round, and it locked onto Hospital (rounds 1-4) and then E-commerce (rounds 5-10): E-commerce
   reached 73.1% accuracy at round 10 and every other client stayed at or below 4.7%.
6. **DP-SGD with a class-weighted loss collapses on imbalanced clients and works where classes
   are balanced.** On Hospital with no noise, clipping at 1.0 gives 0.000 spoofing recall against
   0.503 for plain training; a clip of 100 restores 0.503. Measured per-sample gradient norms
   explain it: spoofing samples had a median norm of 19.1 against 2.8 for benign at the start of
   training, so clipping to 1.0 flattens the class weighting. On E-commerce, DP works: balanced
   accuracy is 0.735 plain and 0.674-0.681 under DP for epsilon from 16.5 down to 0.33.
7. **The membership-inference attack found no leakage.** A loss-threshold attack scored AUC
   0.467-0.511 on 16 E-commerce models, including ones built to overfit. So DP's effect on the
   attack could not be shown. A likely contributor is that 22.7% of distinct E-commerce feature
   vectors appear with both labels, which limits how much any model can memorize.
8. **Training survived client dropout.** With about 30% of client-rounds dropped (3.33 of 5
   clients per round on average), two runs still converged, with larger round-to-round swings and
   large differences between the two runs.

## What this project is, and is not

- **Is:** an end-to-end, reproducible pipeline for a five-client federation, plus an honest
  evaluation of each protection, including the negative results.
- **Is not:** a real deployment. The five "organizations" are five different public datasets reduced
  to a shared set of coarse features. No real hospital or bank data is involved.
- **Does not detect phishing.** None of the four datasets has a phishing label. The classes are
  listed below.

### Detection classes (7)

|        Class      |                              Meaning                                   |
|-------------------|------------------------------------------------------------------------|
|        benign     |                           normal traffic                               |
|       dos_ddos    |                     all DoS and DDoS variants                          |
|        recon      |            port scans, vulnerability scans, fingerprinting             |
|     web_attack    | SQL injection, XSS, web brute force, malicious uploads, anomalous HTTP |
| credential_attack |                      FTP/SSH password guessing                         |
|       malware     |                 botnet, backdoor, ransomware, infiltration             |
|    spoofing_mitm  |                spoofing, data alteration, man-in-the-middle            |

Each client has only some classes, which is real non-IID heterogeneity:

|    Client     |          Classes present              |
|---------------|---------------------------------------|
|     Bank      | benign, credential_attack, web_attack |
|   E-commerce  |         benign, web_attack            |
|    Hospital   |       benign, spoofing_mitm           |
| Manufacturing |               all 7                   |
|   University  |   benign, dos_ddos, recon, malware    |

## Repository layout

```
fedshield-ids/
├── README.md
├── requirements.txt
├── data/
│   ├── schema.py            # feature schema, label taxonomy, per-dataset mappings
│   ├── loaders.py           # one loader per dataset -> shared schema
│   ├── harmonize.py         # runs all loaders, writes data/harmonized/combined.csv
│   ├── inspect_dataset.py   # quick look at any raw CSV
│   ├── raw/                 # YOU put the downloaded datasets here
│   ├── harmonized/
│   └── partitioned/         # per-client train/val/test
├── partition/
│   ├── split.py             # duplicate-group-aware 70/15/15 split per client
│   ├── validate_splits.py   # checks row/label preservation and cross-split feature overlap
│   └── non_iid_metrics.py   # Jensen-Shannon divergence per client
├── models/mlp.py            # 18 -> 64 -> 32 -> 7 MLP (3,527 parameters)
├── client/                  # Flower client, data loading
├── server/                  # FedAvg strategy, Krum, dropout
├── privacy/                 # secure aggregation, epsilon accounting
├── experiments/             # one script per experiment, plus diagnostics
├── dashboard/app.py         # Streamlit dashboard
└── results/                 # raw outputs; results/summary/ holds the tables
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows. macOS/Linux: source venv/bin/activate
python -m pip install -r requirements.txt
```

- Developed on Python 3.11.0, CPU only. Run every command from the project root (the scripts use
  relative paths).
- Use `python -m pip`, not `pip`. A moved or copied virtual environment can leave `pip.exe`
  pointing at its old location.
- `requirements.txt` needs `flwr[simulation]` (it pulls in Ray, which Flower's simulation needs).
- Versions of Flower, PyTorch and Opacus: TODO (paste the output of `python -m pip freeze`, or save
  it as `requirements-lock.txt`).
- The code uses Flower's older `start_simulation` API, which Flower marks deprecated. The warnings
  it prints are harmless, but a future Flower release may remove it.

## Data

The datasets are not included. Download them into `data/raw/` with this layout:

|      Client       |               Dataset                 |                        Expected path under `data/raw/`                                       |
|-------------------|---------------------------------------|----------------------------------------------------------------------------------------------|
| Bank + University |CIC-IDS2017 (8 daily CSVs,`MachineLearningCSV`)|             `bank_university/MachineLearningCVE/*.csv`                               |
|     Hospital      |          WUSTL-EHMS-2020              |             `hospital/wustl-ehms-2020_with_attacks_categories.csv`                           |
|   E-commerce      |        CSIC-2010 (CSV version)        |        `e-commerce/Web-Application-Attack-Datasets-master/CSVData/csic_final.csv`            |
| Manufacturing/IoT | Edge-IIoTset (ML/DL selected dataset) | `manufacturing/Edge-IIoTset_dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv` |

Sources: CIC-IDS2017 from the Canadian Institute for Cybersecurity (University of New Brunswick);
WUSTL-EHMS-2020 from Washington University in St. Louis (also mirrored on Kaggle); CSIC-2010 as
CSVs from the GitHub repository `msudol/Web-Application-Attack-Datasets`; Edge-IIoTset from IEEE
DataPort or Kaggle. Check each dataset's terms and citation requirements before reuse.

How clients are formed: the CIC-IDS2017 daily files are split by day. Bank gets the Tuesday and
Thursday-morning (web attacks) files; University gets the other six.

### Harmonization

The four datasets have different shapes (flow records, single packets, raw HTTP text). They are
mapped to a shared schema of **14 features plus 4 "missing" flags = 18 model inputs**:

|   Group   |                              Features                                                 |   Available for   |
|-----------|---------------------------------------------------------------------------------------|-------------------|
|   size    |               total_packets, total_bytes, avg_bytes_per_packet                        |        all        |
|   port    |                                dst_port                                               |  CIC, WUSTL, Edge |
|  timing   |                               duration_s                                              |     CIC, WUSTL    |
| tcp_flags |                        syn, rst, fin, ack counts                                      |      CIC, Edge    |
|   http    | uri_length, payload_length, special_char_count, sql_keyword_hits, script_pattern_hits |      Edge, CSIC   |

Groups a client cannot supply are zero-filled and flagged, so the model can tell "zero" from
"not measured". Counts use `log1p` and ports are scaled to 0-1; both are fixed transforms with
nothing fitted from data. IPs, MAC addresses, timestamps, raw payload text and the hospital's
patient vital signs are never carried into the harmonized data.

Duplicates: CIC-IDS2017 was deduplicated on its raw columns (about 9% of rows removed) and
WUSTL had none. Edge-IIoTset and CSIC were **not** deduplicated, because their loaders keep only
a few columns and deduplicating the reduced features wrongly removed over 90% of both.

Final harmonized size: 4,870,837 rows (Manufacturing 2,219,201; University 1,988,109;
Bank 586,144; E-commerce 61,065; Hospital 16,318).

## Running the pipeline

```bash
python data/harmonize.py                         # raw -> data/harmonized/combined.csv
python partition/split.py                        # duplicate-group-aware per-client split
python partition/validate_splits.py               # verify row/label preservation and no feature leakage
python partition/non_iid_metrics.py              # heterogeneity table

python experiments/run_baseline.py --rounds 15   # plain FedAvg (final baseline)
python experiments/run_secagg.py --rounds 15     # + secure aggregation
python experiments/run_krum_attack.py --rounds 10   # FedAvg vs Krum, one attacker
python experiments/run_dropout.py --rounds 15 --dropout 0.3
python experiments/secagg_dropout_demo.py        # masks vs a vanishing client

python experiments/run_dp.py --noise-multiplier 1.0 --rounds 10   # federated DP (collapses)
python experiments/dp_sweep.py ecommerce         # DP privacy-utility sweep, one client

python experiments/mia.py ecommerce              # membership attack
python experiments/mia_stress.py ecommerce       # attack under stress
python experiments/check_label_conflicts.py ecommerce
python experiments/roc_curves.py                 # detection + membership ROC

python experiments/summarize_results.py          # results/summary/*.csv + report
streamlit run dashboard/app.py
```

Approximate run times on the author's CPU-only machine (from the console logs): baseline 15 rounds
about 40 min; secure aggregation about 41 min; dropout about 30 min; federated DP 5 rounds about
90 min; the 10-round large-batch federated DP run about 4 hours. Other steps take minutes (not
timed). Diagnostic scripts (`diagnose_*.py`) are single-client and take minutes.

## Dashboard

`streamlit run dashboard/app.py` shows every table in `results/summary/` plus learning-curve
comparisons, heatmaps and ROC curves. It computes nothing new; it only reshapes the summary CSVs.
Re-run `summarize_results.py` after any experiment and press "Reload tables" in the sidebar.

## Method

**Model.** An MLP, 18 -> 64 -> 32 -> 7, dropout 0.3, 3,527 parameters. Kept small on purpose
(small clients overfit; DP-SGD cost grows with parameters).

**Training (final baseline).** FedAvg with equal client weights; class-weighted cross-entropy
(inverse frequency, capped at 20), computed per client from its own data; Adam (lr 1e-3, weight
decay 1e-4); FedProx (mu 0.01); local epochs 10 for Hospital, 8 for E-commerce and 1 for the
others. Evaluation uses **macro-F1**, because accuracy rewards predicting the majority class.

### Secure aggregation

Pairwise masking in the style of Bonawitz et al. (2017): each pair of clients derives a shared
mask, one adds it and the other subtracts it, so the masks cancel in the sum and the server
sees only the aggregate. This implementation is a **simulation**:

- a shared constant stands in for the pairwise key exchange;
- there is no secret sharing, so it cannot recover from a client lost mid-round;
- masks are Gaussian, not uniform over a finite field, so hiding is not information-theoretic.

Masks only cancel if every client has the same weight, which is why the baseline uses equal
weights. The mask scale in `privacy/secure_aggregation.py` was raised from 0.01 to 100 after a check
showed 0.01 hides an update only weakly (masked update about 7% from the real one, against about
660 times the real size at scale 100). The Step 6 training run used the old scale.

### Krum

Krum scores each client's update by its distance to the nearest others and keeps the single
lowest-scoring one. With 5 clients it tolerates at most 1 attacker. The attack tested is
sign-flipping with a 20x scale. Krum needs to see individual updates, so it was run without secure
aggregation: the two conflict.

### Differential privacy

DP-SGD with Opacus (per-sample clipping plus Gaussian noise), RDP accounting, delta = 1e-5.
Epsilon is computed after training from the number of steps and the sampling rate.
Opacus runs with its secure random generator off, which is fine for experiments only.

### Membership inference

A loss-threshold attack (Yeom et al. 2018): records with lower loss are guessed to be training
members. Membership is defined per distinct feature vector, because E-commerce has about 10
copies of each record and a copy in both train and held-out data would otherwise be mislabelled.
The reported AUC has a 95% bootstrap interval.

## Results

All values come from `results/summary/`. Training is not seeded per round, so re-running will give
somewhat different numbers.

### Client heterogeneity (`01_noniid_divergence`)

|     Client   | Jensen-Shannon divergence from pooled data | Classes present |
|--------------|--------------------------------------------|-----------------|
|  E-commerce  |                   0.250                    |        2        |
|   Hospital   |                   0.170                    |        2        |
|     Bank     |                   0.104                    |        3        |
|  University  |                   0.024                    |        4        |
|Manufacturing |                   0.005                    |        7        |

### Baseline and dropout, accuracy / macro-F1 (`07_dropout`, mean of the last 3 rounds)

|     Client    |    Baseline   | With ~30% dropout (run 2 of 2) |
|---------------|---------------|--------------------------------|
|      Bank     | 0.728 / 0.423 |        0.722 / 0.454           |
|   E-commerce  | 0.362 / 0.372 |        0.502 / 0.522           |
|    Hospital   | 0.878 / 0.490 |        0.896 / 0.593           |
| Manufacturing | 0.869 / 0.405 |        0.809 / 0.349           |
|   University  | 0.689 / 0.543 |        0.645 / 0.387           |

Single runs: E-commerce in particular swings a lot between runs. The first dropout run's per-client
numbers were overwritten by the second.

### DP on E-commerce (`06_dp_ecommerce`)

|  Setting  | Epsilon  | Balanced accuracy | Benign recall | Web-attack recall |
|-----------|----------|-------------------|---------------|-------------------|
|   plain   | infinity |       0.735       |     0.716     |      0.754        |
| noise 0.5 |   16.54  |       0.674       |     0.716     |      0.633        |
| noise 1.0 |    2.13  |       0.674       |     0.716     |      0.633        |
| noise 2.0 |    0.74  |       0.674       |     0.716     |      0.633        |
| noise 4.0 |    0.33  |       0.681       |     0.746     |      0.615        |

One client trained alone for 20 epochs (batch 256, clip 1.0). The three identical rows look odd;
a likely reason is that E-commerce has few distinct inputs, so predictions only change near the
decision boundary. This is not verified.

### Federated DP: epsilon differs by client at the same noise (noise 1.0, batch 256, 5 rounds)

|     Client    | Epsilon |
|---------------|---------|
| Manufacturing |   0.53  |
|   University  |   0.54  |
|      Bank     |   0.69  |
|   E-commerce  |   1.59  |
|    Hospital   |   3.37  |

Smaller clients sample a larger fraction of their data per step, so they spend more privacy per
step. These federated DP runs collapsed toward the majority class; do not read their accuracy as a
privacy-utility curve.

### Detection ROC, AUC on each client's held-out test split (`11_roc_detection`)

|          Model           | Bank  | E-commerce | Hospital | Manufacturing | University |
|--------------------------|-------|------------|----------|---------------|------------|
|      baseline (final)    | 0.970 |   0.528    |  0.774   |     0.948     |   TODO     |
|    secure aggregation    | 0.907 |   0.732    |  0.775   |     0.934     |   0.707    |
|      client dropout      | 0.847 |   0.640    |  0.775   |     0.976     |   0.857    |
|        log weights       | 0.838 |   0.617    |  0.774   |     0.828     |   0.907    |
| DP noise 1.0 (federated) | 0.896 |   0.586    |  0.774   |     0.825     |   0.594    |

Each cell is one training run. Differences between models for the same client are mostly
run-to-run variation, not an effect of secure aggregation or dropout.

### Membership attack (`08_mia`, `09_mia_stress`)

|                        Models                         | Count |     AUC     |
|-------------------------------------------------------|-------|-------------|
|         small saved models (plain + 4 DP levels)      |   5   | 0.467-0.472 |
|       bigger "leaky" models (plain, DP 0.5, DP 1.0)   |   3   | 0.503-0.504 |
| stress settings (non-DP, 300 or all training vectors) |   8   | 0.499-0.511 |

0.5 means the attacker is guessing. The small saved models score slightly below 0.5, which is
probably an artifact of how non-members were chosen (not confirmed).

## Privacy and compliance notes

This is a sketch of what each technique contributes, not legal advice and not a compliance claim.

|      Technique       |                          What it contributes here                                     | What it does not do |
|----------------------|---------------------------------------------------------------------------------------|---------------------|
|  Federated learning  |           Raw records stay with each client (data minimization in spirit).            | Model updates and the final model can still leak information, so it is not anonymization. |
|  Identifier removal  | IPs, MACs, timestamps, payload text and patient vitals are dropped from the features. | The labels and the coarse features remain. |
|  Secure aggregation  |                Hides individual updates from the server, in a simulation.             | The simulation has no real key exchange, so it provides no real security. It does not protect the final model. |
| Differential privacy |                   Bounds one record's influence; epsilon is computed.                 | Epsilon here is per row. Records appear about 10 times in E-commerce, so protection for a repeated pattern is weaker than the number suggests. |
|         Krum         |                   Integrity (resisting a poisoned update).                            | It is not a privacy mechanism. |

Regulatory context, in brief:

- **GDPR.** Data minimization (Art. 5), data protection by design (Art. 25) and security of
  processing (Art. 32) are the relevant ideas. Network records can contain personal data such as IP
  addresses. This pipeline drops them, but nothing here establishes that the model is anonymous.
- **HIPAA.** Applies to real protected health information. The hospital dataset is a public
  research set of simulated device data. Real safeguards (access control, encryption in transit,
  audit logs) are not implemented here.
- **PCI-DSS.** Not applicable to the data used: it contains network traffic, not cardholder data.

A real deployment would need legal review.

## Limitations

- **Simulated federation.** Five different public datasets reduced to about 14 coarse features. The
  reduction is lossy: Edge-IIoTset is per-packet and CSIC is HTTP text turned into counts. In
  E-commerce, 1,100 of 4,849 distinct feature vectors (22.7%, covering 67.8% of rows) appear with
  both labels, and the best possible accuracy on these features is 0.897.
- **Split correction and metric freshness.** The previous row-level split placed exact duplicate
  model-input vectors across partitions; the new group-aware split was checked and showed zero
  shared feature hashes between train/validation/test for all five clients. Because feature groups
  can be very large or contain conflicting labels, the resulting row proportions and class
  proportions are not always exactly 70/15/15 or identical across splits. The metrics above are from
  the old split and must not be treated as results for the corrected split; rerun every experiment
  and regenerate the summaries before drawing performance conclusions. If a class occurs in only
  one distinct feature group, leakage-free evaluation of that class is impossible: the group must
  remain in training, so that class will be absent from held-out evaluation for that client.
- **Single runs.** Training is seeded only for the initial weights and the data splits. Run-to-run
  variation is large (for example E-commerce ROC AUC ranged from 0.53 to 0.73 across runs).
- **Hospital remains weak** at the default cutoff. The ROC AUC of about 0.77 suggests the cutoff and
  the class imbalance account for part of that, but this was not tested.
- **Federated DP is not a usable privacy-utility result.** Class weighting interacts with
  clipping; a federated result at a useful epsilon was not obtained. DP was demonstrated only on
  E-commerce.
- **The membership attack is the simplest kind.** Stronger attacks (shadow models, likelihood-ratio
  attacks) and the other four clients were not tried. A null result is not proof of safety.
- **Secure aggregation is a simulation** (see Method), and the 15-round run used weak masks.
- **Krum** was tested against one attack type with one attacker and no secure aggregation.
- **Phishing is not covered.** Only the seven classes above.
- **CPU only**; the largest runs took hours.

## Not implemented

Phishing detection; trimmed-mean or median aggregation and Multi-Krum; personalized federated
learning (FedRep-style heads are the likely fix for Hospital but would change the single-model
design the privacy experiments rely on); concept-drift handling; explainability (SHAP); update
compression; ONNX export; per-client contribution (Shapley) analysis; gradient-inversion attacks;
a combined "secure aggregation plus DP, then attacked" comparison.

## Reproducibility notes

- The first three federated DP runs came from an earlier version of `run_dp.py` that named its
  files `dp_noise<sigma>_*` (batch 256, 2 local epochs, 5 rounds). The current script defaults to
  batch 1024, 5 epochs and tags file names with the settings. `summarize_results.py` expects the old
  names for those three runs.
- `dropout_history.csv` and `dropout_participation.csv` come from the second of two dropout runs.
  Which clients drop in a given round is not reproducible, but the number per round is.
- The "leaky" membership run (`ecommerce_mia_results.csv`) was interrupted and has no noise 2.0 or
  4.0 rows.
- Hand-recorded values: the run times in `summarize_results.py` (`RUNTIMES_S`) come from console
  logs.

## Known issues to clean up

- A comment in `server/dropout_strategy.py` says a run "can be reproduced exactly". The number of
  clients per round repeats, but which clients take part does not.
- The docstring of `experiments/mia.py` implies the "naive" AUC shows duplicates hiding leakage. The
  data did not show that.
- `requirements.txt` lists `imbalanced-learn`, which is not used.

## References

- Bonawitz et al., *Practical Secure Aggregation for Privacy-Preserving Machine Learning*, CCS 2017.
- Blanchard et al., *Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent* (Krum),
  NeurIPS 2017.
- Li et al., *Federated Optimization in Heterogeneous Networks* (FedProx), MLSys 2020.
- Abadi et al., *Deep Learning with Differential Privacy*, CCS 2016.
- Yousefpour et al., *Opacus: User-Friendly Differential Privacy Library in PyTorch*, 2021.
- Beutel et al., *Flower: A Friendly Federated Learning Framework*, 2020.
- Yeom et al., *Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting*, CSF 2018.
- Bagdasaryan et al., *Differential Privacy Has Disparate Impact on Model Accuracy*, NeurIPS 2019.
- Collins et al., *Exploiting Shared Representations for Personalized Federated Learning*
  (FedRep), ICML 2021.
- Sharafaldin et al., *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic
  Characterization* (CIC-IDS2017), ICISSP 2018.
- Hady et al., *Intrusion Detection System for Healthcare Systems Using Medical and Network Data:
  A Comparison Study* (WUSTL-EHMS-2020), IEEE Access 2020.
- Ferrag et al., *Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and
  IIoT Applications for Centralized and Federated Learning*, IEEE Access 2022.
- Torrano-Gimenez, Perez-Villegas and Alvarez Maranon, *HTTP dataset CSIC 2010*.

## License

alembic==1.20.0
altair==6.3.0
annotated-doc==0.0.5
annotated-types==0.8.0
anyio==4.15.1
attrs==26.1.0
certifi==2026.7.22
cffi==2.1.1
charset-normalizer==3.5.2
click==8.5.0
cloudpickle==3.1.2
colorama==0.4.6
contourpy==1.3.3
cryptography==46.0.7
cycler==0.12.1
fastapi==0.138.2
filelock==4.0.12
flwr==1.39.0
fonttools==4.66.1
fsspec==2026.9.0
greenlet==3.5.6
grpcio==1.84.0
grpcio-health-checking==1.81.1
h11==0.16.0
httpcore==1.0.9
httptools==0.8.0
httpx==0.28.1
idna==3.20
imbalanced-learn==0.14.2
iterators==0.0.2
itsdangerous==2.2.0
Jinja2==3.1.6
joblib==1.6.0
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
kiwisolver==1.5.1
Mako==1.4.3
markdown-it-py==4.2.0
MarkupSafe==3.0.4
matplotlib==3.11.2
mdurl==0.1.2
mpmath==1.3.0
msgpack==1.2.3
narwhals==2.26.0
networkx==3.6.1
numpy==2.4.6
opacus==1.6.0
opt_einsum==3.4.0
packaging==25.0
pandas==3.0.6
pathspec==1.1.1
pillow==12.3.0
plotly==7.1.0
prompt_toolkit==3.0.53
protobuf==6.33.6
pyarrow==25.0.1
pycparser==3.1
pycryptodome==3.24.0
pydantic==2.13.5
pydantic_core==2.46.5
pydeck==0.9.3
Pygments==2.21.0
pyparsing==3.3.3
python-dateutil==2.9.0.post0
python-dotenv==1.2.4
python-multipart==0.0.32
PyYAML==6.0.3
ray==2.55.1
referencing==0.37.0
requests==2.34.2
rich==14.3.4
rpds-py==2026.9.1
scikit-learn==1.9.1
scipy==1.17.1
shellingham==1.5.4
six==1.17.0
sklearn-compat==0.1.6
SQLAlchemy==2.1.4
starlette==1.3.1
streamlit==1.65.0
sympy==1.13.1
threadpoolctl==3.7.0
toml==0.10.2
tomli==2.5.0
tomli_w==1.2.0
torch==2.5.1+cpu
typer==0.20.1
typing-inspection==0.4.4
typing_extensions==4.16.0
tzdata==2026.5
urllib3==2.8.0
uv==0.11.33
uvicorn==0.49.0
watchdog==6.0.0
watchfiles==1.3.0
wcwidth==0.9.2
websockets==17.2