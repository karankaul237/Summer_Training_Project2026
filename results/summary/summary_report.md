# FedShield-IDS: results summary

## Pooled class distribution across all clients

| class | proportion |
|---|---|
| benign | 0.783 |
| dos_ddos | 0.135 |
| recon | 0.034 |
| web_attack | 0.027 |
| credential_attack | 0.012 |
| malware | 0.008 |
| spoofing_mitm | 0.001 |

## How different each client's data is (Jensen-Shannon divergence)

_0 = same class mix as the pooled data; higher = more different. n_classes_present is out of 7._

| client | n_rows | n_classes_present | js_divergence |
|---|---|---|---|
| ecommerce | 61065 | 2 | 0.250 |
| hospital | 16318 | 2 | 0.170 |
| bank | 586144 | 3 | 0.104 |
| university | 1988109 | 4 | 0.024 |
| manufacturing | 2219201 | 7 | 0.005 |

## Baseline FedAvg: how client weighting changed the result

_Runs differ in length (see rounds): the sample-count run was only a short first test. Per-client numbers are the mean of the last 3 rounds (or of all rounds if fewer)._

| run | rounds | client | accuracy | macro_f1 |
|---|---|---|---|---|
| equal weights + FedProx (final) | 15 | bank | 0.728 | 0.423 |
| log weights | 15 | bank | 0.624 | 0.283 |
| sample-count weights | 3 | bank | 0.032 | 0.021 |
| equal weights + FedProx (final) | 15 | ecommerce | 0.362 | 0.372 |
| log weights | 15 | ecommerce | 0.590 | 0.371 |
| sample-count weights | 3 | ecommerce | 0.011 | 0.018 |
| equal weights + FedProx (final) | 15 | hospital | 0.878 | 0.490 |
| log weights | 15 | hospital | 0.874 | 0.467 |
| sample-count weights | 3 | hospital | 0.000 | 0.000 |
| equal weights + FedProx (final) | 15 | manufacturing | 0.869 | 0.405 |
| log weights | 15 | manufacturing | 0.707 | 0.195 |
| sample-count weights | 3 | manufacturing | 0.832 | 0.343 |
| equal weights + FedProx (final) | 15 | university | 0.689 | 0.543 |
| log weights | 15 | university | 0.698 | 0.594 |
| sample-count weights | 3 | university | 0.262 | 0.211 |

## Secure aggregation vs plain FedAvg (same settings)

_This run used mask scale 0.01, which a later check showed hides an update only weakly (a masked update was ~7% off the real one). Mask size changes only what a server can see, not the aggregated model (aggregation error ~1e-5 even at scale 100), so these accuracy numbers should still hold._

| client | baseline_accuracy | baseline_macro_f1 | secagg_accuracy | secagg_macro_f1 |
|---|---|---|---|---|
| bank | 0.728 | 0.423 | 0.757 | 0.452 |
| ecommerce | 0.362 | 0.372 | 0.655 | 0.623 |
| hospital | 0.878 | 0.490 | 0.875 | 0.470 |
| manufacturing | 0.869 | 0.405 | 0.867 | 0.399 |
| university | 0.689 | 0.543 | 0.683 | 0.463 |

## Secure aggregation runtime overhead

_Run times are recorded by hand from the console logs (see RUNTIMES_S in this script)._

| run | seconds | overhead_pct |
|---|---|---|
| baseline | 2415.110 | 0.000 |
| secagg | 2434.390 | 0.798 |

## One malicious client ('university'): plain FedAvg vs Krum

_Final round only. Both runs exclude secure aggregation, because Krum needs to see individual updates and SecAgg hides them._

| client | fedavg_attacked_accuracy | fedavg_attacked_macro_f1 | krum_defended_accuracy | krum_defended_macro_f1 |
|---|---|---|---|---|
| bank | 0.000 | 0.000 | 0.004 | 0.002 |
| ecommerce | 0.000 | 0.000 | 0.731 | 0.728 |
| hospital | 0.125 | 0.111 | 0.000 | 0.000 |
| manufacturing | 0.001 | 0.000 | 0.047 | 0.013 |
| university | 0.000 | 0.000 | 0.000 | 0.000 |

## Which client's update Krum picked each round

_Krum keeps ONE client's update per round. If the same client keeps winning, the global model drifts toward that client (the lock-in effect)._

| client | rounds_selected | rounds_total | is_attacker |
|---|---|---|---|
| ecommerce | 6 | 10 | False |
| hospital | 4 | 10 | False |
| bank | 0 | 10 | False |
| manufacturing | 0 | 10 | False |
| university | 0 | 10 | True |

## Differential privacy across all five clients (final round)

_These federated DP runs collapsed toward predicting the majority class. The likely cause, measured on Hospital, is that clipping at 1.0 cancels the class weights (minority gradients were ~7x larger before clipping). Treat the accuracy columns as evidence of the collapse, not as a privacy-utility curve. The epsilon values are correct accounting for each run. Epsilon differs by client because smaller clients sample a larger fraction of their data per step._

| noise_multiplier | batch_size | local_epochs | rounds | client | train_size | total_steps | epsilon | accuracy | macro_f1 |
|---|---|---|---|---|---|---|---|---|---|
| 0.500 | 256 | 2 | 5 | bank | 410300 | 16030 | 5.450 | 0.950 | 0.325 |
| 0.500 | 256 | 2 | 5 | ecommerce | 42745 | 1670 | 12.643 | 0.523 | 0.411 |
| 0.500 | 256 | 2 | 5 | hospital | 11422 | 450 | 21.056 | 0.875 | 0.467 |
| 0.500 | 256 | 2 | 5 | manufacturing | 1553437 | 60690 | 3.670 | 0.730 | 0.129 |
| 0.500 | 256 | 2 | 5 | university | 1391674 | 54370 | 3.781 | 0.797 | 0.268 |
| 1.000 | 256 | 2 | 5 | bank | 410300 | 16030 | 0.691 | 0.975 | 0.329 |
| 1.000 | 256 | 2 | 5 | ecommerce | 42745 | 1670 | 1.594 | 0.590 | 0.371 |
| 1.000 | 256 | 2 | 5 | hospital | 11422 | 450 | 3.375 | 0.875 | 0.467 |
| 1.000 | 256 | 2 | 5 | manufacturing | 1553437 | 60690 | 0.528 | 0.728 | 0.121 |
| 1.000 | 256 | 2 | 5 | university | 1391674 | 54370 | 0.544 | 0.815 | 0.290 |
| 2.000 | 256 | 2 | 5 | bank | 410300 | 16030 | 0.165 | 0.977 | 0.330 |
| 2.000 | 256 | 2 | 5 | ecommerce | 42745 | 1670 | 0.515 | 0.590 | 0.371 |
| 2.000 | 256 | 2 | 5 | hospital | 11422 | 450 | 1.088 | 0.875 | 0.467 |
| 2.000 | 256 | 2 | 5 | manufacturing | 1553437 | 60690 | 0.118 | 0.729 | 0.123 |
| 2.000 | 256 | 2 | 5 | university | 1391674 | 54370 | 0.119 | 0.824 | 0.437 |
| 1.000 | 1024 | 5 | 10 | bank | 410300 | 20050 | 2.055 | 0.972 | 0.329 |
| 1.000 | 1024 | 5 | 10 | ecommerce | 42745 | 2100 | 7.744 | 0.510 | 0.377 |
| 1.000 | 1024 | 5 | 10 | hospital | 11422 | 600 | 17.624 | 0.875 | 0.467 |
| 1.000 | 1024 | 5 | 10 | manufacturing | 1553437 | 75900 | 1.042 | 0.729 | 0.126 |
| 1.000 | 1024 | 5 | 10 | university | 1391674 | 68000 | 1.087 | 0.806 | 0.284 |

## Privacy-utility tradeoff on E-commerce (the client where DP works)

_Single client, trained alone for 20 epochs. Lower epsilon = stronger privacy. Epsilon is per row; each record has ~10 copies, so protection for a repeated pattern is weaker than the nominal number._

| setting | noise_multiplier | clip | epsilon | balanced_accuracy | recall_benign | recall_web_attack | utility_vs_plain |
|---|---|---|---|---|---|---|---|
| plain | - | 1.000 | inf | 0.735 | 0.716 | 0.754 | 1.000 |
| dp_noise0.5 | 0.500 | 1.000 | 16.539 | 0.674 | 0.716 | 0.633 | 0.918 |
| dp_noise1.0 | 1.000 | 1.000 | 2.126 | 0.674 | 0.716 | 0.633 | 0.918 |
| dp_noise2.0 | 2.000 | 1.000 | 0.743 | 0.674 | 0.716 | 0.633 | 0.918 |
| dp_noise4.0 | 4.000 | 1.000 | 0.329 | 0.681 | 0.746 | 0.615 | 0.926 |

## Training with clients dropping out of rounds

_Average participants per round: 3.33 of 5. This is the SECOND of two dropout runs: the first run's per-client numbers were overwritten by the re-run._

| client | baseline_accuracy | baseline_macro_f1 | dropout_accuracy | dropout_macro_f1 | rounds_trained | rounds_total |
|---|---|---|---|---|---|---|
| bank | 0.728 | 0.423 | 0.722 | 0.454 | 10 | 15 |
| ecommerce | 0.362 | 0.372 | 0.502 | 0.522 | 11 | 15 |
| hospital | 0.878 | 0.490 | 0.896 | 0.593 | 12 | 15 |
| manufacturing | 0.869 | 0.405 | 0.809 | 0.349 | 9 | 15 |
| university | 0.689 | 0.543 | 0.645 | 0.387 | 8 | 15 |

## Membership inference attack on E-commerce models

_AUC 0.5 = attacker no better than guessing. 'saved' = the small models from the DP sweep (their AUC below 0.5 is probably an artifact of how non-members were chosen; not confirmed); 'leaky' = bigger models built to overfit. The leaky run was interrupted, so it has no noise 2.0/4.0 rows._

| model | kind | noise_multiplier | epsilon | auc | auc_lo | auc_hi | advantage | acc_members | acc_nonmembers | acc_gap | ci_includes_chance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| saved_plain | saved | - | inf | 0.469 | 0.438 | 0.499 | 0.005 | 0.874 | 0.917 | -0.043 | False |
| saved_dp_noise0.5 | saved | 0.500 | 16.539 | 0.472 | 0.438 | 0.501 | 0.006 | 0.872 | 0.915 | -0.043 | True |
| saved_dp_noise1.0 | saved | 1.000 | 2.126 | 0.470 | 0.438 | 0.500 | 0.006 | 0.872 | 0.915 | -0.043 | False |
| saved_dp_noise2.0 | saved | 2.000 | 0.743 | 0.470 | 0.438 | 0.500 | 0.009 | 0.872 | 0.915 | -0.043 | False |
| saved_dp_noise4.0 | saved | 4.000 | 0.329 | 0.467 | 0.439 | 0.497 | 0.010 | 0.872 | 0.915 | -0.043 | False |
| leaky_plain | leaky | - | inf | 0.504 | 0.490 | 0.518 | 0.029 | 0.827 | 0.834 | -0.007 | True |
| leaky_dp_noise0.5 | leaky | 0.500 | 21.429 | 0.503 | 0.487 | 0.516 | 0.020 | 0.813 | 0.820 | -0.007 | True |
| leaky_dp_noise1.0 | leaky | 1.000 | 2.905 | 0.503 | 0.489 | 0.517 | 0.024 | 0.806 | 0.816 | -0.010 | True |

## Membership inference under stress (few training vectors, big models)

_vectors = number of distinct training records (0 means all of them). No setting leaked, so the DP stage never ran._

| vectors | hidden | epochs | noise_multiplier | epsilon | auc | auc_lo | auc_hi | advantage | acc_members | acc_nonmembers | acc_gap | ci_includes_chance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 256 | 100 | - | inf | 0.503 | 0.488 | 0.516 | 0.019 | 0.825 | 0.835 | -0.010 | True |
| 0 | 256 | 400 | - | inf | 0.502 | 0.486 | 0.517 | 0.026 | 0.837 | 0.840 | -0.003 | True |
| 0 | 512 | 100 | - | inf | 0.504 | 0.491 | 0.517 | 0.027 | 0.823 | 0.833 | -0.010 | True |
| 0 | 512 | 400 | - | inf | 0.502 | 0.486 | 0.516 | 0.025 | 0.828 | 0.834 | -0.006 | True |
| 300 | 256 | 100 | - | inf | 0.499 | 0.454 | 0.548 | 0.027 | 0.813 | 0.817 | -0.003 | True |
| 300 | 256 | 400 | - | inf | 0.511 | 0.470 | 0.560 | 0.043 | 0.817 | 0.813 | 0.003 | True |
| 300 | 512 | 100 | - | inf | 0.503 | 0.460 | 0.551 | 0.047 | 0.817 | 0.817 | 0.000 | True |
| 300 | 512 | 400 | - | inf | 0.511 | 0.472 | 0.559 | 0.050 | 0.817 | 0.813 | 0.003 | True |

## Every training run, round by round

_Full per-round results are in the CSV (used by the dashboard's learning curves and heatmap). This table shows each run's mean macro-F1 over its last 3 rounds. Runs differ in length (see rounds)._

| run | rounds | bank | ecommerce | hospital | manufacturing | university |
|---|---|---|---|---|---|---|
| baseline (final) | 15 | 0.423 | 0.372 | 0.490 | 0.405 | 0.543 |
| secure aggregation | 15 | 0.452 | 0.623 | 0.470 | 0.399 | 0.463 |
| client dropout | 15 | 0.454 | 0.522 | 0.593 | 0.349 | 0.387 |
| log weights | 15 | 0.283 | 0.371 | 0.467 | 0.195 | 0.594 |
| sample-count weights | 3 | 0.021 | 0.018 | 0.000 | 0.343 | 0.211 |
| FedAvg under attack | 10 | 0.000 | 0.000 | 0.111 | 0.000 | 0.000 |
| Krum under attack | 10 | 0.048 | 0.729 | 0.000 | 0.012 | 0.018 |

## Detection ROC: attack vs benign traffic, per client (AUC shown)

_Score = 1 - P(benign); 'attack' means any non-benign class. AUC 1.0 is perfect separation, 0.5 is chance. Measured on each client's held-out TEST split, which was never used for tuning. The CSV holds the full curves (201 points each)._

| model | bank | ecommerce | hospital | manufacturing | university |
|---|---|---|---|---|---|
| baseline (final) | 0.970 | 0.528 | 0.774 | 0.948 | 0.720 |
| secure aggregation | 0.907 | 0.732 | 0.775 | 0.934 | 0.707 |
| client dropout | 0.847 | 0.640 | 0.775 | 0.976 | 0.857 |
| log weights | 0.838 | 0.617 | 0.774 | 0.828 | 0.907 |
| DP noise 1.0 (federated) | 0.896 | 0.586 | 0.774 | 0.825 | 0.594 |

## Membership-attack ROC on the saved E-commerce models (AUC shown)

_Same loss-threshold attack and the same clean member / non-member sets as the main attack table. A curve on the diagonal means the attacker is guessing. Curves slightly BELOW the diagonal are probably an artifact of how non-members were chosen (not confirmed). Only the five saved models are drawn; the leaky and stress models were not saved._

| model | auc |
|---|---|
| plain | 0.469 |
| dp_noise0.5 | 0.472 |
| dp_noise1.0 | 0.470 |
| dp_noise2.0 | 0.470 |
| dp_noise4.0 | 0.467 |
