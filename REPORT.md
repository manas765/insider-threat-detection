## 1. Introduction

Insider threats — malicious or negligent actions by employees, contractors, or other trusted users within an organization — are among the hardest security risks to detect, since insiders already have legitimate access to systems and data. Unlike external attacks, insider threats often look like normal, everyday activity, making them difficult to catch using traditional rule-based security systems.

This project applies machine learning to detect anomalous user behavior that may indicate insider threats, using the **CERT r4.2 Insider Threat dataset** — a synthetic but realistic dataset simulating organizational logs (logons, file access, email, web browsing, and USB device activity) alongside known ground-truth insider threat scenarios.

The goal is to engineer meaningful behavioral features from raw activity logs, then train and compare multiple anomaly detection approaches — **Isolation Forest**, **One-Class SVM**, and an **Autoencoder** — to identify which method most effectively separates malicious behavior from normal daily activity, despite the extreme class imbalance typical of real-world insider threat data.

## 2. Data & Feature Engineering

### Data Sources
The CERT r4.2 insider threat dataset was used, drawing from five raw log files:
- `logon.csv` — user login/logout activity
- `device.csv` — USB device connect/disconnect events
- `file.csv` — file access events
- `email.csv` — email send activity
- `http.csv` — web browsing activity

Each log was aggregated to a **daily, per-user level**, so every row in the final dataset represents one user's activity for one day.

### Features (9 total)

| Feature | Description |
|---|---|
| `login_hour` | Hour of the user's first logon session that day |
| `after_hours_flag` | 1 if the login occurred before 6 AM or after 6 PM, else 0 |
| `session_duration_mins` | Total minutes logged in that day |
| `usb_events_count` | Number of USB connect events that day |
| `files_accessed_count` | Number of file access events that day |
| `email_count` | Number of emails sent that day |
| `unique_domains_visited` | Number of distinct web domains visited that day |
| `email_ext_recipient_count` | Number of emails sent to recipients outside the internal `dtaa.com` domain |
| `file_copy_to_removable` | Number of file accesses that occurred during an active USB connect/disconnect window — a proxy for possible data exfiltration via removable media |

The last feature, `file_copy_to_removable`, was engineered by matching USB connect/disconnect time windows against file access timestamps for the same user and machine, flagging file activity that overlapped with an active USB session.

### Labeling
Ground-truth malicious activity labels came from `insiders.csv`, filtered to the r4.2 dataset. A user-day was labeled `is_malicious = 1` if it fell within that user's known malicious activity window (`start` to `end` date), and `0` otherwise. This produced a strongly imbalanced dataset (328,906 benign user-days vs. 1,362 malicious user-days).

### Train/Test Split
Rather than a random split, the data was sorted chronologically and split **80/20 by date** — the earliest 80% of days form the training set, and the most recent 20% form the test set. This reflects a realistic deployment scenario: detecting anomalies in future behavior after training on past behavior, and avoids leaking future information into training.

### Handling Class Imbalance
Because malicious user-days are rare, three versions of the training data were exported to support different modeling approaches:

| File | Description | Intended use |
|---|---|---|
| `X_train_raw.csv` | Unmodified, imbalanced training data | Models needing the true class distribution |
| `X_train_benign.csv` | Only benign (`is_malicious = 0`) rows | One-class / anomaly detection models (Isolation Forest, OC-SVM, Autoencoder), which learn a profile of "normal" behavior |
| `X_train_smote.csv` | SMOTE-balanced training data (synthetic minority samples added) | Reserved for a supervised baseline comparison only — not used for the anomaly detection models, since synthetic minority samples would distort the "normal" behavior boundary these models rely on |

The test set (`X_test.csv`, `y_test.csv`) was left untouched and unresampled in all cases, ensuring evaluation always reflects real-world class distribution.

## 3. Models

Three anomaly detection approaches were trained and compared, each learning a profile of "normal" user behavior from the benign-only training data (`X_train_benign.csv`), then flagging deviations as potential insider threats.

### Training & Evaluation Methodology
Both classical models were trained exclusively on benign (non-malicious) rows, following a semi-supervised anomaly-detection approach. Hyperparameters were selected using a validation split carved out of the training data — not the final test set — to avoid the optimistic bias that comes from repeatedly tuning against the same evaluation set. X_test/y_test was touched exactly once, for the final reported numbers.

### 3.1 Isolation Forest
Isolation Forest isolates anomalies by randomly partitioning the feature space; anomalous points require fewer partitions to isolate than normal points. Final hyperparameters: `n_estimators=300`, `max_features=0.5`, `max_samples=65536`, `contamination=0.005`. On the held-out test set (66,054 rows, 0.40% malicious), it achieved ROC-AUC 0.84, but at its calibrated threshold correctly flagged only 1 of 265 malicious cases — a clear illustration of ROC-AUC's limits under severe class imbalance.

### 3.2 One-Class SVM
One-Class SVM learns a boundary around normal behavior, flagging points outside it as anomalous. Due to poor scalability (training cost grows roughly quadratically with sample size), it was trained on a random subsample of 30,000 benign rows. Final hyperparameters: `kernel='rbf'`, `nu=0.005`, `gamma=0.8`. It achieved a lower ROC-AUC of 0.62, but a far higher recall of 29.4% (78/265) — at the cost of more false positives (4,630). Despite the weaker ranking metric, OC-SVM was the more practically useful detector of the two.

### 3.3 Autoencoder
The autoencoder was trained to reconstruct normal daily behavior patterns from benign data; anomalies are flagged where reconstruction error exceeds a chosen threshold. Evaluated on the real held-out test set, it achieved ROC-AUC 0.71 and PR-AUC 0.0065 (see `reports/roc_autoencoder.png`, `reports/pr_autoencoder.png`).

## 4. Evaluation

All three models were evaluated on the same held-out, chronologically-split test set (66,054 user-days, 265 malicious — 0.40%), using a shared evaluation script (`notebooks/evaluate.py`) to ensure metrics were computed consistently across models.

### Results Summary

| Model | ROC-AUC | Recall | Malicious Caught | False Positives |
|---|---|---|---|---|
| Isolation Forest | 0.84 | 0.4% | 1 / 265 | Low |
| One-Class SVM | 0.62 | 29.4% | 78 / 265 | 4,630 |
| Autoencoder | 0.71 | — | — | — |

*(See `reports/roc_comparison.png` for the combined ROC curve across all three models, and `reports/pr_autoencoder.png` for the autoencoder's precision-recall curve.)*

### Methodology Note
An earlier version of the comparison chart showed a perfect ROC-AUC of 1.00 for all three models. This was traced back to a bug where placeholder/dummy scores had been used in the comparison notebook instead of real model outputs — not an issue with the underlying data pipeline. Once corrected with real predictions from all three models, results dropped to the realistic, imperfect scores reported above. Hyperparameter tuning for the classical models was also restructured mid-project to use a separate validation split rather than repeated tuning against the test set, avoiding optimistic bias in the final reported numbers.

## 5. Results & Discussion

### ROC-AUC Is Not the Full Picture
Isolation Forest achieved the highest ROC-AUC (0.84) of the three models, yet at its calibrated decision threshold it flagged only 1 of 265 malicious user-days — effectively missing almost every real threat. One-Class SVM, despite a lower ROC-AUC (0.62), caught 78 of 265 malicious cases (29.4% recall) at its threshold. This highlights an important practical lesson: **a higher ranking metric like ROC-AUC does not guarantee better real-world detection performance**, especially under severe class imbalance (0.40% positive rate here). In a real deployment, a security team would likely prefer One-Class SVM's higher recall despite its noisier ROC-AUC and higher false-positive count, since catching more true insider threats — even with more false alarms to review — is generally more valuable than a model that stays quiet.

### Known Limitations
- **No per-user behavioral baselining.** All 9 features currently measure raw daily activity levels (e.g., total USB events, total emails sent) relative to the whole population, rather than relative to each individual user's own historical norm. Insider threat behavior often manifests as a deviation from a *specific person's* typical pattern (e.g., a user who normally has zero USB activity suddenly using a USB device), which population-level features can miss. Incorporating per-user rolling averages or deviation scores (e.g., "today's count vs. this user's 30-day average") is a natural next step and was identified as a high-value improvement, though not implemented due to project time constraints.
- **Severe class imbalance** (0.40% malicious) makes high recall inherently difficult without accepting a high false-positive rate — a fundamental trade-off in this problem domain, not specific to any one model.
- **One-Class SVM's scalability limits** required training on a 30,000-row subsample of benign data rather than the full training set, which may have limited its ability to learn a more precise decision boundary.

## 6. Conclusion

This project built an end-to-end pipeline for insider threat detection using the CERT r4.2 dataset, from raw log ingestion through feature engineering, labeling, and multi-model anomaly detection. Three models — Isolation Forest, One-Class SVM, and an Autoencoder — were trained and fairly compared using a shared evaluation framework. Results showed that ROC-AUC alone can be misleading under extreme class imbalance: One-Class SVM, despite a lower ROC-AUC, was the most practically useful model, catching nearly 30% of real malicious cases compared to Isolation Forest's near-zero detection rate at threshold.

Future improvements identified during this project include per-user behavioral baselining (comparing each user's activity to their own historical norm rather than the population), expanding the feature set with additional log sources, and exploring ensemble approaches that combine the strengths of multiple models. The team also identified and corrected a data leakage issue during development (placeholder scores inflating an early comparison chart) and adopted a validation-split methodology to avoid test-set overfitting during hyperparameter tuning — both of which reflect a rigorous, iterative approach to the modeling process.