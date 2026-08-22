# Insider Threat Detection (CERT r4.2)

ML project to detect insider threats using behavioral features extracted from the CERT r4.2 dataset.

## Team
- **Manas** — data pipeline (feature engineering, labeling, train/test split, exports)
- **Pushkar** — Isolation Forest + One-Class SVM
- **Aakash** — Autoencoder (v2)

## Data Pipeline

Raw CERT r4.2 log files (`logon.csv`, `device.csv`, `file.csv`, `email.csv`, `http.csv`) are aggregated into **daily per-user features**.

### Features (9 total)

| Feature | Description |
|---|---|
| `login_hour` | Hour of first logon session that day |
| `after_hours_flag` | 1 if login was before 6 AM or after 6 PM |
| `session_duration_mins` | Total logged-in minutes that day |
| `usb_events_count` | Number of USB connect events that day |
| `files_accessed_count` | Number of file access events that day |
| `email_count` | Number of emails sent that day |
| `unique_domains_visited` | Number of unique web domains visited that day |
| `email_ext_recipient_count` | Number of emails sent to recipients outside the `dtaa.com` domain |
| `file_copy_to_removable` | Number of file accesses that occurred during an active USB connect/disconnect window (possible exfiltration signal) |

### Labeling

Ground-truth malicious activity comes from `insiders.csv` (filtered to r4.2). A user-day is labeled `is_malicious = 1` if it falls within that user's known malicious activity window (`start` to `end` dates); otherwise `0`.

### Train/Test Split

The data is **sorted by date** and split **80/20** (not random) — the first 80% of days become the train set, the last 20% become the test set. This mimics a real deployment where you train on past behavior and detect anomalies in future behavior.

### Exported Files (`data/processed/`)

| File | Contents | Intended use |
|---|---|---|
| `X_train_raw.csv` / `y_train_raw.csv` | Real, unresampled train data (all classes) | Models that need true class distribution |
| `X_train_benign.csv` | Only benign (`is_malicious == 0`) train rows | **Autoencoder / one-class models** — train on "normal" behavior only |
| `X_train_smote.csv` / `y_train_smote.csv` | SMOTE-balanced train data (synthetic minority samples added) | Supervised baseline models only — **do not use for unsupervised/anomaly models** |
| `X_test.csv` / `y_test.csv` | Real, untouched test set | Evaluation for all models |

⚠️ **Important:** Isolation Forest, OC-SVM, and the autoencoder should train on `X_train_benign.csv` or `X_train_raw.csv` — **not** `X_train_smote.csv`. SMOTE generates synthetic minority-class samples, which distorts what "normal" looks like for anomaly-detection models.

## Status
- [x] Data pipeline built, labeled, split, exported
- [ ] Isolation Forest + OC-SVM trained (Pushkar)
- [ ] Autoencoder retrained on v2 data (Aakash)
- [ ] Shared evaluation across all 3 models
- [ ] Final report

**Deadline:** Aug 30
