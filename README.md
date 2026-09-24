#  Insider Threat Detection (CERT r4.2)

An end-to-end machine learning pipeline for detecting insider threats from enterprise activity logs — built by a 3-person team using the CERT r4.2 dataset.

##  Key Findings
- **Per-user behavioral baselining dramatically improved detection.** After adding features that compare each user's activity to their own historical norm, One-Class SVM's recall jumped from 29.4% to 81.5%, and its ROC-AUC from 0.62 to 0.892.
- **Machine learning meaningfully outperforms simple rules.** A naive rule-based detector caught 0 of 265 malicious cases in testing; every ML model caught more, with One-Class SVM catching ~216 of 265.

See `REPORT.md` for the full methodology and analysis.

##  Team
| Member | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| **Manas** | Data pipeline, feature engineering, labeling, train/test split | Per-user behavioral baselining, combined risk-scoring ensemble | Rule-based baseline comparison |
| **Pushkar** | Isolation Forest + One-Class SVM | SHAP explainability, alert fatigue analysis, model persistence | Open-source packaging (LICENSE, requirements.txt, `run_pipeline.py`) |
| **Aakash** | Autoencoder | Interactive Streamlit dashboard, autoencoder explainability | Live-mode real-time monitoring agent |

##  Data Pipeline
Raw CERT r4.2 log files (~**4.7 GB**: `logon.csv`, `device.csv`, `file.csv`, `email.csv`, `http.csv`) are aggregated into **daily per-user behavioral features**.

### Features (14 total)

**Core activity features (9):**

| Feature | Description |
|---|---|
| `login_hour` | Hour of first logon session that day |
| `after_hours_flag` | 1 if login was before 6 AM or after 6 PM |
| `session_duration_mins` | Total logged-in minutes that day |
| `usb_events_count` | Number of USB connect events that day |
| `files_accessed_count` | Number of file access events that day |
| `email_count` | Number of emails sent that day |
| `unique_domains_visited` | Number of unique web domains visited that day |
| `email_ext_recipient_count` | Number of emails sent outside the `dtaa.com` domain |
| `file_copy_to_removable` | File access during an active USB connect/disconnect window (exfiltration signal) |

**Per-user behavioral baseline features (5) — Phase 2:**

| Feature | Description |
|---|---|
| `usb_events_count_zscore` | Today's USB count vs. this user's own 30-day rolling average |
| `files_accessed_count_zscore` | Today's file access vs. this user's own norm |
| `email_count_zscore` | Today's email count vs. this user's own norm |
| `session_duration_mins_zscore` | Today's session length vs. this user's own norm |
| `days_since_last_spike` | Days since this user last showed unusual activity (any z-score > 2) |

### Labeling
Ground-truth malicious activity from `insiders.csv` (filtered to r4.2). A user-day is labeled `is_malicious = 1` if it falls within a known malicious activity window.

### Train/Test Split
**Time-based 80/20 split** (not random) — trains on past behavior, evaluates on future behavior, avoiding lookahead leakage.

### Exported Files (`data/processed/`)

| File | Contents | Intended use |
|---|---|---|
| `X_train_raw.csv` / `y_train_raw.csv` | Real, unresampled data | Models needing true class distribution |
| `X_train_benign.csv` | Benign-only rows | **Anomaly detection models** (Isolation Forest, OC-SVM, Autoencoder) |
| `X_train_smote.csv` / `y_train_smote.csv` | SMOTE-balanced data | Supervised baseline only — **not** for anomaly models |
| `X_test.csv` / `y_test.csv` | Real, untouched test set | Evaluation for all models |

 Anomaly detection models should train on `X_train_benign.csv` or `X_train_raw.csv` — **never** `X_train_smote.csv`.

##  Results

| Model | ROC-AUC | Recall | Malicious Caught | False Positives |
|---|---|---|---|---|
| Isolation Forest | 0.876 | — | — | — |
| Autoencoder | 0.71 | 12.1% | 32 / 265 | 3,360 |
| One-Class SVM | 0.892 | 81.5% | ~216 / 265 | — |
| Rule-Based Baseline | — | 0% | 0 / 265 | 180 |

See `reports/roc_comparison.png` for the full ROC curve comparison and `reports/alert_fatigue_chart.png` for the precision-vs-workload tradeoff analysis.

##  Explainability
SHAP-based explanations reveal each model attends to different threat signatures:
- **Isolation Forest** → USB activity + file access + unusual login timing (exfiltration pattern)
- **One-Class SVM** → extremely long session durations (13-21+ hours)

See `reports/iso_forest_explanations.csv` and `reports/ocsvm_explanations.csv`.

##  Combined Risk Scoring
All three models' outputs are rescaled to 0-100 and combined into a single risk score per user-day — `reports/combined_risk_scores.csv` — mirroring real-world UEBA (User and Entity Behavior Analytics) security tools.

##  Dashboard & Live Monitoring
An interactive Streamlit dashboard (`src/aakash/dashboard/app.py`) lets you explore per-user activity, flagged days, model comparisons, and risk scores. A live-mode extension (`src/aakash/dashboard/live_mode.py`) simulates real-time event scoring, showing per-model scores, a combined risk score, and a live-updating miss-rate tracker.

```bash
streamlit run src/aakash/dashboard/app.py
streamlit run src/aakash/dashboard/live_mode.py
```

##  ML vs. Rule-Based Detection
A naive rule-based detector (flagging high USB activity or after-hours file access) was built for comparison. It caught **0 of 265** malicious cases in testing, while every ML model caught more — providing concrete justification for the machine learning approach over simple static thresholds. See `REPORT.md` for full analysis.

##  Getting Started
```bash
# Clone the repo
git clone https://github.com/manas765/insider-threat-detection.git
cd insider-threat-detection

# Install dependencies
pip install -r requirements.txt

# Set up the CERT r4.2 raw dataset in data/raw/ (see below)

# Run the full pipeline (trains and saves all models)
python run_pipeline.py
```

**Dataset:** This project uses the CERT r4.2 Insider Threat Dataset (Carnegie Mellon University Software Engineering Institute). The raw dataset (~4.7 GB) is not included in this repo due to size — download it separately and place the raw CSVs in `data/raw/` before running the pipeline.

##  License
Released under the MIT License — see `LICENSE` for details.

## Status

**Phase 1 — Complete**
- [x] Data pipeline, labeling, train/test split, exports
- [x] Isolation Forest + OC-SVM trained
- [x] Autoencoder trained
- [x] Shared evaluation framework
- [x] Final report

**Phase 2 — Complete**
- [x] Per-user behavioral baselining (5 new features)
- [x] SHAP explainability for all 3 models
- [x] Alert fatigue / threshold sweep analysis
- [x] Combined risk-scoring ensemble
- [x] Interactive dashboard

**Phase 3 — Complete**
- [x] Rule-based baseline comparison
- [x] Open-source packaging (LICENSE, requirements.txt, `run_pipeline.py`)
- [x] Live-mode real-time monitoring agent

##  Full Report
See [`REPORT.md`](./REPORT.md) for the complete methodology, results, and discussion.