## 1. Introduction

Insider threats — malicious or negligent actions by employees, contractors, or other trusted users within an organization — are among the hardest security risks to detect, since insiders already have legitimate access to systems and data. Unlike external attacks, insider threats often look like normal, everyday activity, making them difficult to catch using traditional rule-based security systems.

This project applies machine learning to detect anomalous user behavior that may indicate insider threats, using the **CERT r4.2 Insider Threat dataset** — a synthetic but realistic dataset simulating organizational logs (logons, file access, email, web browsing, and USB device activity) alongside known ground-truth insider threat scenarios.

The goal is to engineer meaningful behavioral features from raw activity logs, then train and compare multiple anomaly detection approaches — **Isolation Forest**, **One-Class SVM**, and an **Autoencoder** — to identify which method most effectively separates malicious behavior from normal daily activity, despite the extreme class imbalance typical of real-world insider threat data. The project was later extended (Phase 2-3) with per-user behavioral baselining, explainability, a combined risk-scoring ensemble, a rule-based baseline comparison, open-source packaging, and a live-monitoring dashboard.

## 2. Data & Feature Engineering

### Data Sources
The CERT r4.2 insider threat dataset was used, drawing from five raw log files:
- `logon.csv` — user login/logout activity
- `device.csv` — USB device connect/disconnect events
- `file.csv` — file access events
- `email.csv` — email send activity
- `http.csv` — web browsing activity

The raw dataset totaled approximately **4.7 GB**, with `http.csv` (web browsing logs) as the largest single source — large enough to require chunked processing (500,000-row batches) to read and aggregate efficiently.

Each log was aggregated to a **daily, per-user level**, so every row in the final dataset represents one user's activity for one day.

### Features (14 total)

**Core activity features (9):**

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

**Per-user behavioral baseline features (5) — added in Phase 2:**

| Feature | Description |
|---|---|
| `usb_events_count_zscore` | Today's USB count vs. this user's own 30-day rolling average |
| `files_accessed_count_zscore` | Today's file access vs. this user's own norm |
| `email_count_zscore` | Today's email count vs. this user's own norm |
| `session_duration_mins_zscore` | Today's session length vs. this user's own norm |
| `days_since_last_spike` | Days since this user last showed unusual activity (any z-score above 2) |

These features were motivated by the observation that insider threat behavior often manifests as a deviation from an individual's *own* typical pattern, rather than an unusual value relative to the overall population. Each z-score was computed using only *past* days (via a shifted rolling window) to avoid leaking future information into the feature.

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

Three anomaly detection approaches were trained and compared, each learning a profile of "normal" user behavior from the benign-only training data (`X_train_benign.csv`), then flagging deviations as potential insider threats. All results below reflect training on the full 14-feature set, including the Phase 2 per-user baseline features.

### Training & Evaluation Methodology
Both classical models were trained exclusively on benign (non-malicious) rows, following a semi-supervised anomaly-detection approach. Hyperparameters were selected using a validation split carved out of the training data — not the final test set — to avoid the optimistic bias that comes from repeatedly tuning against the same evaluation set. `X_test`/`y_test` was touched exactly once, for the final reported numbers. Training and evaluation were later consolidated into a single reproducible entry point, `run_pipeline.py`, which trains all models with fixed random seeds and saves them alongside any required preprocessing objects (e.g., the feature scaler used by One-Class SVM), ensuring bit-identical results across machines.

### 3.1 Isolation Forest
Isolation Forest isolates anomalies by randomly partitioning the feature space; anomalous points require fewer partitions to isolate than normal points. Final hyperparameters: `n_estimators=300`, `max_features=0.5`, `max_samples=65536`, `contamination=0.005`. On the held-out test set (66,054 rows, 0.40% malicious), using the full 14-feature set including per-user baselines, it achieved ROC-AUC **0.876**. *(Exact recall and malicious-caught count for this configuration are pending confirmation of the scoring methodology used.)*

### 3.2 One-Class SVM
One-Class SVM learns a boundary around normal behavior, flagging points outside it as anomalous. Due to poor scalability (training cost grows roughly quadratically with sample size), it was trained on a random subsample of 30,000 benign rows, with features standardized via a `StandardScaler` fit only on the training data (One-Class SVM is sensitive to feature scale, and applying a scaler fit on different data was found during Phase 4 testing to produce miscalibrated, unreliable results). Final hyperparameters: `kernel='rbf'`, `nu=0.005`, `gamma=0.8`. It achieved ROC-AUC **0.892** and recall of **81.5%** (216/265, with 49 missed) — a substantial improvement over its pre-baselining performance (0.62 ROC-AUC, 29.4% recall on the original 9 features), confirming that the added per-user behavioral features meaningfully improved detection.

### 3.3 Autoencoder
The autoencoder was trained to reconstruct normal daily behavior patterns from benign data; anomalies are flagged where reconstruction error exceeds a chosen threshold. Evaluated on the real held-out test set using the full 14-feature set, it achieved ROC-AUC 0.71 and PR-AUC 0.0065, catching 32 of 265 malicious cases (12.1% recall) with 3,360 false positives (see `reports/roc_autoencoder.png`, `reports/pr_autoencoder.png`).

## 4. Evaluation

All three models were evaluated on the same held-out, chronologically-split test set (66,054 user-days, 265 malicious — 0.40%), using a shared evaluation script (`notebooks/evaluate.py`) to ensure metrics were computed consistently across models.

### Results Summary

| Model | ROC-AUC | PR-AUC | Recall | Malicious Caught | False Positives |
|---|---|---|---|---|---|
| Isolation Forest | 0.876 | ~0.012 | — | — | — |
| Autoencoder | 0.71 | 0.0065 | 12.1% | 32 / 265 | 3,360 |
| One-Class SVM | 0.892 | — | 81.5% | 216 / 265 | — |

*(See `reports/roc_comparison.png` for the combined ROC curve across all three models, and `reports/pr_autoencoder.png` for the autoencoder's precision-recall curve.)*

### Methodology Note
An earlier version of the comparison chart showed a perfect ROC-AUC of 1.00 for all three models. This was traced back to a bug where placeholder/dummy scores had been used in the comparison notebook instead of real model outputs — not an issue with the underlying data pipeline. Once corrected with real predictions from all three models, results dropped to realistic, imperfect scores. Hyperparameter tuning for the classical models was also restructured mid-project to use a separate validation split rather than repeated tuning against the test set, avoiding optimistic bias in the final reported numbers. When the classical models were later retrained on the Phase 2 per-user baseline features, the same validation-split methodology and hyperparameters were reused (confirmed by identical selected hyperparameters), ruling out re-tuning against the test set as the source of the improvement seen above. A separate scaling-related discrepancy was later identified and resolved during Phase 4 adversarial testing (see Section 5).

## 5. Results & Discussion

### Impact of Per-User Baselining
Before Phase 2's per-user baseline features were added, Isolation Forest achieved the highest ROC-AUC (0.84) of the three models, yet at its calibrated threshold flagged only 1 of 265 malicious user-days — effectively missing almost every real threat. One-Class SVM's lower ROC-AUC (0.62) paired with a much higher recall (29.4%, 78/265) — illustrating that ROC-AUC alone can mislead under severe class imbalance.

After incorporating 5 per-user z-score features (comparing each user's daily activity to their own 30-day rolling norm rather than population-level counts), both classical models improved substantially. One-Class SVM now leads on both ROC-AUC (0.892) and recall (81.5%, 216/265) — a dramatic jump from its pre-baselining numbers. This confirms the hypothesis raised earlier in the project: insider threat behavior is better captured as a deviation from an individual's own baseline than as an absolute, population-level activity count. The general methodological caution about ROC-AUC under class imbalance remains valid, but with richer, personalized features, ranking quality and practical usefulness converged rather than diverged for this model.

### Known Limitations
- **Autoencoder recall remains comparatively low (12.1%)** even after the addition of per-user baseline features, suggesting reconstruction-error-based thresholding may be less sensitive to these particular engineered features than the classical models' decision boundaries. Further threshold tuning or architecture changes could be explored.
- **Severe class imbalance** (0.40% malicious) makes high recall inherently difficult without accepting a high false-positive rate — a fundamental trade-off in this problem domain, not specific to any one model.
- **One-Class SVM's scalability limits** required training on a 30,000-row subsample of benign data rather than the full training set, which may still be limiting its decision boundary precision despite the strong recall improvement.
- **Feature-scaling sensitivity** was identified as a practical pitfall during later adversarial testing: reproducing One-Class SVM's results requires reusing the exact scaler fit during original training. Fitting a new scaler on different data produced substantially miscalibrated, unreliable predictions, underscoring the importance of the project's move toward a single reproducible pipeline entry point (`run_pipeline.py`) that saves models and preprocessing objects together.

### Explainability Findings (SHAP)

To make model decisions interpretable rather than opaque anomaly scores, SHAP-based explanations were generated for each flagged case in the Isolation Forest and One-Class SVM models, identifying the top contributing features behind each flag.

**Isolation Forest** flagged cases based on a mix of signals, most commonly combinations of `usb_events_count`, `files_accessed_count`, `login_hour`, and `email_ext_recipient_count` — consistent with a classic exfiltration pattern (USB activity paired with file access at unusual hours, or high external email contact).

**One-Class SVM**, by contrast, leaned heavily on `session_duration_mins` as its dominant signal — the majority of its flagged cases involved single login sessions lasting 800-1,300+ minutes (13-21+ hours), an extreme deviation from typical daily activity.

This divergence suggests the two models are sensitive to different behavioral signatures of insider threat activity rather than simply agreeing or disagreeing on the same cases with different confidence. This supports the value of the combined risk-scoring ensemble (below): rather than relying on a single model's blind spots, combining scores from models attentive to different signal types increases the chance of catching a wider range of threat behaviors.

### ML vs. Rule-Based Detection (Phase 3)

To justify the use of machine learning over simpler, commonly-deployed security approaches, a naive rule-based detector was built and evaluated on the same held-out test set using the same evaluation methodology. The rule flagged a user-day as malicious if:
- `usb_events_count` exceeded 10, **or**
- the login occurred after hours **and** `files_accessed_count` exceeded 20

This mirrors the kind of simple, static threshold rules commonly used in basic security monitoring tools.

**Results:**

| Model | Recall | Malicious Caught | False Positives |
|---|---|---|---|
| Rule-Based Baseline | 0% | 0 / 265 | 180 |
| Autoencoder | 12.1% | 32 / 265 | 3,360 |
| One-Class SVM | 81.5% | 216 / 265 | — |

The rule-based baseline **failed to catch a single malicious case** in the test set, despite still producing 180 false positives — meaning it would generate alert noise without providing any real detection value. Every ML model substantially outperformed the naive rule-based approach. This provides a concrete, quantified justification for the machine learning approach: insider threat behavior in this dataset is too subtle and multi-dimensional to be reliably captured by simple, static thresholds on individual features. Effective detection requires models capable of learning combinations and contextual patterns across features — exactly what Isolation Forest, One-Class SVM, and the Autoencoder are designed to do.

### Combined Risk Scoring

To move beyond three separate, hard-to-compare model outputs, scores from Isolation Forest, One-Class SVM, and the Autoencoder were each rescaled to a common 0-100 range and averaged into a single **combined risk score** per user-day (`reports/combined_risk_scores.csv`). This produces one interpretable number per record — analogous to a "risk score" in real-world UEBA (User and Entity Behavior Analytics) security tools — rather than requiring an analyst to reconcile three separate model outputs manually.

### Open-Source Packaging and Live Monitoring (Phase 3)

To make the project usable beyond the original team, the repository was packaged with an MIT license, a `requirements.txt` listing all dependencies, and a single reproducible entry point (`run_pipeline.py`) that trains and saves all models with fixed random seeds — allowing anyone to reproduce bit-identical results from a fresh clone without manually running multiple notebooks in sequence.

The combined risk score was also integrated into a live-monitoring dashboard extension (`src/aakash/dashboard/live_mode.py`), which simulates real-time event scoring: incoming events are scored by all three models as they "arrive," displayed alongside the combined risk score, with a live-updating count of alerts raised and threats caught. This moves the project from a purely offline, batch-analysis tool toward a system that demonstrates how detection could operate in a live deployment setting.

## 6. Conclusion

This project built an end-to-end pipeline for insider threat detection using the CERT r4.2 dataset, from raw log ingestion through feature engineering, labeling, and multi-model anomaly detection, later extended with per-user behavioral baselining, SHAP explainability, a combined risk-scoring ensemble, a rule-based baseline comparison, open-source packaging, and a live-monitoring dashboard.

Initial results showed that ROC-AUC alone can be misleading under extreme class imbalance: One-Class SVM's lower ROC-AUC (0.62) still outperformed Isolation Forest's higher ROC-AUC (0.84) in practical recall. After introducing per-user behavioral baseline features — comparing each user's daily activity to their own historical norm rather than population-level counts — both classical models improved substantially, with One-Class SVM emerging as the strongest model overall on both ROC-AUC (0.892) and recall (81.5%). A naive rule-based baseline, built for comparison, caught 0 of 265 malicious cases, providing concrete, quantified justification for the machine learning approach over simple static thresholds.

The team also identified and corrected two distinct methodological pitfalls during development: an early data leakage issue (placeholder scores inflating a comparison chart) and a later feature-scaling mismatch that produced unreliable results when reproducing One-Class SVM's predictions outside its original training pipeline. Both were traced to their root cause and resolved rather than papered over, and the project's move to a single reproducible pipeline entry point directly addresses the second issue for future use. Future directions include further tuning of the autoencoder's thresholding approach, rigorous multi-seed statistical validation of the baselining improvement, adversarial robustness testing against evasion strategies, expanding the feature set with additional log sources, and testing generalization across other CERT dataset scenarios (e.g., r5.2, r6.2).