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
