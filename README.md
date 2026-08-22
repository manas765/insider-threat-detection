\# Insider Threat / Data Exfiltration Detection



A machine learning project to detect insider threats and data exfiltration from corporate network/endpoint logs, built on the CERT Insider Threat Dataset (r4.2).



\## Problem



Malicious data exfiltration by employees or compromised accounts is extremely rare compared to normal daily activity, creating an extreme class imbalance problem — similar to fraud detection. This project builds and compares multiple anomaly detection approaches on real, labeled insider-threat data.



\## Team



| Person | Responsibility |

|---|---|

| Manas | Data pipeline: parsing, feature engineering, labeling, SMOTE, train/test split |

| Pushkar | Isolation Forest + One-Class SVM models |

| Aakash | Autoencoder model + final comparison report |



\## Dataset



CERT Insider Threat Dataset r4.2 — includes logon, device (USB), file access, and email activity logs for \~1000 simulated employees, with 70 labeled malicious insider scenarios.



\## Features Used



One row per user per day, built from raw event logs:



\- `login\_hour` — hour of first login

\- `after\_hours\_flag` — 1 if any login occurred before 6am or after 6pm

\- `session\_duration\_mins` — total time logged in that day

\- `usb\_events\_count` — number of USB connect events

\- `files\_accessed\_count` — number of files accessed

\- `email\_count` — number of emails sent

\- `unique\_domains\_visited` — distinct websites visited

\- `email\_ext\_recipient\_count` — emails sent to external (non-company) domains



Label: `is\_malicious` (1 = insider threat activity, 0 = normal), sourced from ground-truth insider scenario data.



\## Methodology



1\. Parsed raw CERT logs into daily aggregated features (see `notebooks/manas\_sessions.py`)

2\. Labeled rows using ground-truth insider scenario date ranges

3\. Time-aware train/test split (80/20) to avoid temporal data leakage

4\. Applied SMOTE to the training set only, to address extreme class imbalance (\~0.4% malicious)

5\. Trained and compared: Isolation Forest, One-Class SVM, and a Dense Autoencoder

6\. Evaluated using AUC-ROC, Precision-Recall AUC, and F1-score (not raw accuracy, due to imbalance)



\## Repository Structure



\\`\\`\\`

insider-threat-detection/

├── data/processed/       # X\_train, X\_test, y\_train, y\_test (exported, ready to use)

├── notebooks/             # exploration and pipeline scripts

├── src/

│   ├── pushkar/            # Isolation Forest + OC-SVM

│   └── aakash/            # Autoencoder + comparison

├── reports/               # ROC curves, comparison plots

└── requirements.txt

\\`\\`\\`



\## Setup



\\`\\`\\`

python -m venv venv

venv\\\\Scripts\\\\activate

pip install -r requirements.txt

\\`\\`\\`



\## Results



\*(To be filled in once all three models are compared — final AUC-ROC, PR-AUC, F1 scores for each model)\*

