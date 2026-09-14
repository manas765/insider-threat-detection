"""
Alert Fatigue / Precision-vs-Workload Analysis
Person 2 (Pushkar) -- Phase 2

Sweeps decision thresholds across each model's anomaly scores and
reports, for each threshold: how many alerts an analyst would need
to review, and what fraction of real threats get caught at that
volume. Reframes the precision/recall trade-off in terms analysts
and managers actually think in -- workload vs. coverage -- rather
than abstract metric names.

Loads the ALREADY-TRAINED models saved by skeleton_anomaly_detection.py
-- run that script first if models/ doesn't exist yet.
"""

import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

sys.path.append("notebooks")
from evaluate import X_test, y_test

FEATURE_COLUMNS = [
    "login_hour",
    "after_hours_flag",
    "session_duration_mins",
    "usb_events_count",
    "files_accessed_count",
    "email_count",
    "unique_domains_visited",
    "email_ext_recipient_count",
    "days_since_last_spike",
    "email_count_zscore",
    "files_accessed_count_zscore",
    "session_duration_mins_zscore",
    "usb_events_count_zscore"
]

scaler = joblib.load("models/scaler.pkl")
iso_model = joblib.load("models/iso_forest.pkl")
ocsvm_model = joblib.load("models/ocsvm.pkl")

FEATURE_COLUMNS = list(scaler.feature_names_in_)

missing_features = [col for col in FEATURE_COLUMNS if col not in X_test.columns]

if missing_features:
    raise ValueError(
        f"Test data is missing required features: {missing_features}"
    )

X_test_features = X_test[FEATURE_COLUMNS]

X_test_scaled = scaler.transform(X_test_features)

iso_scores = -iso_model.score_samples(X_test_scaled)
svm_scores = -ocsvm_model.decision_function(X_test_scaled)

y_true = np.asarray(y_test)
total_malicious = int(y_true.sum())
print(f"Test set: {len(y_true)} rows, {total_malicious} actually malicious\n")


def workload_curve(scores, model_name):
    """For a sweep of thresholds, report alert volume vs. recall achieved."""
    thresholds = np.percentile(scores, np.arange(80, 100.5, 0.5))  # top 20% down to top 0%
    rows = []
    for t in thresholds:
        alerts = scores >= t
        n_alerts = int(alerts.sum())
        n_caught = int((alerts & (y_true == 1)).sum())
        recall = n_caught / total_malicious if total_malicious else 0
        rows.append({"threshold": t, "alerts": n_alerts, "caught": n_caught, "recall": recall})

    df = pd.DataFrame(rows).drop_duplicates(subset="alerts").sort_values("alerts")
    print(f"--- {model_name}: workload vs. recall ---")
    for _, r in df.iloc[::3].iterrows():  # print a sample, not every single row
        print(f"Review {int(r['alerts']):5d} alerts  ->  catch {r['recall']*100:5.1f}% of threats "
              f"({int(r['caught'])}/{total_malicious})")
    print()
    return df


iso_df = workload_curve(iso_scores, "Isolation Forest")
svm_df = workload_curve(svm_scores, "OC-SVM")

plt.figure(figsize=(8, 6))
plt.plot(iso_df["alerts"], iso_df["recall"] * 100, marker='o', markersize=3, label="Isolation Forest")
plt.plot(svm_df["alerts"], svm_df["recall"] * 100, marker='o', markersize=3, label="OC-SVM")
plt.xlabel("Number of Alerts an Analyst Reviews")
plt.ylabel("% of Real Threats Caught (Recall)")
plt.title("Alert Fatigue: Analyst Workload vs. Threat Coverage")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("reports/alert_fatigue_chart.png")
plt.show()

iso_df.to_csv("reports/alert_fatigue_isoforest.csv", index=False)
svm_df.to_csv("reports/alert_fatigue_ocsvm.csv", index=False)
print("Saved reports/alert_fatigue_chart.png, alert_fatigue_isoforest.csv, alert_fatigue_ocsvm.csv")
