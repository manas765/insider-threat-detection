"""
Isolation Forest + One-Class SVM -- Insider Threat Detection Project
Person 2 (Pushkar)

FINAL locked schema (confirmed by Manas & Aakash, CERT r4.2-based):
  13 total columns:
    user_id, date, login_hour, after_hours_flag, session_duration_mins,
    files_accessed_count, bytes_transferred, usb_events_count,
    unique_domains_visited, email_count, email_ext_recipient_count,
    file_copy_to_removable, is_malicious

  10 model input features (user_id, date, is_malicious excluded):
    login_hour, after_hours_flag, session_duration_mins,
    files_accessed_count, bytes_transferred, usb_events_count,
    unique_domains_visited, email_count, email_ext_recipient_count,
    file_copy_to_removable

  Note: failed_login_count was DROPPED (logon.csv has no failed-attempt
  data) -- replaced by email_ext_recipient_count and file_copy_to_removable.

DELIVERY (confirmed by Manas): real data lands as four separate files
in data/processed/ on the main branch:
  data/processed/X_train.csv
  data/processed/X_test.csv
  data/processed/y_train.csv
  data/processed/y_test.csv
Once he pushes those: git pull on main, then flip USE_REAL_DATA to
True below. Nothing else in this script needs to change.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================
# STEP 1: Load data
# ------------------------------------------------------------
# Flip this to True once you've pulled Manas's real files from
# data/processed/ on main. That's the ONLY line you need to change.
# ============================================================
USE_REAL_DATA = False

FEATURE_COLUMNS = [
    "login_hour", "after_hours_flag", "session_duration_mins",
    "files_accessed_count", "bytes_transferred", "usb_events_count",
    "unique_domains_visited", "email_count",
    "email_ext_recipient_count", "file_copy_to_removable",
]

if USE_REAL_DATA:
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

else:
    # FAKE DATA -- shaped to match the final locked schema, with a
    # synthetic exfiltration pattern baked in so the models have
    # something real to detect while we're testing the pipeline.
    N_NORMAL = 5880
    N_MALICIOUS = 120  # ~2% anomaly rate, matching expected imbalance

    def fake_rows(n, malicious):
        if not malicious:
            return pd.DataFrame({
                "user_id": [f"EMP_{i:04d}" for i in np.random.randint(1, 201, n)],
                "date": pd.to_datetime("2026-01-01") + pd.to_timedelta(np.random.randint(0, 60, n), unit="D"),
                "login_hour": np.clip(np.random.normal(9, 1.5, n), 0, 23).astype(int),
                "session_duration_mins": np.clip(np.random.normal(480, 60, n), 30, None),
                "files_accessed_count": np.random.poisson(15, n),
                "bytes_transferred": np.random.lognormal(mean=13, sigma=0.5, size=n),
                "usb_events_count": np.random.poisson(0.2, n),
                "unique_domains_visited": np.random.poisson(8, n),
                "email_count": np.random.poisson(20, n),
                "email_ext_recipient_count": np.random.poisson(0.3, n),
                "file_copy_to_removable": np.random.poisson(0.1, n),
            })
        else:
            return pd.DataFrame({
                "user_id": [f"EMP_{i:04d}" for i in np.random.randint(1, 201, n)],
                "date": pd.to_datetime("2026-01-01") + pd.to_timedelta(np.random.randint(0, 60, n), unit="D"),
                "login_hour": np.clip(np.random.normal(22, 3, n), 0, 23).astype(int),
                "session_duration_mins": np.clip(np.random.normal(200, 100, n), 10, None),
                "files_accessed_count": np.random.poisson(60, n),
                "bytes_transferred": np.random.lognormal(mean=16, sigma=0.7, size=n),
                "usb_events_count": np.random.poisson(2.5, n),
                "unique_domains_visited": np.random.poisson(15, n),
                "email_count": np.random.poisson(10, n),
                "email_ext_recipient_count": np.random.poisson(3, n),
                "file_copy_to_removable": np.random.poisson(3.5, n),
            })

    df = pd.concat([
        fake_rows(N_NORMAL, malicious=False).assign(is_malicious=0),
        fake_rows(N_MALICIOUS, malicious=True).assign(is_malicious=1),
    ], ignore_index=True)

    df["after_hours_flag"] = ((df["login_hour"] < 6) | (df["login_hour"] > 18)).astype(int)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    X = df[FEATURE_COLUMNS]
    y = df["is_malicious"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )

print(f"Train shape: {X_train.shape}, anomalies in train: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"Test shape:  {X_test.shape}, anomalies in test:  {y_test.sum()} ({y_test.mean()*100:.2f}%)")

# ============================================================
# STEP 2: Scale features
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# STEP 3: Train Isolation Forest
# ============================================================
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    max_features=1.0,
    random_state=RANDOM_STATE
)
iso_forest.fit(X_train_scaled)
iso_scores = -iso_forest.decision_function(X_test_scaled)

# ============================================================
# STEP 4: Train One-Class SVM
# ============================================================
oc_svm = OneClassSVM(kernel="rbf", nu=0.02, gamma="scale")
oc_svm.fit(X_train_scaled)
svm_scores = -oc_svm.decision_function(X_test_scaled)

# ============================================================
# STEP 5: Evaluate both models
# ============================================================
def evaluate(y_true, scores, model_name):
    roc_auc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)

    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    best_f1 = f1s[best_idx]

    preds = (scores >= best_threshold).astype(int)
    cm = confusion_matrix(y_true, preds)

    print(f"\n--- {model_name} ---")
    print(f"ROC-AUC:  {roc_auc:.4f}")
    print(f"PR-AUC:   {pr_auc:.4f}")
    print(f"Best F1:  {best_f1:.4f}  (at threshold {best_threshold:.4f})")
    print(f"Confusion Matrix:\n{cm}")

    return {"roc_auc": roc_auc, "pr_auc": pr_auc, "f1": best_f1, "cm": cm}

iso_results = evaluate(y_test, iso_scores, "Isolation Forest")
svm_results = evaluate(y_test, svm_scores, "One-Class SVM")

# ============================================================
# STEP 6: Plot ROC curves
# ============================================================
plt.figure(figsize=(7, 6))
for name, scores in [("Isolation Forest", iso_scores), ("One-Class SVM", svm_scores)]:
    fpr, tpr, _ = roc_curve(y_test, scores)
    auc = roc_auc_score(y_test, scores)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("roc_comparison.png")
plt.show()

print("\nDone. Once real data is in place (USE_REAL_DATA = True), everything above runs unchanged.")
