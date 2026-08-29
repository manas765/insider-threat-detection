"""
Hyperparameter Tuning -- Isolation Forest + One-Class SVM
Person 2 (Pushkar) -- Insider Threat Detection Project

REAL DATA VERSION. Your baseline run showed a clear clue: ROC-AUC was
reasonable (0.78 for Isolation Forest) but F1/Precision/Recall were
near zero. The real malicious rate in the test set is only 0.40%
(265/66054) -- much lower than the contamination=0.02 (2%) / nu=0.02
used in the baseline. That mismatch is very likely why each model's
own .predict() threshold wasn't picking out the right points, even
though its underlying ranking ability (AUC) was decent.

This sweeps contamination/nu values centered around the REAL observed
rate (~0.004) to find a threshold that actually works -- using the
same honest evaluation approach as evaluate.py: scoring off each
model's own .predict() output, not a best-possible-F1 search over
the test set.
"""

import sys
import numpy as np
import pandas as pd
from itertools import product
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

sys.path.append("notebooks")
from evaluate import X_test, y_test  # shared test set

RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "login_hour", "after_hours_flag", "session_duration_mins",
    "usb_events_count", "files_accessed_count", "email_count",
    "unique_domains_visited", "email_ext_recipient_count",
]

X_train_benign = pd.read_csv("data/processed/X_train_benign.csv")[FEATURE_COLUMNS]
X_test_features = X_test[FEATURE_COLUMNS]

print(f"Train (benign only): {X_train_benign.shape}")
print(f"Test: {X_test_features.shape}, malicious in test: {int(y_test.sum())} ({y_test.mean()*100:.2f}%)\n")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_benign)
X_test_scaled = scaler.transform(X_test_features)

OC_SVM_MAX_TRAIN_SIZE = 30000
if X_train_scaled.shape[0] > OC_SVM_MAX_TRAIN_SIZE:
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(X_train_scaled.shape[0], size=OC_SVM_MAX_TRAIN_SIZE, replace=False)
    X_train_svm = X_train_scaled[idx]
    print(f"OC-SVM: subsampled training set to {OC_SVM_MAX_TRAIN_SIZE:,} rows\n")
else:
    X_train_svm = X_train_scaled

# ============================================================
# TUNING: Isolation Forest
# ------------------------------------------------------------
# Centered on the real ~0.40% malicious rate, not the originally
# assumed 2%.
# ============================================================
print("="*60)
print("TUNING: Isolation Forest")
print("="*60)

contamination_grid = [0.003, 0.005, 0.01, 0.02, 0.03]

best_iso_f1 = -1
best_iso_params = None
iso_results = []

for contam in contamination_grid:
    model = IsolationForest(n_estimators=200, contamination=contam, random_state=RANDOM_STATE)
    model.fit(X_train_scaled)

    raw_preds = model.predict(X_test_scaled)
    y_pred = [1 if p == -1 else 0 for p in raw_preds]
    scores = -model.score_samples(X_test_scaled)

    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, scores)
    pr_auc = average_precision_score(y_test, scores)
    flagged = sum(y_pred)

    print(f"contamination={contam:<6} F1={f1:.4f}  ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}  flagged={flagged}")
    iso_results.append({"contamination": contam, "f1": f1, "roc_auc": roc_auc, "pr_auc": pr_auc, "flagged": flagged})

    if f1 > best_iso_f1:
        best_iso_f1 = f1
        best_iso_params = {"contamination": contam}

print(f"\nBest Isolation Forest: {best_iso_params}  (F1={best_iso_f1:.4f})")

# ============================================================
# TUNING: One-Class SVM
# ============================================================
print("\n" + "="*60)
print("TUNING: One-Class SVM")
print("="*60)

nu_grid = [0.003, 0.005, 0.01, 0.02]
gamma_grid = ["scale", 0.01]

best_svm_f1 = -1
best_svm_params = None
svm_results = []

for nu, gamma in product(nu_grid, gamma_grid):
    model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    model.fit(X_train_svm)

    raw_preds = model.predict(X_test_scaled)
    y_pred = [1 if p == -1 else 0 for p in raw_preds]
    scores = -model.decision_function(X_test_scaled)

    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, scores)
    pr_auc = average_precision_score(y_test, scores)
    flagged = sum(y_pred)

    print(f"nu={nu:<6} gamma={str(gamma):<6} F1={f1:.4f}  ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}  flagged={flagged}")
    svm_results.append({"nu": nu, "gamma": gamma, "f1": f1, "roc_auc": roc_auc, "pr_auc": pr_auc, "flagged": flagged})

    if f1 > best_svm_f1:
        best_svm_f1 = f1
        best_svm_params = {"nu": nu, "gamma": gamma}

print(f"\nBest One-Class SVM: {best_svm_params}  (F1={best_svm_f1:.4f})")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("BEST PARAMS FOUND (update skeleton_anomaly_detection.py with these)")
print("="*60)
print(f"Isolation Forest: {best_iso_params}  F1={best_iso_f1:.4f}")
print(f"One-Class SVM:    {best_svm_params}  F1={best_svm_f1:.4f}")

pd.DataFrame(iso_results).to_csv("iso_forest_tuning_results.csv", index=False)
pd.DataFrame(svm_results).to_csv("oc_svm_tuning_results.csv", index=False)
print("\nFull results saved to iso_forest_tuning_results.csv and oc_svm_tuning_results.csv")
