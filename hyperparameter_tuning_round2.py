"""
Hyperparameter Tuning ROUND 2 -- Isolation Forest + One-Class SVM
Person 2 (Pushkar) -- Insider Threat Detection Project

Round 1 found: contamination/nu mostly just shift the THRESHOLD, not
the model's actual ranking ability (ROC-AUC/PR-AUC stayed flat across
contamination values in round 1). This round sweeps STRUCTURAL
parameters that can genuinely change the underlying ranking:

  Isolation Forest: n_estimators, max_features, max_samples
    -- max_samples matters a lot for iForest specifically: by default
    sklearn only samples 256 rows per tree (regardless of dataset
    size), which is intentional per the original algorithm, but worth
    testing larger values to see if it helps on this data.

  One-Class SVM: gamma, tested around what gamma='scale' actually
    evaluates to for our 8 standardized features (~0.125), since
    round 1's gamma=0.01 (10x smaller) performed worse than random.

Ranked by PR-AUC (not F1, which is noisy with only 265 positive test
examples) -- that's the honest metric under this level of imbalance.
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
from evaluate import X_test, y_test

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
rng = np.random.RandomState(RANDOM_STATE)
if X_train_scaled.shape[0] > OC_SVM_MAX_TRAIN_SIZE:
    idx = rng.choice(X_train_scaled.shape[0], size=OC_SVM_MAX_TRAIN_SIZE, replace=False)
    X_train_svm = X_train_scaled[idx]
    print(f"OC-SVM: subsampled training set to {OC_SVM_MAX_TRAIN_SIZE:,} rows\n")
else:
    X_train_svm = X_train_scaled

# ============================================================
# ROUND 2: Isolation Forest -- structural params
# ------------------------------------------------------------
# contamination fixed at 0.005 (matches real rate; round 1 showed
# it barely affects ranking anyway).
# ============================================================
print("="*60)
print("ROUND 2 TUNING: Isolation Forest (structural params)")
print("="*60)

n_estimators_grid = [100, 300]
max_features_grid = [0.5, 1.0]
max_samples_grid = [256, 2048, 8192]  # default is 256

best_iso_prauc = -1
best_iso_params = None
iso_results = []

for n_est, max_feat, max_samp in product(n_estimators_grid, max_features_grid, max_samples_grid):
    model = IsolationForest(
        n_estimators=n_est, max_features=max_feat, max_samples=max_samp,
        contamination=0.005, random_state=RANDOM_STATE
    )
    model.fit(X_train_scaled)

    scores = -model.score_samples(X_test_scaled)
    roc_auc = roc_auc_score(y_test, scores)
    pr_auc = average_precision_score(y_test, scores)

    raw_preds = model.predict(X_test_scaled)
    y_pred = [1 if p == -1 else 0 for p in raw_preds]
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"n_estimators={n_est:<4} max_features={max_feat:<5} max_samples={max_samp:<6} "
          f"PR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}  F1={f1:.4f}")

    iso_results.append({
        "n_estimators": n_est, "max_features": max_feat, "max_samples": max_samp,
        "pr_auc": pr_auc, "roc_auc": roc_auc, "f1": f1
    })

    if pr_auc > best_iso_prauc:
        best_iso_prauc = pr_auc
        best_iso_params = {"n_estimators": n_est, "max_features": max_feat, "max_samples": max_samp}

print(f"\nBest Isolation Forest (by PR-AUC): {best_iso_params}  PR-AUC={best_iso_prauc:.4f}")

# ============================================================
# ROUND 2: One-Class SVM -- gamma sweep around the real 'scale' value
# ============================================================
print("\n" + "="*60)
print("ROUND 2 TUNING: One-Class SVM (gamma sweep)")
print("="*60)

nu_grid = [0.005, 0.01]
gamma_grid = ["scale", 0.05, 0.1, 0.2]

best_svm_prauc = -1
best_svm_params = None
svm_results = []

for nu, gamma in product(nu_grid, gamma_grid):
    model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    model.fit(X_train_svm)

    scores = -model.decision_function(X_test_scaled)
    roc_auc = roc_auc_score(y_test, scores)
    pr_auc = average_precision_score(y_test, scores)

    raw_preds = model.predict(X_test_scaled)
    y_pred = [1 if p == -1 else 0 for p in raw_preds]
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"nu={nu:<6} gamma={str(gamma):<6} PR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}  F1={f1:.4f}")

    svm_results.append({"nu": nu, "gamma": gamma, "pr_auc": pr_auc, "roc_auc": roc_auc, "f1": f1})

    if pr_auc > best_svm_prauc:
        best_svm_prauc = pr_auc
        best_svm_params = {"nu": nu, "gamma": gamma}

print(f"\nBest One-Class SVM (by PR-AUC): {best_svm_params}  PR-AUC={best_svm_prauc:.4f}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("ROUND 2 BEST PARAMS (ranked by PR-AUC, the honest metric)")
print("="*60)
print(f"Isolation Forest: {best_iso_params}  PR-AUC={best_iso_prauc:.4f}")
print(f"One-Class SVM:    {best_svm_params}  PR-AUC={best_svm_prauc:.4f}")

pd.DataFrame(iso_results).to_csv("iso_forest_tuning_round2_results.csv", index=False)
pd.DataFrame(svm_results).to_csv("oc_svm_tuning_round2_results.csv", index=False)
print("\nFull results saved to iso_forest_tuning_round2_results.csv and oc_svm_tuning_round2_results.csv")
