"""
Hyperparameter Tuning -- Validation Split Version
Person 2 (Pushkar) -- Insider Threat Detection Project

Manas approved this fix after I flagged a concern: repeatedly tuning
against the same X_test/y_test (rounds 1-3) risked a mild version of
the leakage issue he found earlier -- the more times you search
against a fixed set, the more the "best" result gets optimistically
biased, even without anyone doing anything wrong on purpose.

FIX: carve a validation split OUT OF THE TRAINING DATA (X_train.csv +
y_train.csv, which still has both classes before benign-filtering).
All tuning below is scored against this validation split.
X_test/y_test (from evaluate.py) is NOT touched anywhere in this
script -- it stays completely clean for ONE final evaluation later,
in skeleton_anomaly_detection.py.

Search ranges below reuse the promising directions found in rounds
1-3 (larger max_samples for Isolation Forest, higher gamma for
OC-SVM) as a reasonable starting point -- that's not leakage, since
picking a sensible range from general findings differs from picking
the exact winning number by repeatedly checking the test set. The
actual SELECTION here is validated fresh against X_val, which rounds
1-3 never touched.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "login_hour", "after_hours_flag", "session_duration_mins",
    "usb_events_count", "files_accessed_count", "email_count",
    "unique_domains_visited", "email_ext_recipient_count",
]

# ============================================================
# STEP 1: Build the validation split from TRAINING data only
# ------------------------------------------------------------
# X_test/y_test is never loaded in this script -- that's the point.
# ============================================================
X_train_full = pd.read_csv("data/processed/X_train_raw.csv")[FEATURE_COLUMNS]
y_train_full = pd.read_csv("data/processed/y_train_raw.csv").squeeze()

X_fit, X_val, y_fit, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15,
    stratify=y_train_full, random_state=RANDOM_STATE
)

X_fit_benign = X_fit[y_fit == 0]  # unsupervised models still fit on benign-only rows

print(f"Fit (benign only): {X_fit_benign.shape}")
print(f"Validation: {X_val.shape}, malicious in validation: {int(y_val.sum())} ({y_val.mean()*100:.2f}%)\n")

scaler = StandardScaler()
X_fit_scaled = scaler.fit_transform(X_fit_benign)
X_val_scaled = scaler.transform(X_val)

OC_SVM_MAX_TRAIN_SIZE = 30000
rng = np.random.RandomState(RANDOM_STATE)
if X_fit_scaled.shape[0] > OC_SVM_MAX_TRAIN_SIZE:
    idx = rng.choice(X_fit_scaled.shape[0], size=OC_SVM_MAX_TRAIN_SIZE, replace=False)
    X_fit_svm = X_fit_scaled[idx]
else:
    X_fit_svm = X_fit_scaled

# ============================================================
# STEP 2: Tune Isolation Forest against X_val
# ============================================================
print("="*60)
print("TUNING (on validation set): Isolation Forest")
print("="*60)

max_samples_grid = [8192, 16384, 32768, 65536]

best_iso_prauc = -1
best_iso_params = None

for max_samp in max_samples_grid:
    model = IsolationForest(
        n_estimators=300, max_features=0.5, max_samples=max_samp,
        contamination=0.005, random_state=RANDOM_STATE
    )
    model.fit(X_fit_scaled)

    scores = -model.score_samples(X_val_scaled)
    roc_auc = roc_auc_score(y_val, scores)
    pr_auc = average_precision_score(y_val, scores)

    print(f"max_samples={max_samp:<7} PR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}")

    if pr_auc > best_iso_prauc:
        best_iso_prauc = pr_auc
        best_iso_params = {"n_estimators": 300, "max_features": 0.5, "max_samples": max_samp, "contamination": 0.005}

print(f"\nBest Isolation Forest (on validation): {best_iso_params}  PR-AUC={best_iso_prauc:.4f}")

# ============================================================
# STEP 3: Tune One-Class SVM against X_val
# ============================================================
print("\n" + "="*60)
print("TUNING (on validation set): One-Class SVM")
print("="*60)

gamma_grid = [0.2, 0.3, 0.5, 0.8]

best_svm_prauc = -1
best_svm_params = None

for gamma in gamma_grid:
    model = OneClassSVM(kernel="rbf", nu=0.005, gamma=gamma)
    model.fit(X_fit_svm)

    scores = -model.decision_function(X_val_scaled)
    roc_auc = roc_auc_score(y_val, scores)
    pr_auc = average_precision_score(y_val, scores)

    print(f"gamma={gamma:<6} PR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}")

    if pr_auc > best_svm_prauc:
        best_svm_prauc = pr_auc
        best_svm_params = {"nu": 0.005, "gamma": gamma}

print(f"\nBest One-Class SVM (on validation): {best_svm_params}  PR-AUC={best_svm_prauc:.4f}")

print("\n" + "="*60)
print("FINAL CHOSEN PARAMS (share these back so skeleton_anomaly_detection.py can be finalized)")
print("="*60)
print(f"Isolation Forest: {best_iso_params}")
print(f"One-Class SVM:    {best_svm_params}")
print("\nX_test/y_test was NEVER touched in this script -- clean for the one final evaluation.")
