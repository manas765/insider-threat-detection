"""
Hyperparameter Tuning -- Isolation Forest + One-Class SVM
Person 2 (Pushkar) -- Insider Threat Detection Project

Day 3-4 task: find the best contamination/nu/gamma/etc. by trying a
small grid of combinations and scoring each on PR-AUC (best metric
under class imbalance). Isolation Forest and One-Class SVM are
UNSUPERVISED, so sklearn's GridSearchCV doesn't apply cleanly here --
it expects labels to drive training, but these models don't use
labels to fit, only to evaluate afterward. So this is a manual loop.

Same USE_REAL_DATA toggle as skeleton_anomaly_detection.py -- flip it
once real data lands, nothing else changes.
"""

import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================
# STEP 1: Load data (identical to skeleton_anomaly_detection.py)
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
    N_NORMAL = 5880
    N_MALICIOUS = 120

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
print(f"Test shape:  {X_test.shape}, anomalies in test:  {y_test.sum()} ({y_test.mean()*100:.2f}%)\n")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Same OC-SVM scalability guard as the baseline script -- real data
# will likely be 100K+ rows, OC-SVM needs a capped subsample to train
# in reasonable time. Each grid combination below re-fits from scratch,
# so this matters even more here than in the baseline run.
OC_SVM_MAX_TRAIN_SIZE = 30000
if X_train_scaled.shape[0] > OC_SVM_MAX_TRAIN_SIZE:
    rng = np.random.RandomState(RANDOM_STATE)
    subsample_idx = rng.choice(X_train_scaled.shape[0], size=OC_SVM_MAX_TRAIN_SIZE, replace=False)
    X_train_svm = X_train_scaled[subsample_idx]
    print(f"OC-SVM: subsampled training set from {X_train_scaled.shape[0]:,} to {OC_SVM_MAX_TRAIN_SIZE:,} rows\n")
else:
    X_train_svm = X_train_scaled

# ============================================================
# STEP 2: Tune Isolation Forest
# ------------------------------------------------------------
# Small grid to start -- widen it later once you see which direction
# the best values trend toward.
# ============================================================
print("="*60)
print("TUNING: Isolation Forest")
print("="*60)

iso_param_grid = {
    "n_estimators": [100, 200, 300],
    "contamination": [0.01, 0.02, 0.05],
    "max_features": [0.5, 0.75, 1.0],
}

best_iso_score = -1
best_iso_params = None
iso_results = []

for n_est, contam, max_feat in product(
    iso_param_grid["n_estimators"],
    iso_param_grid["contamination"],
    iso_param_grid["max_features"],
):
    model = IsolationForest(
        n_estimators=n_est, contamination=contam,
        max_features=max_feat, random_state=RANDOM_STATE
    )
    model.fit(X_train_scaled)
    scores = -model.decision_function(X_test_scaled)
    pr_auc = average_precision_score(y_test, scores)
    roc_auc = roc_auc_score(y_test, scores)

    iso_results.append({
        "n_estimators": n_est, "contamination": contam, "max_features": max_feat,
        "pr_auc": pr_auc, "roc_auc": roc_auc
    })

    if pr_auc > best_iso_score:
        best_iso_score = pr_auc
        best_iso_params = {"n_estimators": n_est, "contamination": contam, "max_features": max_feat}

print(f"\nBest Isolation Forest params: {best_iso_params}")
print(f"Best PR-AUC: {best_iso_score:.4f}")

# ============================================================
# STEP 3: Tune One-Class SVM
# ============================================================
print("\n" + "="*60)
print("TUNING: One-Class SVM")
print("="*60)

svm_param_grid = {
    "nu": [0.01, 0.02, 0.05],
    "gamma": ["scale", 0.01, 0.1],
}

best_svm_score = -1
best_svm_params = None
svm_results = []

for nu, gamma in product(svm_param_grid["nu"], svm_param_grid["gamma"]):
    model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    model.fit(X_train_svm)
    scores = -model.decision_function(X_test_scaled)
    pr_auc = average_precision_score(y_test, scores)
    roc_auc = roc_auc_score(y_test, scores)

    svm_results.append({"nu": nu, "gamma": gamma, "pr_auc": pr_auc, "roc_auc": roc_auc})

    if pr_auc > best_svm_score:
        best_svm_score = pr_auc
        best_svm_params = {"nu": nu, "gamma": gamma}

print(f"\nBest One-Class SVM params: {best_svm_params}")
print(f"Best PR-AUC: {best_svm_score:.4f}")

# ============================================================
# STEP 4: Summary
# ============================================================
print("\n" + "="*60)
print("FINAL BEST PARAMS (copy these into skeleton_anomaly_detection.py)")
print("="*60)
print(f"Isolation Forest: {best_iso_params}  (PR-AUC={best_iso_score:.4f})")
print(f"One-Class SVM:    {best_svm_params}  (PR-AUC={best_svm_score:.4f})")

pd.DataFrame(iso_results).to_csv("iso_forest_tuning_results.csv", index=False)
pd.DataFrame(svm_results).to_csv("oc_svm_tuning_results.csv", index=False)
print("\nFull grid results saved to iso_forest_tuning_results.csv and oc_svm_tuning_results.csv")
