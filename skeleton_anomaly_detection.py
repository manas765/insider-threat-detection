"""
Isolation Forest + One-Class SVM -- Insider Threat Detection Project
Person 2 (Pushkar) -- FINAL VERSION

Hyperparameters were tuned against a validation split carved out of
X_train_raw.csv/y_train_raw.csv (per Manas's approved fix, to avoid
repeatedly leaking information from X_test/y_test during search).
X_test/y_test is touched EXACTLY ONCE below, for this final,
reportable evaluation -- consistent with the team's methodology and
notebooks/evaluate.py.

FINAL FEATURE SET (8 columns, confirmed by Manas):
  login_hour, after_hours_flag, session_duration_mins, usb_events_count,
  files_accessed_count, email_count, unique_domains_visited,
  email_ext_recipient_count
(file_copy_to_removable is present in the raw CSVs but excluded --
Manas found it unreliable for this dataset and dropped it.)

FINAL TUNED HYPERPARAMETERS (chosen via clean validation-split search):
  Isolation Forest: n_estimators=300, max_features=0.5,
                     max_samples=65536, contamination=0.005
  One-Class SVM:     nu=0.005, gamma=0.8

Trained on the FULL X_train_benign.csv (not the reduced validation-
split portion) since no validation set needs holding back anymore --
maximizes training data for the final model.

Results exported to reports/ for Aakash's comparison.ipynb.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

sys.path.append("notebooks")
from evaluate import evaluate_model, X_test, y_test  # shared test set + evaluator

RANDOM_STATE = 42

# ============================================================
# STEP 1: Load data
# ============================================================
FEATURE_COLUMNS = [
    "login_hour", "after_hours_flag", "session_duration_mins",
    "usb_events_count", "files_accessed_count", "email_count",
    "unique_domains_visited", "email_ext_recipient_count",
]

X_train_benign = pd.read_csv("data/processed/X_train_benign.csv")[FEATURE_COLUMNS]
X_test_features = X_test[FEATURE_COLUMNS]

print(f"Train (benign only, full set): {X_train_benign.shape}")
print(f"Test: {X_test_features.shape}, malicious in test: {int(y_test.sum())} ({y_test.mean()*100:.2f}%)")

# ============================================================
# STEP 2: Scale features
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_benign)
X_test_scaled = scaler.transform(X_test_features)

# ============================================================
# STEP 3: OC-SVM scalability guard
# ============================================================
OC_SVM_MAX_TRAIN_SIZE = 30000
if X_train_scaled.shape[0] > OC_SVM_MAX_TRAIN_SIZE:
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(X_train_scaled.shape[0], size=OC_SVM_MAX_TRAIN_SIZE, replace=False)
    X_train_svm = X_train_scaled[idx]
    print(f"OC-SVM: subsampled training set from {X_train_scaled.shape[0]:,} to {OC_SVM_MAX_TRAIN_SIZE:,} rows")
else:
    X_train_svm = X_train_scaled

# ============================================================
# STEP 4: Train + evaluate Isolation Forest (FINAL tuned params)
# ============================================================
iso_model = IsolationForest(
    n_estimators=300, max_features=0.5, max_samples=65536,
    contamination=0.005, random_state=RANDOM_STATE
)
iso_model.fit(X_train_scaled)

raw_preds = iso_model.predict(X_test_scaled)
y_pred_iso = [1 if p == -1 else 0 for p in raw_preds]
scores_iso = -iso_model.score_samples(X_test_scaled)

iso_results = evaluate_model(y_test, y_pred_iso, scores_iso, model_name="Isolation Forest")

# ============================================================
# STEP 5: Train + evaluate One-Class SVM (FINAL tuned params)
# ============================================================
ocsvm_model = OneClassSVM(kernel="rbf", nu=0.005, gamma=0.8)
ocsvm_model.fit(X_train_svm)

raw_preds = ocsvm_model.predict(X_test_scaled)
y_pred_svm = [1 if p == -1 else 0 for p in raw_preds]
scores_svm = -ocsvm_model.decision_function(X_test_scaled)

svm_results = evaluate_model(y_test, y_pred_svm, scores_svm, model_name="OC-SVM")

# ============================================================
# STEP 6: Export scores to reports/ for Aakash's comparison notebook
# ============================================================
results_df = pd.DataFrame({
    "true_label": np.asarray(y_test),
    "iso_forest_score": scores_iso,
    "oc_svm_score": scores_svm,
})
results_df.to_csv("reports/model_scores_for_comparison.csv", index=False)
print(f"\nSaved FINAL scores to reports/model_scores_for_comparison.csv ({len(results_df)} rows)")

print("\nDone. This is the final, validated run -- X_test/y_test touched exactly once.")
