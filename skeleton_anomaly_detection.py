"""
Isolation Forest + One-Class SVM -- Insider Threat Detection Project
Person 2 (Pushkar)

REAL DATA VERSION -- the fake-data/dummy-data era of this script is over.
Manas's real dataset has landed and is confirmed usable (263,117 benign
rows in X_train_benign.csv).

FINAL locked feature set (8 columns). Note: X_train_benign.csv currently
still contains a 9th column, file_copy_to_removable -- Manas investigated
this feature and found it's broken for this dataset (USB activity is too
dense/constant to meaningfully signal file copying) and recommended
dropping it. The CSV export just hasn't been updated to remove it yet,
so we explicitly select only the 8 confirmed-good columns below rather
than trusting the file to already be clean.

TRAINING METHODOLOGY (per notebooks/evaluate.py, and to avoid repeating
the 100% AUC bug the team found):
  - Train ONLY on X_train_benign.csv (already filtered to normal rows).
  - Predict using each model's OWN built-in .predict() (the
    contamination/nu threshold set at train time) -- NOT a
    best-possible-F1 threshold searched over the test set. Searching
    for the "best" threshold using test labels is a subtle form of
    leakage/optimism bias, and is very likely part of why the earlier
    comparison chart showed suspicious 100% scores.
  - Evaluate through the SHARED notebooks/evaluate.py evaluate_model()
    function, using its own X_test/y_test, so results stay comparable
    with Aakash's and consistent with the team's methodology.
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

print(f"Train (benign only): {X_train_benign.shape}")
print(f"Test: {X_test_features.shape}, malicious in test: {int(y_test.sum())} ({y_test.mean()*100:.2f}%)")

# ============================================================
# STEP 2: Scale features
# ------------------------------------------------------------
# evaluate.py's example doesn't scale, but OC-SVM (RBF kernel) is
# sensitive to feature scale, so we scale anyway -- fit on train,
# apply to test, standard practice.
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_benign)
X_test_scaled = scaler.transform(X_test_features)

# ============================================================
# STEP 3: OC-SVM scalability guard
# ------------------------------------------------------------
# 263K rows is way past the ~50k practical limit for OC-SVM training.
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
# STEP 4: Train + evaluate Isolation Forest
# ============================================================
iso_model = IsolationForest(n_estimators=200, contamination=0.02, random_state=RANDOM_STATE)
iso_model.fit(X_train_scaled)

raw_preds = iso_model.predict(X_test_scaled)           # -1 = anomaly, 1 = normal
y_pred_iso = [1 if p == -1 else 0 for p in raw_preds]   # convert to match y_test (0/1)
scores_iso = -iso_model.score_samples(X_test_scaled)    # higher = more anomalous

iso_results = evaluate_model(y_test, y_pred_iso, scores_iso, model_name="Isolation Forest")

# ============================================================
# STEP 5: Train + evaluate One-Class SVM
# ============================================================
ocsvm_model = OneClassSVM(kernel="rbf", nu=0.02, gamma="scale")
ocsvm_model.fit(X_train_svm)

raw_preds = ocsvm_model.predict(X_test_scaled)
y_pred_svm = [1 if p == -1 else 0 for p in raw_preds]
scores_svm = -ocsvm_model.decision_function(X_test_scaled)

svm_results = evaluate_model(y_test, y_pred_svm, scores_svm, model_name="OC-SVM")

# ============================================================
# STEP 6: Export scores for Aakash's comparison notebook
# ============================================================
results_df = pd.DataFrame({
    "true_label": np.asarray(y_test),
    "iso_forest_score": scores_iso,
    "oc_svm_score": scores_svm,
})
results_df.to_csv("model_scores_for_comparison.csv", index=False)
print(f"\nSaved scores to model_scores_for_comparison.csv ({len(results_df)} rows)")

print("\nDone.")
