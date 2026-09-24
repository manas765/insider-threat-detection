import pandas as pd
import numpy as np
import joblib

# ---------- Load real test data + labels ----------
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

# ---------- Identify real malicious cases ----------
malicious_idx = y_test[y_test == 1].index
print(f"Found {len(malicious_idx)} malicious cases in test set")

# ---------- Evasion Strategy 1: Smooth spike features ----------
def evade_by_smoothing(row, spread_factor=4):
    """
    Simulates an insider spreading their activity across more days
    instead of concentrating it in one spike, to stay under
    per-user rolling z-score thresholds.
    """
    row = row.copy()
    spike_features = ['usb_events_count', 'files_accessed_count', 'email_count']
    for feat in spike_features:
        row[feat] = row[feat] / spread_factor  # reduce single-day intensity
    # Recompute z-scores as if this smoothed value were normal (roughly 0)
    zscore_features = ['usb_events_count_zscore', 'files_accessed_count_zscore', 'email_count_zscore']
    for feat in zscore_features:
        row[feat] = row[feat] / spread_factor
    return row

X_test_evaded = X_test.copy()

# Convert relevant columns to float first, so they can hold decimal values
cols_to_convert = ['usb_events_count', 'files_accessed_count', 'email_count',
                    'usb_events_count_zscore', 'files_accessed_count_zscore', 'email_count_zscore']
X_test_evaded[cols_to_convert] = X_test_evaded[cols_to_convert].astype(float)
X_test_evaded.loc[malicious_idx] = X_test_evaded.loc[malicious_idx].apply(
    lambda row: evade_by_smoothing(row), axis=1
)

print("\nOriginal malicious case example:")
print(X_test.loc[malicious_idx[0]])
print("\nEvaded version of same case:")
print(X_test_evaded.loc[malicious_idx[0]])

# ---------- Save for re-evaluation ----------
X_test_evaded.to_csv('data/processed/X_test_evaded_strategy1.csv', index=False)
print("\nSaved evaded test set to data/processed/X_test_evaded_strategy1.csv")

# ---------- Load saved models AND the OC-SVM scaler ----------
iso_forest = joblib.load('models/isolation_forest.pkl')
oc_svm = joblib.load('models/oc_svm.pkl')
oc_svm_scaler = joblib.load('models/oc_svm_scaler.pkl')

# ---------- Score ORIGINAL malicious cases ----------
X_malicious_original = X_test.loc[malicious_idx]

# Isolation Forest — no scaling needed
iso_scores_before = iso_forest.predict(X_malicious_original)

# OC-SVM — MUST scale using the same scaler fit during training
X_malicious_original_scaled = oc_svm_scaler.transform(X_malicious_original)
svm_scores_before = oc_svm.predict(X_malicious_original_scaled)

caught_before_iso = (iso_scores_before == -1).sum()
caught_before_svm = (svm_scores_before == -1).sum()

# ---------- Score EVADED malicious cases ----------
X_malicious_evaded = X_test_evaded.loc[malicious_idx]

iso_scores_after = iso_forest.predict(X_malicious_evaded)

X_malicious_evaded_scaled = oc_svm_scaler.transform(X_malicious_evaded)
svm_scores_after = oc_svm.predict(X_malicious_evaded_scaled)

caught_after_iso = (iso_scores_after == -1).sum()
caught_after_svm = (svm_scores_after == -1).sum()

# ---------- Report results ----------
total = len(malicious_idx)
print("\n===== Adversarial Evasion Results (Strategy 1: Smoothing) =====")
print(f"Total malicious cases tested: {total}")
print(f"\nIsolation Forest:")
print(f"  Caught BEFORE evasion: {caught_before_iso} / {total}")
print(f"  Caught AFTER evasion:  {caught_after_iso} / {total}")
print(f"\nOne-Class SVM:")
print(f"  Caught BEFORE evasion: {caught_before_svm} / {total}")
print(f"  Caught AFTER evasion:  {caught_after_svm} / {total}")

# ---------- Sanity check: overall flag rate on full test set ----------
X_test_scaled = oc_svm_scaler.transform(X_test)
all_preds = oc_svm.predict(X_test_scaled)
flagged_count = (all_preds == -1).sum()
print(f"\nSanity check — OC-SVM flags {flagged_count} / {len(X_test)} total rows as anomalous")