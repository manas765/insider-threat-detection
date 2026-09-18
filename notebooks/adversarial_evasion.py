import pandas as pd
import numpy as np

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