"""
SHAP Explainability -- Isolation Forest + One-Class SVM
Person 2 (Pushkar) -- Phase 2

For each flagged (anomalous) case, computes which features
contributed most -- turning a bare anomaly score into something an
analyst can act on ("flagged due to: usb_events_count=8 vs. typical
0.2, session_duration_mins=15 vs. typical 480").

Isolation Forest is tree-based -> shap.TreeExplainer (fast, exact).
One-Class SVM has no tree structure -> shap.KernelExplainer
(model-agnostic, much slower) -- only the TOP_N most anomalous
flagged cases are explained to keep runtime reasonable. That also
matches the real use case: a dashboard only needs to explain the
alerts an analyst actually opens, not every row in the test set.

Requires: pip install shap --break-system-packages
Loads the ALREADY-TRAINED models saved by skeleton_anomaly_detection.py.
"""

import sys
import numpy as np
import pandas as pd
import joblib
import shap

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
    "usb_events_count_zscore",
    "files_accessed_count_zscore",
    "email_count_zscore",
    "session_duration_mins_zscore",
    "days_since_last_spike",
]

TOP_N_SVM = 100  # cap on how many OC-SVM cases get explained (KernelExplainer is slow)

X_test_features = X_test[FEATURE_COLUMNS].reset_index(drop=True)

scaler = joblib.load("models/scaler.pkl")
iso_model = joblib.load("models/iso_forest.pkl")
ocsvm_model = joblib.load("models/ocsvm.pkl")

X_test_scaled = scaler.transform(X_test_features)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=FEATURE_COLUMNS)


def top_features_explanation(shap_row, feature_names, original_row, n=3):
    """Return a readable string naming the top-n features driving this score,
    with their ACTUAL (unscaled) values for context an analyst can read."""
    order = np.argsort(-np.abs(shap_row))[:n]
    parts = [f"{feature_names[i]}={original_row[feature_names[i]]:.1f}" for i in order]
    return "; ".join(parts)


# ============================================================
# Isolation Forest -- TreeExplainer
# ============================================================
raw_preds_iso = iso_model.predict(X_test_scaled)
flagged_iso_idx = np.where(raw_preds_iso == -1)[0]
print(f"Isolation Forest flagged {len(flagged_iso_idx)} cases")

print("Computing SHAP values for Isolation Forest (TreeExplainer, fast)...")
iso_explainer = shap.TreeExplainer(iso_model)
iso_shap_values = iso_explainer.shap_values(X_test_scaled_df.iloc[flagged_iso_idx])

iso_explanations = []
for pos, row_idx in enumerate(flagged_iso_idx):
    reason = top_features_explanation(iso_shap_values[pos], FEATURE_COLUMNS, X_test_features.iloc[row_idx])
    iso_explanations.append({"row_index": int(row_idx), "top_reasons": reason})

pd.DataFrame(iso_explanations).to_csv("reports/iso_forest_explanations.csv", index=False)
print(f"Saved {len(iso_explanations)} explanations to reports/iso_forest_explanations.csv\n")

# ============================================================
# One-Class SVM -- KernelExplainer (slower, so capped to TOP_N)
# ============================================================
svm_scores = -ocsvm_model.decision_function(X_test_scaled)
raw_preds_svm = ocsvm_model.predict(X_test_scaled)
flagged_svm_idx = np.where(raw_preds_svm == -1)[0]
print(f"OC-SVM flagged {len(flagged_svm_idx)} cases -- explaining top {TOP_N_SVM} by score to keep runtime reasonable")

top_order = np.argsort(-svm_scores[flagged_svm_idx])[:TOP_N_SVM]
flagged_svm_idx_sample = flagged_svm_idx[top_order]

print("Computing SHAP values for OC-SVM (KernelExplainer -- this will take a few minutes)...")
background = shap.sample(X_test_scaled_df, 100, random_state=42)
svm_explainer = shap.KernelExplainer(ocsvm_model.decision_function, background)
svm_shap_values = svm_explainer.shap_values(X_test_scaled_df.iloc[flagged_svm_idx_sample], nsamples=100)

svm_explanations = []
for pos, row_idx in enumerate(flagged_svm_idx_sample):
    reason = top_features_explanation(svm_shap_values[pos], FEATURE_COLUMNS, X_test_features.iloc[row_idx])
    svm_explanations.append({"row_index": int(row_idx), "top_reasons": reason})

pd.DataFrame(svm_explanations).to_csv("reports/ocsvm_explanations.csv", index=False)
print(f"Saved {len(svm_explanations)} explanations to reports/ocsvm_explanations.csv")

print("\nDone. These per-case explanations are what Aakash's dashboard will display.")