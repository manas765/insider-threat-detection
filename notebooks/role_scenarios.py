import pandas as pd
import numpy as np
import joblib

# ---------- Load data and models ----------
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

iso_forest = joblib.load('models/isolation_forest.pkl')
oc_svm = joblib.load('models/oc_svm.pkl')
oc_svm_scaler = joblib.load('models/oc_svm_scaler.pkl')

def score_row(row):
    row_df = pd.DataFrame([row])
    iso_pred = iso_forest.predict(row_df)[0]
    svm_scaled = oc_svm_scaler.transform(row_df)
    svm_pred = oc_svm.predict(svm_scaled)[0]
    return iso_pred, svm_pred

# =========================================================
# SCENARIO A: Legitimate role change — activity shifts,
# but should NOT be flagged once role context is applied
# =========================================================
print("===== SCENARIO A: Legitimate Role Change (promotion) =====")

# Pick a normal (benign) row as our "before promotion" baseline
benign_idx = y_test[y_test == 0].index
before_promotion = X_test.loc[benign_idx[100]].copy()

print("\nBEFORE promotion (old role, normal for them):")
print(before_promotion[['files_accessed_count', 'files_accessed_count_zscore',
                          'session_duration_mins', 'session_duration_mins_zscore']])

# Simulate: after promotion, this person's role now legitimately
# involves more file access — their PERSONAL z-score spikes,
# but it's actually normal for their NEW role
after_promotion = before_promotion.copy()
after_promotion['files_accessed_count'] = after_promotion['files_accessed_count'] * 3
after_promotion['files_accessed_count_zscore'] = 2.8  # looks like a spike vs OLD personal history

print("\nAFTER promotion (personal z-score spikes, but role_files_accessed_zscore would be ~0):")
print(after_promotion[['files_accessed_count', 'files_accessed_count_zscore']])

iso_a, svm_a = score_row(after_promotion)
print(f"\nWithout role context — Isolation Forest: {'FLAGGED' if iso_a == -1 else 'not flagged'}")
print(f"Without role context — One-Class SVM:    {'FLAGGED' if svm_a == -1 else 'not flagged'}")
print("(This is exactly the false-positive problem role-awareness is meant to fix —")
print(" once Pushkar's role_files_accessed_zscore is ~0 for this new role, the combined")
print(" risk score should drop even though personal z-score alone looks high.)")

# =========================================================
# SCENARIO B: Malicious behavior AFTER a role change —
# should STILL be flagged, since it's abnormal even for
# the new role, not just abnormal for their old history
# =========================================================
print("\n\n===== SCENARIO B: Malicious Behavior After Role Change =====")

malicious_idx = y_test[y_test == 1].index
malicious_case = X_test.loc[malicious_idx[5]].copy()

print("\nMalicious case (abnormal even accounting for a new role):")
print(malicious_case[['usb_events_count', 'usb_events_count_zscore',
                       'files_accessed_count', 'files_accessed_count_zscore']])

iso_b, svm_b = score_row(malicious_case)
print(f"\nIsolation Forest: {'FLAGGED' if iso_b == -1 else 'missed'}")
print(f"One-Class SVM:    {'FLAGGED' if svm_b == -1 else 'missed'}")
print("(This should stay flagged even with role-awareness, since it's a real threat —")
print(" the point is role context reduces FALSE alarms, not real ones.)")