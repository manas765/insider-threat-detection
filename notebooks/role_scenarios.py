import pandas as pd
import numpy as np
import joblib

# ---------- Load real role-based features (from Pushkar) ----------
test_role = pd.read_csv('data/processed/test_role_features.csv')

print("Columns available:", test_role.columns.tolist())
print(f"\nTotal rows: {len(test_role)}")

# ---------- Load models ----------
iso_forest = joblib.load('models/isolation_forest.pkl')
oc_svm = joblib.load('models/oc_svm.pkl')
oc_svm_scaler = joblib.load('models/oc_svm_scaler.pkl')

# =========================================================
# SCENARIO A: Legitimate role change — find a benign case
# where the OLD z-score would spike but role-aware z-score
# stays near 0 (i.e., normal for their role)
# =========================================================
print("\n===== SCENARIO A: Legitimate Role Change =====")

benign = test_role[test_role['is_malicious'] == 0].copy()

# Find a case where the PERSONAL files_accessed spike is high,
# but the ROLE-aware z-score is low (near 0) — exactly the
# "false alarm that role-awareness fixes" case
candidate = benign[
    (benign['files_accessed_count_zscore'] > 2.0) &
    (benign['role_files_accessed_count_zscore'].abs() < 0.5)
]

if len(candidate) > 0:
    case_a = candidate.iloc[0]
    print(f"\nUser: {case_a['user']}, Day: {case_a['day']}, Role: {case_a['role']}")
    print(f"Personal files_accessed z-score: {case_a['files_accessed_count_zscore']:.2f}  (looks like a spike vs OLD history)")
    print(f"Role-aware files_accessed z-score: {case_a['role_files_accessed_count_zscore']:.2f}  (normal for this role)")

    row_df = test_role.drop(columns=['user', 'day', 'role', 'is_malicious']).loc[[case_a.name]]
    # Note: adjust dropped columns to match whatever feature_cols the models expect
else:
    print("No clean example found for this exact pattern — try loosening thresholds.")

# =========================================================
# SCENARIO B: Malicious behavior after role change — should
# STILL be flagged even with role-awareness applied
# =========================================================
print("\n\n===== SCENARIO B: Malicious Behavior After Role Change =====")

malicious = test_role[test_role['is_malicious'] == 1].copy()
case_b = malicious.sort_values('role_usb_events_count_zscore', ascending=False).iloc[0]

print(f"\nUser: {case_b['user']}, Day: {case_b['day']}, Role: {case_b['role']}")
print(f"Role-aware USB z-score: {case_b['role_usb_events_count_zscore']:.2f}  (abnormal even for this role)")
print(f"Role-aware file access z-score: {case_b['role_files_accessed_count_zscore']:.2f}")