import pandas as pd

test_role = pd.read_csv('data/processed/test_role_features.csv')
combined_risk = pd.read_csv('reports/combined_risk_scores.csv')

print(f"Role features rows: {len(test_role)}, Combined risk rows: {len(combined_risk)}")

# ---------- Proper merge on user + day (no more row-position guessing) ----------
merged = test_role.merge(
    combined_risk[['user', 'day', 'combined_risk_score']],
    on=['user', 'day'],
    how='inner'
)
print(f"Merged rows: {len(merged)}")

def categorize(zscore, high_threshold=2.0, low_threshold=-1.0):
    if zscore > high_threshold:
        return 'HIGH'
    elif zscore < low_threshold:
        return 'LOW'
    else:
        return 'NORMAL'

merged['USB Activity'] = merged['role_usb_events_count_zscore'].apply(categorize)
merged['File Access'] = merged['role_files_accessed_count_zscore'].apply(categorize)
merged['Session Duration'] = merged['role_session_duration_mins_zscore'].apply(categorize)
merged['Email Activity'] = merged['role_email_count_zscore'].apply(categorize)

summary = merged[['user', 'day', 'role', 'USB Activity', 'File Access',
                   'Session Duration', 'Email Activity', 'combined_risk_score', 'is_malicious']]
summary = summary.sort_values('combined_risk_score', ascending=False)

print("\n===== TOP 15 HIGHEST-RISK EMPLOYEE-DAYS =====")
print(summary.head(15).to_string(index=False))

# ---------- Verify CEJ0109 case specifically ----------
check = summary[(summary['user'] == 'CEJ0109') & (summary['day'] == '2011-03-22')]
print("\n===== CEJ0109 2011-03-22 verification (should be 0.7378) =====")
print(check.to_string(index=False))

summary.to_csv('reports/employee_risk_summary.csv', index=False)
print("\nSaved to reports/employee_risk_summary.csv")