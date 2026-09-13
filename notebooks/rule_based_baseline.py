import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# ---------- Load test data ----------
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

# ---------- Simple rule-based detector ----------
# Flag as malicious if:
# - USB events are unusually high, OR
# - after-hours login AND high file access
def rule_based_flag(row):
    if row['usb_events_count'] > 10:
        return 1
    if row['after_hours_flag'] == 1 and row['files_accessed_count'] > 20:
        return 1
    return 0

y_pred_rules = X_test.apply(rule_based_flag, axis=1)

# ---------- Evaluate, same way as the ML models ----------
precision = precision_score(y_test, y_pred_rules, zero_division=0)
recall = recall_score(y_test, y_pred_rules, zero_division=0)
f1 = f1_score(y_test, y_pred_rules, zero_division=0)

print("===== Rule-Based Baseline =====")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

cm = confusion_matrix(y_test, y_pred_rules)
print("\nConfusion Matrix:")
print(cm)

malicious_caught = cm[1][1]
total_malicious = cm[1][0] + cm[1][1]
false_positives = cm[0][1]

print(f"\nMalicious caught: {malicious_caught} / {total_malicious}")
print(f"False positives: {false_positives}")

# ---------- Save results ----------
results = pd.DataFrame([{
    'model': 'Rule-Based Baseline',
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'malicious_caught': malicious_caught,
    'total_malicious': total_malicious,
    'false_positives': false_positives
}])
results.to_csv('reports/rule_based_baseline_results.csv', index=False)
print("\nSaved to reports/rule_based_baseline_results.csv")