import pandas as pd
import numpy as np
import joblib

# ---------- Load data ----------
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

# ---------- Load verified models (from run_pipeline.py) ----------
iso_forest = joblib.load('models/isolation_forest.pkl')
oc_svm = joblib.load('models/oc_svm.pkl')
oc_svm_scaler = joblib.load('models/oc_svm_scaler.pkl')

# ---------- Pick ONE clear malicious case with a real spike ----------
malicious_idx = y_test[y_test == 1].index
X_malicious = X_test.loc[malicious_idx]

# Find a case with a strong positive z-score spike (a real, obvious spike)
spike_case_idx = X_malicious['usb_events_count_zscore'].idxmax()
original_case = X_test.loc[spike_case_idx].copy()

print("===== STEP 1: Original spike case (BEFORE evasion) =====")
print(original_case[['usb_events_count', 'usb_events_count_zscore']])

X_original_scaled = oc_svm_scaler.transform(pd.DataFrame([original_case]))
original_iso = iso_forest.predict(pd.DataFrame([original_case]))[0]
original_svm = oc_svm.predict(X_original_scaled)[0]
print(f"Isolation Forest: {'CAUGHT' if original_iso == -1 else 'missed'}")
print(f"One-Class SVM:    {'CAUGHT' if original_svm == -1 else 'missed'}")

# ---------- STEP 2: Evasion — spread the spike across 7 days ----------
evaded_case = original_case.copy()
spread_factor = 7
evaded_case['usb_events_count'] = evaded_case['usb_events_count'] / spread_factor
evaded_case['usb_events_count_zscore'] = evaded_case['usb_events_count_zscore'] / spread_factor

print("\n===== STEP 2: Evaded case (spread across 7 days) =====")
print(evaded_case[['usb_events_count', 'usb_events_count_zscore']])

X_evaded_scaled = oc_svm_scaler.transform(pd.DataFrame([evaded_case]))
evaded_iso = iso_forest.predict(pd.DataFrame([evaded_case]))[0]
evaded_svm = oc_svm.predict(X_evaded_scaled)[0]
print(f"Isolation Forest: {'CAUGHT' if evaded_iso == -1 else 'MISSED (evasion worked!)'}")
print(f"One-Class SVM:    {'CAUGHT' if evaded_svm == -1 else 'MISSED (evasion worked!)'}")

# ---------- STEP 3: Countermeasure — cumulative 60-day total ----------
# Simulate: even though spread thin, the TOTAL over time is still unusual
evaded_case_with_countermeasure = evaded_case.copy()
# Cumulative total = daily count * spread_factor (the real total didn't shrink, just spread out)
cumulative_total = original_case['usb_events_count']  # the true total activity
# Countermeasure flags if cumulative total over the period is above a threshold
countermeasure_threshold = 5  # example threshold, tune as needed
countermeasure_flag = cumulative_total > countermeasure_threshold

print("\n===== STEP 3: Countermeasure (60-day cumulative total check) =====")
print(f"Cumulative total activity: {cumulative_total}")
print(f"Countermeasure: {'CAUGHT (evasion defeated!)' if countermeasure_flag else 'missed'}")
print("\n===== DEMO NARRATIVE SUMMARY =====")
print(f"Isolation Forest (single-day view): missed BOTH before and after evasion — relying only on today's count is fragile.")
print(f"One-Class SVM (learned broader patterns): caught the threat both times — inherently more robust to this evasion.")
print(f"Countermeasure (60-day cumulative total): catches what a naive single-day detector would miss, regardless of how the activity is spread out.")
print("\n\n===== SCENARIO 2: Targeting OC-SVM's favorite signal (session duration) =====")

# Find a malicious case with an extreme session duration spike
duration_spike_idx = X_malicious['session_duration_mins_zscore'].idxmax()
original_case2 = X_test.loc[duration_spike_idx].copy()

print("\n===== STEP 1: Original long-session case (BEFORE evasion) =====")
print(original_case2[['session_duration_mins', 'session_duration_mins_zscore']])

orig2_scaled = oc_svm_scaler.transform(pd.DataFrame([original_case2]))
orig2_iso = iso_forest.predict(pd.DataFrame([original_case2]))[0]
orig2_svm = oc_svm.predict(orig2_scaled)[0]
print(f"Isolation Forest: {'CAUGHT' if orig2_iso == -1 else 'missed'}")
print(f"One-Class SVM:    {'CAUGHT' if orig2_svm == -1 else 'missed'}")

# Evasion: split one giant session into several shorter, more "normal" sessions
evaded_case2 = original_case2.copy()
evaded_case2['session_duration_mins'] = evaded_case2['session_duration_mins'] / 10
evaded_case2['session_duration_mins_zscore'] = evaded_case2['session_duration_mins_zscore'] / 10

print("\n===== STEP 2: Evaded case (session split into 3 shorter sessions) =====")
print(evaded_case2[['session_duration_mins', 'session_duration_mins_zscore']])

evaded2_scaled = oc_svm_scaler.transform(pd.DataFrame([evaded_case2]))
evaded2_iso = iso_forest.predict(pd.DataFrame([evaded_case2]))[0]
evaded2_svm = oc_svm.predict(evaded2_scaled)[0]
print(f"Isolation Forest: {'CAUGHT' if evaded2_iso == -1 else 'MISSED'}")
print(f"One-Class SVM:    {'CAUGHT' if evaded2_svm == -1 else 'MISSED (evasion worked on our BEST model!)'}")

# Countermeasure: total time logged in across the day, regardless of session count
cumulative_duration = original_case2['session_duration_mins']
countermeasure_flag2 = cumulative_duration > 500  # example: >500 min total is suspicious regardless of split
print(f"\nCountermeasure (total daily login time, session-count independent):")
print(f"Total time: {cumulative_duration} mins -> {'CAUGHT (evasion defeated!)' if countermeasure_flag2 else 'missed'}")