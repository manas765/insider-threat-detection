import pandas as pd
import numpy as np

# ---------- Load real test labels ----------
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

# ---------- Load each model's scores on the test set ----------
# Pushkar's classical model scores (already combined in his file)
pushkar_scores = pd.read_csv('reports/model_scores_for_comparison.csv')
# Expected columns: true_label, iso_forest_score, oc_svm_score

# Aakash's autoencoder scores (numpy array, same order as y_test)
ae_scores = np.load('reports/ae_scores.npy')

print("Pushkar scores shape:", pushkar_scores.shape)
print("Autoencoder scores shape:", ae_scores.shape)
print("y_test shape:", y_test.shape)

# ---------- Normalize each model's scores to 0-100 ----------
def normalize_0_100(scores):
    scores = np.array(scores)
    min_val, max_val = scores.min(), scores.max()
    return (scores - min_val) / (max_val - min_val) * 100

iso_scaled = normalize_0_100(pushkar_scores['iso_forest_score'])
svm_scaled = normalize_0_100(pushkar_scores['oc_svm_score'])
ae_scaled = normalize_0_100(ae_scores)

# ---------- Combine into one risk score (simple average for now) ----------
risk_score = (iso_scaled + svm_scaled + ae_scaled) / 3

print("\nCombined risk score stats:")
print(pd.Series(risk_score).describe())

# ---------- Save it ----------
risk_df = pd.DataFrame({
    'true_label': pushkar_scores['true_label'],
    'iso_forest_scaled': iso_scaled,
    'oc_svm_scaled': svm_scaled,
    'autoencoder_scaled': ae_scaled,
    'combined_risk_score': risk_score
})
risk_df.to_csv('reports/combined_risk_scores.csv', index=False)
print("\nSaved to reports/combined_risk_scores.csv")