import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# ---------- Load training data ----------
X_train_benign = pd.read_csv('data/processed/X_train_benign.csv')

# ---------- Train Isolation Forest (Pushkar's final hyperparameters) ----------
print("Training Isolation Forest...")
iso_forest = IsolationForest(
    n_estimators=300,
    max_features=0.5,
    max_samples=65536,
    contamination=0.005,
    random_state=42
)
iso_forest.fit(X_train_benign)
print("Isolation Forest trained.")

# ---------- Train One-Class SVM (Pushkar's final hyperparameters) ----------
print("Training One-Class SVM...")
# Note: OC-SVM was trained on a 30,000-row subsample due to scalability
# OC-SVM is sensitive to feature scale, so we standardize features first
X_train_svm_sample = X_train_benign.sample(n=30000, random_state=42)

scaler = StandardScaler()
X_train_svm_scaled = scaler.fit_transform(X_train_svm_sample)

oc_svm = OneClassSVM(
    kernel='rbf',
    nu=0.005,
    gamma=0.8
)
oc_svm.fit(X_train_svm_scaled)
print("One-Class SVM trained.")

# ---------- Save both models AND the scaler ----------
os.makedirs('models', exist_ok=True)
joblib.dump(iso_forest, 'models/isolation_forest.pkl')
joblib.dump(oc_svm, 'models/oc_svm.pkl')
joblib.dump(scaler, 'models/oc_svm_scaler.pkl')

print("\nModels saved to models/isolation_forest.pkl and models/oc_svm.pkl")
print("Scaler saved to models/oc_svm_scaler.pkl (needed before any OC-SVM prediction)")