"""
run_pipeline.py -- Single entry point for the classical anomaly-detection
pipeline (Isolation Forest + One-Class SVM).

Usage:
    python run_pipeline.py            # train + evaluate only
    python run_pipeline.py --full     # also runs alert-fatigue analysis + SHAP

Assumes data/processed/*.csv and notebooks/evaluate.py already exist --
raw CERT log processing (Manas's pipeline) is a separate, heavier step
not included here, since the source dataset (~5GB) isn't distributed
with this repo. See README for details.
"""

import argparse
import subprocess
import sys
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

sys.path.append("notebooks")
from evaluate import evaluate_model, X_test, y_test

RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "login_hour", "after_hours_flag", "session_duration_mins",
    "usb_events_count", "files_accessed_count", "email_count",
    "unique_domains_visited", "email_ext_recipient_count",
    "usb_events_count_zscore", "files_accessed_count_zscore",
    "email_count_zscore", "session_duration_mins_zscore",
    "days_since_last_spike",
]

# Final tuned hyperparameters -- selected via validation-split search.
# See tune_on_validation.py for the methodology behind these values.
ISO_FOREST_PARAMS = dict(
    n_estimators=300, max_features=0.5, max_samples=65536,
    contamination=0.005, random_state=RANDOM_STATE,
)
OCSVM_PARAMS = dict(kernel="rbf", nu=0.005, gamma=0.8)
OC_SVM_MAX_TRAIN_SIZE = 30000


def train_and_evaluate():
    print("Loading data...")
    X_train_benign = pd.read_csv("data/processed/X_train_benign.csv")[FEATURE_COLUMNS]
    X_test_features = X_test[FEATURE_COLUMNS]

    print(f"Train (benign only): {X_train_benign.shape}")
    print(f"Test: {X_test_features.shape}, malicious in test: {int(y_test.sum())} ({y_test.mean()*100:.2f}%)")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_benign)
    X_test_scaled = scaler.transform(X_test_features)

    if X_train_scaled.shape[0] > OC_SVM_MAX_TRAIN_SIZE:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(X_train_scaled.shape[0], size=OC_SVM_MAX_TRAIN_SIZE, replace=False)
        X_train_svm = X_train_scaled[idx]
    else:
        X_train_svm = X_train_scaled

    print("\nTraining Isolation Forest...")
    iso_model = IsolationForest(**ISO_FOREST_PARAMS)
    iso_model.fit(X_train_scaled)
    raw_preds = iso_model.predict(X_test_scaled)
    y_pred_iso = [1 if p == -1 else 0 for p in raw_preds]
    scores_iso = -iso_model.score_samples(X_test_scaled)
    evaluate_model(y_test, y_pred_iso, scores_iso, model_name="Isolation Forest")

    print("\nTraining One-Class SVM...")
    ocsvm_model = OneClassSVM(**OCSVM_PARAMS)
    ocsvm_model.fit(X_train_svm)
    raw_preds = ocsvm_model.predict(X_test_scaled)
    y_pred_svm = [1 if p == -1 else 0 for p in raw_preds]
    scores_svm = -ocsvm_model.decision_function(X_test_scaled)
    evaluate_model(y_test, y_pred_svm, scores_svm, model_name="OC-SVM")

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(iso_model, "models/iso_forest.pkl")
    joblib.dump(ocsvm_model, "models/ocsvm.pkl")

    results_df = pd.DataFrame({
        "true_label": np.asarray(y_test),
        "iso_forest_score": scores_iso,
        "oc_svm_score": scores_svm,
    })
    results_df.to_csv("reports/model_scores_for_comparison.csv", index=False)
    print(f"\nSaved scores to reports/model_scores_for_comparison.csv ({len(results_df)} rows)")
    print("Saved trained models to models/")


def main():
    parser = argparse.ArgumentParser(description="Run the classical anomaly-detection pipeline")
    parser.add_argument(
        "--full", action="store_true",
        help="Also run alert-fatigue analysis and SHAP explainability after training",
    )
    args = parser.parse_args()

    train_and_evaluate()

    if args.full:
        print("\n--full flag set: running alert-fatigue analysis and SHAP explainability...")
        subprocess.run([sys.executable, "alert_fatigue_analysis.py"], check=True)
        subprocess.run([sys.executable, "shap_explainability.py"], check=True)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()