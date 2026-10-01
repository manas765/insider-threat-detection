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


# ----------------------------------------------------------------------
# 18 role-aware features
# ----------------------------------------------------------------------
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
    "role_usb_events_count_zscore",
    "role_files_accessed_count_zscore",
    "role_email_count_zscore",
    "role_session_duration_mins_zscore",
    "role_after_hours_deviation",
]


# ----------------------------------------------------------------------
# Final tuned hyperparameters
# ----------------------------------------------------------------------
# Selected via validation-split search.
# See tune_on_validation.py for the methodology behind these values.
ISO_FOREST_PARAMS = dict(
    n_estimators=300,
    max_features=0.5,
    max_samples=65536,
    contamination=0.005,
    random_state=RANDOM_STATE,
)

OCSVM_PARAMS = dict(
    kernel="rbf",
    nu=0.005,
    gamma=0.8,
)

OC_SVM_MAX_TRAIN_SIZE = 30000


# ----------------------------------------------------------------------
# Combined risk score helper
# ----------------------------------------------------------------------
def min_max_normalize(scores):
    """
    Normalize anomaly scores to the range [0, 1].

    Higher value means more anomalous.

    Isolation Forest and One-Class SVM produce scores on different
    numerical scales, so normalization is performed before combining
    the two model outputs.
    """
    scores = np.asarray(scores, dtype=float)

    score_min = scores.min()
    score_max = scores.max()

    # Avoid division by zero if every score is identical.
    if score_max == score_min:
        return np.zeros_like(scores)

    return (scores - score_min) / (score_max - score_min)


def train_and_evaluate():
    print("Loading data...")

    # ------------------------------------------------------------------
    # Load role-aware training data.
    #
    # Only benign rows are used to train the unsupervised anomaly
    # detectors.
    # ------------------------------------------------------------------
    train_role = pd.read_csv(
        "data/processed/train_role_features.csv"
    )

    X_train_benign = train_role.loc[
        train_role["is_malicious"] == 0,
        FEATURE_COLUMNS
    ].copy()

    # ------------------------------------------------------------------
    # Load role-aware test features.
    # ------------------------------------------------------------------
    X_test_role = pd.read_csv(
        "data/processed/test_role_features.csv"
    )

    X_test_features = X_test_role[FEATURE_COLUMNS].copy()

    print(
        f"Train (benign only): {X_train_benign.shape}"
    )

    print(
        f"Test: {X_test_features.shape}, "
        f"malicious in test: {int(y_test.sum())} "
        f"({y_test.mean() * 100:.2f}%)"
    )

    # ------------------------------------------------------------------
    # Scale features using the benign training data only.
    #
    # IMPORTANT:
    # The scaler is fitted only on training data.
    # Test data is transformed using the same scaler.
    # ------------------------------------------------------------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train_benign
    )

    X_test_scaled = scaler.transform(
        X_test_features
    )

    # ------------------------------------------------------------------
    # One-Class SVM can be expensive on the full training set,
    # so use a reproducible subset when necessary.
    # ------------------------------------------------------------------
    if X_train_scaled.shape[0] > OC_SVM_MAX_TRAIN_SIZE:

        rng = np.random.RandomState(
            RANDOM_STATE
        )

        idx = rng.choice(
            X_train_scaled.shape[0],
            size=OC_SVM_MAX_TRAIN_SIZE,
            replace=False,
        )

        X_train_svm = X_train_scaled[idx]

    else:
        X_train_svm = X_train_scaled

    # ------------------------------------------------------------------
    # Isolation Forest
    # ------------------------------------------------------------------
    print("\nTraining Isolation Forest...")

    iso_model = IsolationForest(
        **ISO_FOREST_PARAMS
    )

    iso_model.fit(
        X_train_scaled
    )

    raw_preds = iso_model.predict(
        X_test_scaled
    )

    y_pred_iso = [
        1 if p == -1 else 0
        for p in raw_preds
    ]

    # Higher score = more anomalous
    scores_iso = -iso_model.score_samples(
        X_test_scaled
    )

    evaluate_model(
        y_test,
        y_pred_iso,
        scores_iso,
        model_name="Isolation Forest",
    )

    # ------------------------------------------------------------------
    # One-Class SVM
    # ------------------------------------------------------------------
    print("\nTraining One-Class SVM...")

    ocsvm_model = OneClassSVM(
        **OCSVM_PARAMS
    )

    ocsvm_model.fit(
        X_train_svm
    )

    raw_preds = ocsvm_model.predict(
        X_test_scaled
    )

    y_pred_svm = [
        1 if p == -1 else 0
        for p in raw_preds
    ]

    # Higher score = more anomalous
    scores_svm = -ocsvm_model.decision_function(
        X_test_scaled
    )

    evaluate_model(
        y_test,
        y_pred_svm,
        scores_svm,
        model_name="OC-SVM",
    )

    # ------------------------------------------------------------------
    # Save trained models and scaler.
    # ------------------------------------------------------------------
    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        scaler,
        "models/scaler.pkl",
    )

    joblib.dump(
        iso_model,
        "models/iso_forest.pkl",
    )

    joblib.dump(
        ocsvm_model,
        "models/ocsvm.pkl",
    )

    # ------------------------------------------------------------------
    # Combined risk score
    #
    # The two anomaly models produce scores on different numerical
    # scales. Normalize each score to [0, 1] before combining.
    #
    # Higher normalized value = more anomalous.
    #
    # Equal weighting is used so that the combined score is transparent
    # and reproducible.
    # ------------------------------------------------------------------

    iso_risk = min_max_normalize(
        scores_iso
    )

    svm_risk = min_max_normalize(
        scores_svm
    )

    combined_risk_score = (
        0.5 * iso_risk
        +
        0.5 * svm_risk
    )

    # ------------------------------------------------------------------
    # Build final model-score dataframe.
    # ------------------------------------------------------------------
    results_df = pd.DataFrame({
        "user": X_test_role["user"].reset_index(drop=True),
        "day": X_test_role["day"].reset_index(drop=True),
        "role": X_test_role["role"].reset_index(drop=True),
        "true_label": np.asarray(y_test),
        "iso_forest_score": scores_iso,
        "oc_svm_score": scores_svm,
        "iso_forest_risk": iso_risk,
        "oc_svm_risk": svm_risk,
    "combined_risk_score": combined_risk_score,
    })

    # ------------------------------------------------------------------
    # Save reports.
    # ------------------------------------------------------------------
    os.makedirs(
        "reports",
        exist_ok=True
    )

    # Existing model comparison report.
    results_df.to_csv(
        "reports/model_scores_for_comparison.csv",
        index=False,
    )

    # Dashboard-compatible combined risk report.
    results_df.to_csv(
        "reports/combined_risk_scores.csv",
        index=False,
    )

    print(
        f"\nSaved scores to "
        f"reports/model_scores_for_comparison.csv "
        f"({len(results_df)} rows)"
    )

    print(
        f"Saved dashboard risk scores to "
        f"reports/combined_risk_scores.csv "
        f"({len(results_df)} rows)"
    )

    print(
        f"Combined risk range: "
        f"{combined_risk_score.min():.4f} - "
        f"{combined_risk_score.max():.4f}"
    )

    print(
        "\nSaved trained models to models/"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the classical anomaly-detection pipeline"
        )
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Also run alert-fatigue analysis and "
            "SHAP explainability after training"
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Train models and generate evaluation/risk reports.
    # ------------------------------------------------------------------
    train_and_evaluate()

    # ------------------------------------------------------------------
    # Optional full pipeline.
    # ------------------------------------------------------------------
    if args.full:

        print(
            "\n--full flag set: running alert-fatigue analysis "
            "and SHAP explainability..."
        )

        subprocess.run(
            [
                sys.executable,
                "alert_fatigue_analysis.py"
            ],
            check=True,
        )

        subprocess.run(
            [
                sys.executable,
                "shap_explainability.py"
            ],
            check=True,
        )

    print(
        "\nPipeline complete."
    )


if __name__ == "__main__":
    main()