from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

BASE_FEATURES = [
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

ROLE_FEATURES = [
    "role_usb_events_count_zscore",
    "role_files_accessed_count_zscore",
    "role_email_count_zscore",
    "role_session_duration_mins_zscore",
    "role_after_hours_deviation",
]

ISO_PARAMS = {
    "n_estimators": 300,
    "max_features": 0.5,
    "max_samples": 65536,
    "contamination": 0.005,
    "random_state": 42,
}

OCSVM_PARAMS = {
    "kernel": "rbf",
    "nu": 0.005,
    "gamma": 0.8,
}

OCSVM_MAX_TRAIN_SIZE = 30000


def calculate_metrics(y_true, scores, predictions):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    predictions = np.asarray(predictions)

    malicious_caught = int(((predictions == 1) & (y_true == 1)).sum())
    false_positives = int(((predictions == 1) & (y_true == 0)).sum())
    true_positives = malicious_caught
    alerts = int((predictions == 1).sum())

    return {
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "malicious_caught": malicious_caught,
        "false_positives": false_positives,
        "alerts": alerts,
        "false_positive_rate": false_positives / max((y_true == 0).sum(), 1),
        "true_positives": true_positives,
    }


def evaluate_configuration(train, test, features, configuration_name):
    train_benign = train[train["is_malicious"] == 0]

    X_train = train_benign[features]
    X_test = test[features]
    y_test = test["is_malicious"].to_numpy()

    results = []

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n=== {configuration_name} ===")
    print(f"Features: {len(features)}")

    iso_model = IsolationForest(**ISO_PARAMS)
    iso_model.fit(X_train_scaled)

    iso_scores = -iso_model.score_samples(X_test_scaled)
    iso_predictions = (iso_model.predict(X_test_scaled) == -1).astype(int)

    iso_metrics = calculate_metrics(
        y_test,
        iso_scores,
        iso_predictions,
    )

    results.append(
        {
            "configuration": configuration_name,
            "model": "Isolation Forest",
            **iso_metrics,
        }
    )

    rng = np.random.RandomState(42)

    if len(X_train_scaled) > OCSVM_MAX_TRAIN_SIZE:
        indices = rng.choice(
            len(X_train_scaled),
            size=OCSVM_MAX_TRAIN_SIZE,
            replace=False,
        )
        X_train_svm = X_train_scaled[indices]
    else:
        X_train_svm = X_train_scaled

    ocsvm_model = OneClassSVM(**OCSVM_PARAMS)
    ocsvm_model.fit(X_train_svm)

    svm_scores = -ocsvm_model.decision_function(X_test_scaled)
    svm_predictions = (ocsvm_model.predict(X_test_scaled) == -1).astype(int)

    svm_metrics = calculate_metrics(
        y_test,
        svm_scores,
        svm_predictions,
    )

    results.append(
        {
            "configuration": configuration_name,
            "model": "One-Class SVM",
            **svm_metrics,
        }
    )

    return results


def main():
    train = pd.read_csv(
        PROCESSED / "train_role_features.csv"
    )
    test = pd.read_csv(
        PROCESSED / "test_role_features.csv"
    )

    results = []

    results.extend(
        evaluate_configuration(
            train,
            test,
            BASE_FEATURES,
            "13-feature baseline",
        )
    )

    results.extend(
        evaluate_configuration(
            train,
            test,
            BASE_FEATURES + ROLE_FEATURES,
            "18-feature role-aware",
        )
    )

    results_df = pd.DataFrame(results)

    output_path = (
        PROCESSED / "role_model_comparison_detailed.csv"
    )
    results_df.to_csv(output_path, index=False)

    print("\n=== DETAILED COMPARISON ===")
    print(
        results_df[
            [
                "configuration",
                "model",
                "roc_auc",
                "pr_auc",
                "precision",
                "recall",
                "f1",
                "malicious_caught",
                "false_positives",
                "alerts",
                "false_positive_rate",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()