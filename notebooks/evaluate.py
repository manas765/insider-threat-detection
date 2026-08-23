import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# ---------- Load the shared test set (same for everyone) ----------
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()  # squeeze turns single-column df into a series

def evaluate_model(y_true, y_pred, y_scores=None, model_name="Model"):
    """
    y_true   : actual labels (0 = normal, 1 = malicious) — from y_test
    y_pred   : model's predicted labels (0 or 1) for each test row
    y_scores : optional — model's raw anomaly/probability scores (needed for ROC-AUC)
    model_name : just a label for printing
    """
    print(f"\n===== {model_name} =====")
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    if y_scores is not None:
        try:
            auc = roc_auc_score(y_true, y_scores)
            print(f"ROC-AUC:   {auc:.4f}")
        except ValueError as e:
            print(f"ROC-AUC:   could not compute ({e})")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nFull report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Malicious'], zero_division=0))

    return {'model': model_name, 'precision': precision, 'recall': recall, 'f1': f1}


# ---------- Example usage (each person fills in their own model's predictions) ----------

# Isolation Forest example (Pushkar):
# from sklearn.ensemble import IsolationForest
# iso_model = IsolationForest(random_state=42)
# iso_model.fit(X_train_benign)  # trained on benign-only data
# raw_preds = iso_model.predict(X_test)  # gives -1 (anomaly) or 1 (normal)
# y_pred_iso = [1 if p == -1 else 0 for p in raw_preds]  # convert to 0/1 to match y_test
# scores_iso = -iso_model.score_samples(X_test)  # higher score = more anomalous
# evaluate_model(y_test, y_pred_iso, scores_iso, model_name="Isolation Forest")

# OC-SVM example (Pushkar):
# from sklearn.svm import OneClassSVM
# ocsvm_model = OneClassSVM(kernel='rbf')
# ocsvm_model.fit(X_train_benign)
# raw_preds = ocsvm_model.predict(X_test)
# y_pred_svm = [1 if p == -1 else 0 for p in raw_preds]
# scores_svm = -ocsvm_model.decision_function(X_test)
# evaluate_model(y_test, y_pred_svm, scores_svm, model_name="OC-SVM")

# Autoencoder example (Aakash):
# reconstructed = autoencoder_model.predict(X_test)
# reconstruction_error = ((X_test - reconstructed) ** 2).mean(axis=1)
# threshold = reconstruction_error.quantile(0.95)  # pick a cutoff, e.g. top 5% = anomaly
# y_pred_ae = (reconstruction_error > threshold).astype(int)
# evaluate_model(y_test, y_pred_ae, reconstruction_error, model_name="Autoencoder")