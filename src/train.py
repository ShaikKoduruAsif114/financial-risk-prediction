import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report,
    f1_score, precision_score, recall_score
)
import joblib
import os

# ─── LOAD PROCESSED DATA ───────────────────────────────────

def load_data(prefix):
    X_train = np.load(f"data/processed/{prefix}_X_train.npy")
    X_test  = np.load(f"data/processed/{prefix}_X_test.npy")
    y_train = np.load(f"data/processed/{prefix}_y_train.npy")
    y_test  = np.load(f"data/processed/{prefix}_y_test.npy")
    return X_train, X_test, y_train, y_test


# ─── MODELS TO TRAIN ───────────────────────────────────────

def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost":            XGBClassifier(n_estimators=100, random_state=42,
                                            eval_metric="logloss", verbosity=0),
        "LightGBM":           LGBMClassifier(n_estimators=100, random_state=42,
                                             verbosity=-1)
    }


# ─── TRAIN + LOG ONE MODEL ─────────────────────────────────

def train_and_log(model_name, model, X_train, X_test, y_train, y_test, task):
    with mlflow.start_run(run_name=f"{task}_{model_name}"):
        # Log model name and task as params
        mlflow.log_param("model", model_name)
        mlflow.log_param("task", task)

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        auc       = roc_auc_score(y_test, y_proba)
        f1        = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)

        # Log metrics to MLflow
        mlflow.log_metric("auc_roc",   auc)
        mlflow.log_metric("f1_score",  f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall",    recall)

        # Log model artifact
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"  {model_name:20s} | AUC: {auc:.4f} | F1: {f1:.4f} | "
              f"Precision: {precision:.4f} | Recall: {recall:.4f}")

    return auc, model


# ─── TRAIN ALL MODELS FOR ONE TASK ─────────────────────────

def train_task(prefix, task_name):
    print(f"\n{'='*55}")
    print(f"TRAINING: {task_name}")
    print(f"{'='*55}")

    X_train, X_test, y_train, y_test = load_data(prefix)

    best_auc   = 0
    best_model = None
    best_name  = ""

    for model_name, model in get_models().items():
        auc, trained_model = train_and_log(
            model_name, model,
            X_train, X_test,
            y_train, y_test,
            task=prefix
        )
        if auc > best_auc:
            best_auc   = auc
            best_model = trained_model
            best_name  = model_name

    # Save best model to disk
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, f"models/{prefix}_best_model.pkl")

    print(f"\n  Best model: {best_name} with AUC: {best_auc:.4f}")
    print(f"  Saved to: models/{prefix}_best_model.pkl")

    return best_model, best_name


# ─── MAIN ──────────────────────────────────────────────────

if __name__ == "__main__":
    mlflow.set_experiment("Financial-Risk-Prediction")

    # Train both tasks
    credit_model, credit_best = train_task("credit", "Credit Risk")
    fraud_model,  fraud_best  = train_task("fraud",  "Fraud Detection")

    print(f"\n{'='*55}")
    print("TRAINING COMPLETE")
    print(f"  Credit Risk  → Best: {credit_best}")
    print(f"  Fraud        → Best: {fraud_best}")
    print(f"{'='*55}")
    print("\nRun this to view MLflow UI:")
    print("  mlflow ui")