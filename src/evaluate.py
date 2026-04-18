import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

def load_data(prefix):
    X_test  = np.load(f"data/processed/{prefix}_X_test.npy")
    y_test  = np.load(f"data/processed/{prefix}_y_test.npy")
    return X_test, y_test

def get_feature_names(prefix):
    if prefix == "credit":
        return [
            'LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE',
            'PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
            'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
            'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'
        ]
    else:
        return [
            'Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
            'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
            'V21','V22','V23','V24','V25','V26','V27','V28','Amount'
        ]

def run_shap(prefix, model_path):
    print(f"\n{'='*50}")
    print(f"SHAP Analysis: {prefix.upper()}")
    print(f"{'='*50}")

    # Load model and data
    model   = joblib.load(model_path)
    X_test, y_test = load_data(prefix)
    feature_names  = get_feature_names(prefix)

    # Use sample for speed — SHAP on full test set is slow
    sample_size = min(500, len(X_test))
    X_sample    = X_test[:sample_size]

    print(f"Running SHAP on {sample_size} samples...")

    # TreeExplainer works for XGBoost, LightGBM, RandomForest
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Handle LightGBM binary output shape
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    os.makedirs("models", exist_ok=True)

    # Plot 1 — Summary bar plot (global feature importance)
    plt.figure()
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.title(f"{prefix.upper()} — Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(f"models/{prefix}_shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: models/{prefix}_shap_bar.png")

    # Plot 2 — Beeswarm plot (impact direction)
    plt.figure()
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        show=False
    )
    plt.title(f"{prefix.upper()} — SHAP Beeswarm")
    plt.tight_layout()
    plt.savefig(f"models/{prefix}_shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: models/{prefix}_shap_beeswarm.png")

if __name__ == "__main__":
    run_shap("credit", "models/credit_best_model.pkl")
    run_shap("fraud",  "models/fraud_best_model.pkl")
    print("\nSHAP analysis complete.")