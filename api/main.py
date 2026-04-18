from fastapi import FastAPI
from api.schemas import CreditInput, FraudInput, PredictionResponse
import joblib
import numpy as np
import shap

app = FastAPI(
    title="Financial Risk Prediction API",
    description="Predicts credit default risk and transaction fraud",
    version="1.0.0"
)

# ─── LOAD MODELS ONCE AT STARTUP ───────────────────────────

credit_model  = joblib.load("models/credit_best_model.pkl")
fraud_model   = joblib.load("models/fraud_best_model.pkl")
credit_scaler = joblib.load("models/credit_scaler.pkl")
fraud_scaler  = joblib.load("models/fraud_scaler.pkl")

credit_features = [
    'LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE',
    'PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
    'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
    'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'
]

fraud_features = [
    'Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
    'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
    'V21','V22','V23','V24','V25','V26','V27','V28','Amount'
]

# ─── SHAP EXPLAINERS ───────────────────────────────────────

credit_explainer = shap.TreeExplainer(credit_model)
fraud_explainer  = shap.TreeExplainer(fraud_model)

# ─── HELPER ────────────────────────────────────────────────

def get_top_factors(shap_values, feature_names, top_n=3):
    indices = np.argsort(np.abs(shap_values))[::-1][:top_n]
    return [feature_names[i] for i in indices]

def get_risk_level(probability):
    if probability < 0.3:
        return "LOW"
    elif probability < 0.6:
        return "MEDIUM"
    else:
        return "HIGH"

# ─── ROUTES ────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Financial Risk Prediction API is running"}

@app.post("/predict/credit", response_model=PredictionResponse)
def predict_credit(data: CreditInput):
    # Convert input to array
    X = np.array([[getattr(data, f) for f in credit_features]])

    # Scale
    X_scaled = credit_scaler.transform(X)

    # Predict
    prediction  = int(credit_model.predict(X_scaled)[0])
    probability = float(credit_model.predict_proba(X_scaled)[0][1])

    # SHAP
    shap_vals = credit_explainer.shap_values(X_scaled)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    top_factors = get_top_factors(shap_vals[0], credit_features)

    return PredictionResponse(
        prediction=prediction,
        probability=round(probability, 4),
        risk_level=get_risk_level(probability),
        top_factors=top_factors
    )

@app.post("/predict/fraud", response_model=PredictionResponse)
def predict_fraud(data: FraudInput):
    # Convert input to array
    X = np.array([[getattr(data, f) for f in fraud_features]])

    # Scale Amount and Time
    X_df = X.copy()
    X[:, [0, -1]] = fraud_scaler.transform(X[:, [0, -1]])

    # Predict
    prediction  = int(fraud_model.predict(X)[0])
    probability = float(fraud_model.predict_proba(X)[0][1])

    # SHAP
    shap_vals   = fraud_explainer.shap_values(X)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    top_factors = get_top_factors(shap_vals[0], fraud_features)

    return PredictionResponse(
        prediction=prediction,
        probability=round(probability, 4),
        risk_level=get_risk_level(probability),
        top_factors=top_factors
    )