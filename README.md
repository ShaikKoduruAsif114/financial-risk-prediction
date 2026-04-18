# Financial Risk Prediction System

An end-to-end ML system for credit default risk scoring and fraud detection, 
with experiment tracking, explainability, and a deployed REST API.

## Results

| Task | Best Model | AUC-ROC |
|---|---|---|
| Credit Risk | LightGBM | 0.7787 |
| Fraud Detection | XGBoost | 0.9792 |

## Architecture

Raw Data (UCI Credit + Kaggle Fraud)
↓
Preprocessing + SMOTE (handle imbalance)
↓
4 Models trained: LogReg, RandomForest, XGBoost, LightGBM
All runs tracked with MLflow
↓
Best model selected per task
SHAP explainability analysis
↓
FastAPI — POST /predict/credit
POST /predict/fraud
↓
Live deployment on Render

## Tech Stack

| Layer | Tool |
|---|---|
| Data Processing | pandas, scikit-learn, imbalanced-learn |
| Model Training | XGBoost, LightGBM, RandomForest, LogisticRegression |
| Experiment Tracking | MLflow |
| Explainability | SHAP |
| API | FastAPI |
| Deployment | Render |

## API Usage

### Fraud Detection
```bash
curl -X POST "http://localhost:8000/predict/fraud" \
-H "Content-Type: application/json" \
-d '{"Time": 406.0, "V1": -2.31, "V2": 1.95, ..., "Amount": 149.62}'
```

### Response
```json
{
  "prediction": 1,
  "probability": 0.98,
  "risk_level": "HIGH",
  "top_factors": ["V14", "V7", "V17"]
}
```

## Project Structure

financial-risk-prediction/
├── src/
│   ├── preprocess.py    # Data cleaning + SMOTE
│   ├── train.py         # Train 4 models + MLflow logging
│   └── evaluate.py      # SHAP analysis
├── api/
│   ├── main.py          # FastAPI endpoints
│   └── schemas.py       # Request/response schemas
└── requirements.txt

## Key Features

- **SMOTE** — handles severe class imbalance in fraud data (0.17% fraud rate)
- **MLflow tracking** — every model run logged with params and metrics
- **SHAP explainability** — every prediction explained with top contributing features
- **Two endpoints** — credit risk + fraud detection in one API