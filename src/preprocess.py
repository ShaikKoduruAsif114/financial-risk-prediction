import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

# ─── CREDIT RISK ───────────────────────────────────────────

def load_credit_risk():
    df = pd.read_csv("data/raw/UCI_Credit_Card.csv")
    
    # Drop ID column - useless for prediction
    df = df.drop(columns=["ID"])
    
    # Rename target column to something clear
    df = df.rename(columns={"default.payment.next.month": "target"})
    
    print(f"Credit Risk dataset shape: {df.shape}")
    print(f"Default rate: {df['target'].mean():.2%}")
    
    return df


def preprocess_credit_risk(df):
    X = df.drop(columns=["target"])
    y = df["target"]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for use in API later
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/credit_scaler.pkl")
    
    print(f"Train size: {X_train_scaled.shape}, Test size: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()


# ─── FRAUD DETECTION ───────────────────────────────────────

def load_fraud():
    df = pd.read_csv("data/raw/creditcard.csv")
    
    print(f"Fraud dataset shape: {df.shape}")
    print(f"Fraud rate: {df['Class'].mean():.4%}")
    
    return df


def preprocess_fraud(df):
    X = df.drop(columns=["Class"])
    y = df["Class"]
    
    # Train/test split BEFORE SMOTE - important
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale Amount and Time columns
    scaler = StandardScaler()
    X_train[["Amount", "Time"]] = scaler.fit_transform(X_train[["Amount", "Time"]])
    X_test[["Amount", "Time"]] = scaler.transform(X_test[["Amount", "Time"]])
    
    # Save scaler
    joblib.dump(scaler, "models/fraud_scaler.pkl")
    
    # Apply SMOTE only on training data
    print("Applying SMOTE to balance fraud training data...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"After SMOTE - Train size: {X_train_resampled.shape}")
    print(f"Fraud rate after SMOTE: {y_train_resampled.mean():.2%}")
    
    return X_train_resampled, X_test, y_train_resampled, y_test, X.columns.tolist()


# ─── SAVE PROCESSED DATA ───────────────────────────────────

def save_processed(X_train, X_test, y_train, y_test, prefix):
    os.makedirs("data/processed", exist_ok=True)
    
    np.save(f"data/processed/{prefix}_X_train.npy", X_train)
    np.save(f"data/processed/{prefix}_X_test.npy", X_test)
    np.save(f"data/processed/{prefix}_y_train.npy", y_train)
    np.save(f"data/processed/{prefix}_y_test.npy", y_test)
    
    print(f"Saved processed data for: {prefix}")


# ─── MAIN ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("PROCESSING CREDIT RISK DATA")
    print("=" * 50)
    df_credit = load_credit_risk()
    X_train, X_test, y_train, y_test, cols = preprocess_credit_risk(df_credit)
    save_processed(X_train, X_test, y_train, y_test, prefix="credit")
    
    print("\n" + "=" * 50)
    print("PROCESSING FRAUD DATA")
    print("=" * 50)
    df_fraud = load_fraud()
    X_train, X_test, y_train, y_test, cols = preprocess_fraud(df_fraud)
    save_processed(X_train, X_test, y_train, y_test, prefix="fraud")
    
    print("\nAll preprocessing done.")