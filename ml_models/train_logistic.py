"""
train_logistic.py
Train Logistic Regression on processed MIMIC-IV data.
NO synthetic data - uses only processed_vitals.csv
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Paths
ML_MODELS_DIR = Path(__file__).resolve().parent
PROCESSED_DATA_PATH = ML_MODELS_DIR / "processed_vitals.csv"
MODEL_PATH = ML_MODELS_DIR / "logistic_model.pkl"
SCALER_PATH = ML_MODELS_DIR / "logistic_scaler.pkl"


def load_data():
    """Load processed MIMIC-IV data."""
    print(f"[TRAIN] Loading data from {PROCESSED_DATA_PATH}")
    
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"processed_vitals.csv not found. Run preprocess_mimic.py first."
        )
    
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"[TRAIN] Loaded {len(df)} samples")
    
    # Features and labels
    feature_cols = ['heart_rate', 'spo2', 'temperature']
    
    # Drop rows with missing values in features or label
    df = df.dropna(subset=feature_cols + ['label'])
    
    X = df[feature_cols].values
    y = df['label'].values
    
    print(f"[TRAIN] Features shape: {X.shape}")
    print(f"[TRAIN] Label distribution: Normal={np.sum(y==0)}, Abnormal={np.sum(y==1)}")
    
    return X, y


def train_model(X, y):
    """Train Logistic Regression model."""
    print("\n[TRAIN] Training Logistic Regression...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"[TRAIN] Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with class balancing
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print("\n[TRAIN] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal']))
    
    print("[TRAIN] Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
    
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\n[TRAIN] ROC AUC: {roc_auc:.4f}")
    except Exception as e:
        print(f"[TRAIN] ROC AUC calculation error: {e}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    print(f"[TRAIN] CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Feature coefficients
    print("\n[TRAIN] Feature Coefficients:")
    feature_names = ['heart_rate', 'spo2', 'temperature']
    for name, coef in zip(feature_names, model.coef_[0]):
        print(f"  {name}: {coef:.4f}")
    
    return model, scaler


def save_model(model, scaler):
    """Save model and scaler."""
    print(f"\n[TRAIN] Saving model to {MODEL_PATH}")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"[TRAIN] Saving scaler to {SCALER_PATH}")
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)


def test_inference():
    """Test saved model inference."""
    print("\n[TRAIN] Testing saved model...")
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    # Test cases
    test_cases = [
        {'hr': 75, 'spo2': 98, 'temp': 37.0, 'expected': 'Normal'},
        {'hr': 120, 'spo2': 92, 'temp': 38.5, 'expected': 'Abnormal'},
        {'hr': 50, 'spo2': 88, 'temp': 35.0, 'expected': 'Abnormal'},
    ]
    
    for case in test_cases:
        X = np.array([[case['hr'], case['spo2'], case['temp']]])
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        
        status = 'Abnormal' if pred == 1 else 'Normal'
        confidence = proba[1] if pred == 1 else proba[0]
        
        print(f"  Input: HR={case['hr']}, SpO2={case['spo2']}, Temp={case['temp']}")
        print(f"  Prediction: {status} (confidence: {confidence:.3f}), Expected: {case['expected']}")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Logistic Regression Training")
    print("=" * 60)
    
    # Load data
    X, y = load_data()
    
    # Train model
    model, scaler = train_model(X, y)
    
    # Save model
    save_model(model, scaler)
    
    # Test inference
    test_inference()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()