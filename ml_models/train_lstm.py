"""
train_lstm.py
Train LSTM model for heart rate prediction on MIMIC-IV data.
Uses sliding windows of 30 timesteps to predict next HR value.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Paths
ML_MODELS_DIR = Path(__file__).resolve().parent
PROCESSED_DATA_PATH = ML_MODELS_DIR / "processed_vitals.csv"
MODEL_PATH = ML_MODELS_DIR / "lstm_model.h5"
SCALER_PATH = ML_MODELS_DIR / "scaler.pkl"

# Configuration
WINDOW_SIZE = 30
EPOCHS = 50
BATCH_SIZE = 64
LSTM_UNITS = 64


def load_data():
    """Load heart rate data from processed MIMIC-IV data."""
    print(f"[TRAIN] Loading data from {PROCESSED_DATA_PATH}")
    
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"processed_vitals.csv not found. Run preprocess_mimic.py first."
        )
    
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"[TRAIN] Loaded {len(df)} records")
    
    return df


def create_sequences(df, window_size=WINDOW_SIZE):
    """Create sliding window sequences for LSTM training."""
    print(f"[TRAIN] Creating sequences with window size {window_size}")
    
    X_sequences = []
    y_sequences = []
    
    # Group by subject and create sequences
    for subject_id, group in df.groupby('subject_id'):
        group = group.sort_values('charttime')
        hr_values = group['heart_rate'].dropna().values
        
        if len(hr_values) < window_size + 1:
            continue
        
        for i in range(len(hr_values) - window_size):
            X_sequences.append(hr_values[i:i + window_size])
            y_sequences.append(hr_values[i + window_size])
    
    X = np.array(X_sequences)
    y = np.array(y_sequences)
    
    print(f"[TRAIN] Created {len(X)} sequences")
    print(f"[TRAIN] X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y


def prepare_data(X, y):
    """Scale data and prepare for LSTM."""
    print("[TRAIN] Preparing data...")
    
    # Fit scaler on all HR values
    scaler = MinMaxScaler(feature_range=(0, 1))
    all_values = np.concatenate([X.flatten(), y.flatten()]).reshape(-1, 1)
    scaler.fit(all_values)
    
    # Scale X and y
    n_samples = X.shape[0]
    X_scaled = scaler.transform(X.reshape(-1, 1)).reshape(n_samples, WINDOW_SIZE, 1)
    y_scaled = scaler.transform(y.reshape(-1, 1)).flatten()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    print(f"[TRAIN] Train: {len(X_train)}, Test: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, scaler


def build_model():
    """Build LSTM model architecture."""
    print("[TRAIN] Building LSTM model...")
    
    model = Sequential([
        LSTM(LSTM_UNITS, return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    model.summary()
    return model


def train_model(model, X_train, y_train, X_test, y_test):
    """Train the LSTM model."""
    print("\n[TRAIN] Training LSTM model...")
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            str(MODEL_PATH),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def evaluate_model(model, X_test, y_test, scaler):
    """Evaluate model performance."""
    print("\n[TRAIN] Evaluating model...")
    
    # Predict
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Inverse transform
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred_scaled).flatten()
    
    # Metrics
    mse = np.mean((y_test_inv - y_pred_inv) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_inv - y_pred_inv))
    mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
    
    print(f"[TRAIN] MSE: {mse:.4f}")
    print(f"[TRAIN] RMSE: {rmse:.4f}")
    print(f"[TRAIN] MAE: {mae:.4f}")
    print(f"[TRAIN] MAPE: {mape:.2f}%")
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape}


def save_scaler(scaler):
    """Save the scaler."""
    print(f"[TRAIN] Saving scaler to {SCALER_PATH}")
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)


def test_inference(scaler):
    """Test model inference."""
    print("\n[TRAIN] Testing inference...")
    
    model = tf.keras.models.load_model(str(MODEL_PATH),compile=False)
    
    # Generate test sequence
    test_hr = np.array([75 + np.sin(i/5) * 5 for i in range(WINDOW_SIZE)])
    print(f"  Input (last 5): {test_hr[-5:].round(1)}")
    
    # Scale and predict
    test_scaled = scaler.transform(test_hr.reshape(-1, 1)).reshape(1, WINDOW_SIZE, 1)
    pred_scaled = model.predict(test_scaled, verbose=0)
    pred = scaler.inverse_transform(pred_scaled)[0, 0]
    
    print(f"  Predicted next HR: {pred:.1f} BPM")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("LSTM Training for Heart Rate Prediction")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    
    # Load data
    df = load_data()
    
    # Create sequences
    X, y = create_sequences(df)
    
    if len(X) < 100:
        print("[WARNING] Limited data available. Model may not train well.")
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(X, y)
    
    # Build model
    model = build_model()
    
    # Train model
    history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Evaluate
    best_model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    metrics = evaluate_model(best_model, X_test, y_test, scaler)
    
    # Save scaler
    save_scaler(scaler)
    
    # Test inference
    test_inference(scaler)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Scaler saved: {SCALER_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()