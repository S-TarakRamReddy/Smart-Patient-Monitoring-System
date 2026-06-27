"""
anomaly_detection.py
ML model loading and inference for anomaly detection and HR prediction.
Separate from training - runtime inference only.
"""

import os
import pickle
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path

# TensorFlow import with error handling
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARNING] TensorFlow not available")

# Model paths
ML_MODELS_DIR = Path(__file__).resolve().parent.parent / "ml_models"
LOGISTIC_MODEL_PATH = ML_MODELS_DIR / "logistic_model.pkl"
LOGISTIC_SCALER_PATH = ML_MODELS_DIR / "logistic_scaler.pkl"
LSTM_MODEL_PATH = ML_MODELS_DIR / "lstm_model.h5"
LSTM_SCALER_PATH = ML_MODELS_DIR / "scaler.pkl"

# Window size for LSTM
LSTM_WINDOW_SIZE = 30


class AnomalyDetector:
    """Runtime anomaly detection and prediction."""
    
    def __init__(self):
        self.logistic_model = None
        self.logistic_scaler = None
        self.lstm_model = None
        self.lstm_scaler = None
        self.models_loaded = False
        
        self._load_models()
    
    def _load_models(self):
        """Load trained models from disk."""
        # Load Logistic Regression
        try:
            if LOGISTIC_MODEL_PATH.exists() and LOGISTIC_SCALER_PATH.exists():
                with open(LOGISTIC_MODEL_PATH, 'rb') as f:
                    self.logistic_model = pickle.load(f)
                with open(LOGISTIC_SCALER_PATH, 'rb') as f:
                    self.logistic_scaler = pickle.load(f)
                print("[ANOMALY] Logistic Regression model loaded")
            else:
                print("[ANOMALY] Logistic model not found - using threshold-based detection")
        except Exception as e:
            print(f"[ANOMALY] Error loading Logistic model: {e}")
        
        # Load LSTM
        if TF_AVAILABLE:
            try:
                if LSTM_MODEL_PATH.exists() and LSTM_SCALER_PATH.exists():
                    self.lstm_model = tf.keras.models.load_model(str(LSTM_MODEL_PATH), compile=False)
                    with open(LSTM_SCALER_PATH, 'rb') as f:
                        self.lstm_scaler = pickle.load(f)
                    print("[ANOMALY] LSTM model loaded")
                else:
                    print("[ANOMALY] LSTM model not found - using trend-based prediction")
            except Exception as e:
                print(f"[ANOMALY] Error loading LSTM model: {e}")
        
        self.models_loaded = (self.logistic_model is not None or self.lstm_model is not None)
    
    def predict_status(self, hr: float, spo2: float, temp: float) -> Tuple[str, float]:
        """
        Predict patient status using Logistic Regression.
        
        Args:
            hr: Heart rate
            spo2: Blood oxygen saturation
            temp: Body temperature
            
        Returns:
            Tuple of (status, confidence)
        """
        # ML prediction
        if self.logistic_model is not None and self.logistic_scaler is not None:
            try:
                X = np.array([[hr, spo2, temp]])
                X_scaled = self.logistic_scaler.transform(X)
                
                prediction = self.logistic_model.predict(X_scaled)[0]
                proba = self.logistic_model.predict_proba(X_scaled)[0]
                
                status = 'Critical' if prediction == 1 else 'Normal'
                confidence = proba[1] if prediction == 1 else proba[0]
                
                return status, float(confidence)
            except Exception as e:
                print(f"[ANOMALY] Logistic prediction error: {e}")
        
        # Fallback: threshold-based
        is_abnormal = (hr < 60 or hr > 100 or spo2 < 95 or temp < 36.1 or temp > 37.8)
        status = 'Critical' if is_abnormal else 'Normal'
        confidence = 0.8 if is_abnormal else 0.9
        
        return status, confidence
    
    def predict_next_hr(self, hr_history: List[float]) -> Optional[float]:
        """
        Predict next heart rate using LSTM.
        
        Args:
            hr_history: List of historical heart rate values (need >= 30)
            
        Returns:
            Predicted heart rate or None if insufficient data
        """
        if len(hr_history) < LSTM_WINDOW_SIZE:
            return None
        
        # LSTM prediction
        if self.lstm_model is not None and self.lstm_scaler is not None:
            try:
                # Get last WINDOW_SIZE values
                sequence = np.array(hr_history[-LSTM_WINDOW_SIZE:])
                
                # Scale
                sequence_scaled = self.lstm_scaler.transform(sequence.reshape(-1, 1))
                
                # Reshape for LSTM: (1, window_size, 1)
                X = sequence_scaled.reshape(1, LSTM_WINDOW_SIZE, 1)
                
                # Predict
                pred_scaled = self.lstm_model.predict(X, verbose=0)
                
                # Inverse transform
                pred = self.lstm_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
                
                # Clamp to physiological range
                pred = max(30, min(200, pred))
                
                return float(pred)
            except Exception as e:
                print(f"[ANOMALY] LSTM prediction error: {e}")
        
        # Fallback: simple trend
        return self._predict_trend(hr_history)
    
    def _predict_trend(self, hr_history: List[float]) -> float:
        """Simple trend-based prediction fallback."""
        if len(hr_history) < 5:
            return hr_history[-1] if hr_history else 75.0
        
        recent = hr_history[-10:]
        
        # Linear regression
        x = np.arange(len(recent))
        coeffs = np.polyfit(x, recent, 1)
        
        # Predict next value
        predicted = coeffs[0] * (len(recent)) + coeffs[1]
        
        # Clamp
        predicted = max(40, min(180, predicted))
        
        return float(predicted)
    
    def get_model_status(self) -> dict:
        """Get status of loaded models."""
        return {
            'logistic_loaded': self.logistic_model is not None,
            'lstm_loaded': self.lstm_model is not None,
            'tensorflow_available': TF_AVAILABLE
        }


# Singleton instance
_detector = None


def get_detector() -> AnomalyDetector:
    """Get singleton detector instance."""
    global _detector
    if _detector is None:
        _detector = AnomalyDetector()
    return _detector


if __name__ == "__main__":
    # Test the detector
    detector = get_detector()
    
    print("\nModel Status:")
    print(detector.get_model_status())
    
    print("\nTest 1: Normal vitals")
    status, conf = detector.predict_status(75, 98, 37.0)
    print(f"  Status: {status}, Confidence: {conf:.3f}")
    
    print("\nTest 2: Abnormal vitals")
    status, conf = detector.predict_status(120, 91, 38.5)
    print(f"  Status: {status}, Confidence: {conf:.3f}")
    
    print("\nTest 3: HR Prediction")
    hr_history = [75 + np.sin(i/5)*5 for i in range(40)]
    predicted = detector.predict_next_hr(hr_history)
    print(f"  Predicted next HR: {predicted:.1f}")