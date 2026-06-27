
"""
signal_processing.py
Signal processing with autocorrelation and moving average.
"""

import numpy as np
from typing import List, Tuple, Optional
from collections import deque


class PatientBuffer:
    """Per-patient data buffer for signal processing."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.heart_rate = deque(maxlen=max_size)
        self.spo2 = deque(maxlen=max_size)
        self.temperature = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
    
    def add(self, timestamp, hr: float, spo2: float, temp: float):
        """Add a reading to the buffer."""
        self.timestamps.append(timestamp)
        self.heart_rate.append(hr)
        self.spo2.append(spo2)
        self.temperature.append(temp)
    
    def get_hr_array(self) -> np.ndarray:
        """Get heart rate as numpy array."""
        return np.array(list(self.heart_rate))
    
    def __len__(self):
        return len(self.heart_rate)


class SignalProcessor:
    """Signal processing for vital signs."""
    
    def __init__(self):
        self.patient_buffers = {}
    
    def get_buffer(self, patient_id: str) -> PatientBuffer:
        """Get or create buffer for a patient."""
        if patient_id not in self.patient_buffers:
            self.patient_buffers[patient_id] = PatientBuffer(max_size=100)
        return self.patient_buffers[patient_id]
    
    def add_reading(self, patient_id: str, timestamp, hr: float, spo2: float, temp: float):
        """Add a reading to patient buffer."""
        buffer = self.get_buffer(patient_id)
        buffer.add(timestamp, hr, spo2, temp)
    
    def get_hr_history(self, patient_id: str) -> List[float]:
        """Get heart rate history for a patient."""
        buffer = self.get_buffer(patient_id)
        return list(buffer.heart_rate)
    
    def get_buffer_length(self, patient_id: str) -> int:
        """Get buffer length for a patient."""
        if patient_id not in self.patient_buffers:
            return 0
        return len(self.patient_buffers[patient_id])


def moving_average(data: List[float], window_size: int = 5) -> float:
    """
    Compute moving average of the most recent values.
    
    Args:
        data: List of values
        window_size: Window size for averaging
        
    Returns:
        Moving average value
    """
    if not data:
        return 0.0
    
    if len(data) < window_size:
        return np.mean(data)
    
    return np.mean(data[-window_size:])


def moving_average_full(data: List[float], window_size: int = 5) -> List[float]:
    """
    Compute moving average for entire series.
    
    Args:
        data: List of values
        window_size: Window size
        
    Returns:
        List of smoothed values
    """
    if len(data) < window_size:
        return [np.mean(data)] * len(data)
    
    result = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        result.append(np.mean(data[start:i+1]))
    
    return result


def compute_autocorrelation(data: List[float], lag: int = 1) -> float:
    """
    Compute autocorrelation at a specific lag.
    
    Autocorrelation measures how similar a signal is to a delayed version of itself.
    
    Args:
        data: List of values (heart rate history)
        lag: Time lag for autocorrelation
        
    Returns:
        Autocorrelation coefficient at the specified lag
    """
    if len(data) < lag + 2:
        return 0.0
    
    data = np.array(data)
    n = len(data)
    
    # Center the data
    mean = np.mean(data)
    data_centered = data - mean
    
    # Variance
    variance = np.var(data)
    if variance == 0:
        return 0.0
    
    # Autocorrelation at lag
    autocorr = np.sum(data_centered[:-lag] * data_centered[lag:]) / ((n - lag) * variance)
    
    return float(autocorr)


def compute_autocorrelation_series(data: List[float], max_lag: int = 20) -> List[float]:
    """
    Compute autocorrelation for multiple lags.
    
    Args:
        data: List of values
        max_lag: Maximum lag to compute
        
    Returns:
        List of autocorrelation values for lags 0 to max_lag
    """
    if len(data) < 3:
        return [1.0]
    
    max_lag = min(max_lag, len(data) - 1)
    autocorr = []
    
    for lag in range(max_lag + 1):
        if lag == 0:
            autocorr.append(1.0)
        else:
            autocorr.append(compute_autocorrelation(data, lag))
    
    return autocorr


def compute_hr_variability(data: List[float]) -> dict:
    """
    Compute heart rate variability metrics.
    
    Args:
        data: List of heart rate values
        
    Returns:
        Dictionary with variability metrics
    """
    if len(data) < 2:
        return {'std': 0.0, 'rmssd': 0.0, 'range': 0.0}
    
    data = np.array(data)
    
    # Standard deviation
    std = np.std(data)
    
    # RMSSD (root mean square of successive differences)
    diff = np.diff(data)
    rmssd = np.sqrt(np.mean(diff ** 2)) if len(diff) > 0 else 0.0
    
    # Range
    data_range = np.max(data) - np.min(data)
    
    return {
        'std': float(std),
        'rmssd': float(rmssd),
        'range': float(data_range)
    }


def detect_anomaly_threshold(hr: float, spo2: float, temp: float) -> Tuple[bool, str]:
    """
    Detect anomaly using medical thresholds.
    
    Args:
        hr: Heart rate
        spo2: Blood oxygen saturation
        temp: Body temperature
        
    Returns:
        Tuple of (is_abnormal, reason)
    """
    reasons = []
    
    # Heart rate thresholds
    if hr < 60:
        reasons.append(f"Bradycardia (HR={hr:.1f})")
    elif hr > 100:
        reasons.append(f"Tachycardia (HR={hr:.1f})")
    
    # SpO2 threshold
    if spo2 < 95:
        reasons.append(f"Low SpO2 ({spo2:.1f}%)")
    
    # Temperature thresholds
    if temp < 36.1:
        reasons.append(f"Hypothermia ({temp:.1f}°C)")
    elif temp > 37.8:
        reasons.append(f"Fever ({temp:.1f}°C)")
    
    is_abnormal = len(reasons) > 0
    reason = "; ".join(reasons) if reasons else "Normal"
    
    return is_abnormal, reason


# Singleton processor instance
_processor = None


def get_processor() -> SignalProcessor:
    """Get singleton processor instance."""
    global _processor
    if _processor is None:
        _processor = SignalProcessor()
    return _processor


if __name__ == "__main__":
    # Test autocorrelation
    import random
    
    # Generate test data with pattern
    test_data = [70 + 5 * np.sin(i / 5) + random.gauss(0, 1) for i in range(100)]
    
    # Compute autocorrelation
    autocorr = compute_autocorrelation(test_data, lag=1)
    print(f"Autocorrelation (lag=1): {autocorr:.4f}")
    
    autocorr_series = compute_autocorrelation_series(test_data, max_lag=10)
    print(f"Autocorrelation series: {[f'{a:.3f}' for a in autocorr_series]}")
    
    # Test moving average
    smoothed = moving_average(test_data, window_size=5)
    print(f"Moving average (last 5): {smoothed:.2f}")
    
    # Test variability
    variability = compute_hr_variability(test_data)
    print(f"Variability: {variability}")