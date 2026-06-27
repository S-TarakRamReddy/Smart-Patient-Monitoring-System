#python backend/mqtt_listener.py
"""
mqtt_listener.py
MQTT subscriber with alert suppression and clinical language.
No model-specific labels exposed - unified decision engine.
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Tuple, Optional
import numpy as np

import paho.mqtt.client as mqtt

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import (
    init_database, insert_vital_reading, update_patient_status,
    insert_alert, validate_device_token, register_device,
    recent_alert_exists
)
from backend.signal_processing import (
    get_processor, moving_average, compute_autocorrelation
)
from backend.anomaly_detection import get_detector

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuration
MQTT_BROKER = os.environ.get('MQTT_BROKER', 'localhost')
MQTT_PORT = int(os.environ.get('MQTT_PORT', 1883))

TOPIC_VITALS = "hospital/vitals/#"
TOPIC_ALERTS = "hospital/alerts"

# Valid device tokens
VALID_TOKENS = {
    "sensor_001": os.environ.get("SENSOR_001_TOKEN", "secure_token_001"),
    "sensor_002": os.environ.get("SENSOR_002_TOKEN", "secure_token_002"),
    "sensor_003": os.environ.get("SENSOR_003_TOKEN", "secure_token_003"),
    "sensor_004": os.environ.get("SENSOR_004_TOKEN", "secure_token_004"),
    "sensor_005": os.environ.get("SENSOR_005_TOKEN", "secure_token_005")
}

# Minimum buffer for LSTM prediction
MIN_BUFFER_SIZE = 30

# Alert suppression window (seconds)
ALERT_SUPPRESSION_WINDOW = 60

# ============================================================
# PHYSIOLOGICAL THRESHOLDS (Medically Grounded)
# ============================================================
SEVERE_BRADYCARDIA_THRESHOLD = 40
MODERATE_BRADYCARDIA_THRESHOLD = 50
BRADYCARDIA_THRESHOLD = 60
SEVERE_TACHYCARDIA_THRESHOLD = 150
MODERATE_TACHYCARDIA_THRESHOLD = 120
TACHYCARDIA_THRESHOLD = 100
RAPID_DROP_THRESHOLD = 15
PROGRESSIVE_DECLINE_SLOPE = -2.0
UNSTABLE_AUTOCORR_THRESHOLD = 0.2

# ============================================================
# CLINICAL ALERT MESSAGES (No model-specific language)
# ============================================================
CLINICAL_MESSAGES = {
    'severe_bradycardia': "Cardiac Risk: Severe Bradycardia",
    'moderate_bradycardia': "Cardiac Risk: Bradycardia Detected",
    'severe_tachycardia': "Cardiac Risk: Severe Tachycardia",
    'escalating_tachycardia': "Cardiac Risk: Escalating Heart Rate",
    'rapid_drop': "Cardiac Risk: Rapid Heart Rate Decline",
    'progressive_decline': "Cardiac Risk: Progressive Deterioration",
    'unstable_pattern': "Cardiac Risk: Unstable Vital Pattern",
    'general_anomaly': "Physiological Threshold Breach"
}


class CardiacRiskAnalyzer:
    """
    Analyzes heart rate patterns for physiologically significant cardiac risks.
    Returns clinical descriptions without model-specific labels.
    """
    
    @staticmethod
    def detect_severe_bradycardia(hr: float) -> Tuple[bool, float, str, str]:
        """
        Detect severe bradycardia (HR < 40 bpm).
        Returns: (detected, confidence, clinical_message, alert_type)
        """
        if hr < SEVERE_BRADYCARDIA_THRESHOLD:
            return True, 0.99, f"{CLINICAL_MESSAGES['severe_bradycardia']} (HR={hr:.0f})", "SEVERE_BRADYCARDIA"
        return False, 0.0, "", ""
    
    @staticmethod
    def detect_moderate_bradycardia(hr: float) -> Tuple[bool, float, str, str]:
        """Detect moderate bradycardia (40-50 bpm)."""
        if SEVERE_BRADYCARDIA_THRESHOLD <= hr < MODERATE_BRADYCARDIA_THRESHOLD:
            return True, 0.90, f"{CLINICAL_MESSAGES['moderate_bradycardia']} (HR={hr:.0f})", "MODERATE_BRADYCARDIA"
        return False, 0.0, "", ""
    
    @staticmethod
    def detect_severe_tachycardia(hr: float) -> Tuple[bool, float, str, str]:
        """Detect severe tachycardia (HR > 150 bpm)."""
        if hr > SEVERE_TACHYCARDIA_THRESHOLD:
            return True, 0.99, f"{CLINICAL_MESSAGES['severe_tachycardia']} (HR={hr:.0f})", "SEVERE_TACHYCARDIA"
        return False, 0.0, "", ""
    
    @staticmethod
    def detect_escalating_tachycardia(hr: float, hr_history: list) -> Tuple[bool, float, str, str]:
        """Detect escalating tachycardia (HR > 120 and rising)."""
        if hr > MODERATE_TACHYCARDIA_THRESHOLD and len(hr_history) >= 3:
            recent = hr_history[-3:]
            if all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
                return True, 0.95, f"{CLINICAL_MESSAGES['escalating_tachycardia']} (HR={hr:.0f})", "ESCALATING_TACHYCARDIA"
        return False, 0.0, "", ""
    
    @staticmethod
    def detect_rapid_drop(hr_history: list) -> Tuple[bool, float, str, str]:
        """Detect rapid heart rate drop (>15 bpm in 5 readings)."""
        if len(hr_history) >= 5:
            recent = hr_history[-5:]
            drop = recent[0] - recent[-1]
            if drop > RAPID_DROP_THRESHOLD:
                return True, 0.95, f"{CLINICAL_MESSAGES['rapid_drop']} ({drop:.0f} bpm drop)", "RAPID_DROP"
        return False, 0.0, "", ""
    
    @staticmethod
    def detect_progressive_decline(hr_history: list) -> Tuple[bool, float, str, str]:
        """Detect progressive HR decline (slope < -2 bpm/reading)."""
        if len(hr_history) >= 10:
            recent = hr_history[-10:]
            x = np.arange(len(recent))
            slope, _ = np.polyfit(x, recent, 1)
            
            if slope < PROGRESSIVE_DECLINE_SLOPE:
                return True, 0.93, f"{CLINICAL_MESSAGES['progressive_decline']} (trend: {slope:.1f}/reading)", "PROGRESSIVE_DECLINE"
        return False, 0.0, "", ""
    
    @staticmethod
    def detect_unstable_cardiac_pattern(hr: float, autocorr: float, 
                                         hr_history: list) -> Tuple[bool, float, str, str]:
        """Detect unstable cardiac pattern: low HR + low autocorrelation + negative trend."""
        if hr < MODERATE_BRADYCARDIA_THRESHOLD and autocorr < UNSTABLE_AUTOCORR_THRESHOLD:
            if len(hr_history) >= 5:
                recent = hr_history[-5:]
                x = np.arange(len(recent))
                slope, _ = np.polyfit(x, recent, 1)
                
                if slope < 0:
                    return True, 0.99, f"{CLINICAL_MESSAGES['unstable_pattern']} (HR={hr:.0f})", "UNSTABLE_PATTERN"
        return False, 0.0, "", ""
    
    @classmethod
    def analyze(cls, hr: float, autocorr: float, hr_history: list, 
                logistic_status: str, logistic_confidence: float) -> Tuple[str, float, str, str]:
        """
        Comprehensive cardiac risk analysis with priority-based override.
        
        Returns:
            Tuple of (status, confidence, clinical_message, alert_type)
        """
        # === PRIORITY 1: Unstable Cardiac Pattern ===
        detected, conf, msg, alert_type = cls.detect_unstable_cardiac_pattern(hr, autocorr, hr_history)
        if detected:
            return "Critical", conf, msg, alert_type
        
        # === PRIORITY 2: Severe Bradycardia ===
        detected, conf, msg, alert_type = cls.detect_severe_bradycardia(hr)
        if detected:
            return "Critical", conf, msg, alert_type
        
        # === PRIORITY 3: Severe Tachycardia ===
        detected, conf, msg, alert_type = cls.detect_severe_tachycardia(hr)
        if detected:
            return "Critical", conf, msg, alert_type
        
        # === PRIORITY 4: Rapid Drop ===
        detected, conf, msg, alert_type = cls.detect_rapid_drop(hr_history)
        if detected:
            return "Critical", conf, msg, alert_type
        
        # === PRIORITY 5: Progressive Decline ===
        detected, conf, msg, alert_type = cls.detect_progressive_decline(hr_history)
        if detected:
            return "Critical", conf, msg, alert_type
        
        # === PRIORITY 6: Escalating Tachycardia ===
        detected, conf, msg, alert_type = cls.detect_escalating_tachycardia(hr, hr_history)
        if detected:
            return "Critical", conf, msg, alert_type
        
        # === PRIORITY 7: Moderate Bradycardia ===
        detected, conf, msg, alert_type = cls.detect_moderate_bradycardia(hr)
        if detected:
            return "Critical", conf, msg, alert_type
        
        # === PRIORITY 8: ML-based detection (no model name exposed) ===
        if logistic_status == "Critical":
            return "Critical", logistic_confidence, CLINICAL_MESSAGES['general_anomaly'], "VITAL_ANOMALY"
        
        return logistic_status, logistic_confidence, "", ""


class MQTTListener:
    """MQTT listener with alert suppression and clinical messaging."""
    
    def __init__(self):
        self.client = None
        self.processor = get_processor()
        self.detector = get_detector()
        self.cardiac_analyzer = CardiacRiskAnalyzer()
        self.authenticated_devices = set()
        self.stats = {
            'messages': 0,
            'predictions': 0,
            'alerts_created': 0,
            'alerts_suppressed': 0,
            'auth_failures': 0
        }
        
        # Initialize database
        init_database()
        
        # Pre-register devices
        self._register_devices()
    
    def _register_devices(self):
        """Register known devices in database."""
        for device_id, token in VALID_TOKENS.items():
            patient_id = f"P{device_id.split('_')[1]}"
            register_device(device_id, token, patient_id)
        print(f"[MQTT] Registered {len(VALID_TOKENS)} devices")
    
    def _validate_device(self, device_id: str, token: str) -> bool:
        """Validate device authentication token."""
        if device_id in self.authenticated_devices:
            return True
        
        if device_id in VALID_TOKENS and VALID_TOKENS[device_id] == token:
            self.authenticated_devices.add(device_id)
            return True
        
        if validate_device_token(device_id, token):
            self.authenticated_devices.add(device_id)
            return True
        
        return False
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connect callback."""
        if rc == 0:
            print(f"[MQTT] Connected to {MQTT_BROKER}:{MQTT_PORT}")
            client.subscribe(TOPIC_VITALS)
            print(f"[MQTT] Subscribed to {TOPIC_VITALS}")
        else:
            print(f"[MQTT] Connection failed: {rc}")
    
    def _on_message(self, client, userdata, msg):
        """MQTT message callback."""
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            self._process_message(msg.topic, payload)
        except json.JSONDecodeError:
            print("[MQTT] Invalid JSON received")
        except Exception as e:
            print(f"[MQTT] Error processing message: {e}")
            import traceback
            traceback.print_exc()
    
    def _process_message(self, topic: str, payload: Dict):
        """Process incoming vital signs message with alert suppression."""
        self.stats['messages'] += 1
        
        # Extract patient ID from topic
        parts = topic.split('/')
        if len(parts) < 3:
            return
        
        patient_id = parts[2]
        
        # Extract fields
        device_id = payload.get('device_id')
        token = payload.get('token', '')
        
        # === DEVICE AUTHENTICATION ===
        if not self._validate_device(device_id, token):
            self.stats['auth_failures'] += 1
            print(f"[MQTT] AUTH FAILED: {device_id}")
            return
        
        # Extract vitals
        hr = float(payload.get('heart_rate', 0))
        spo2 = float(payload.get('spo2', 0))
        temp = float(payload.get('temperature', 0))
        timestamp = datetime.fromisoformat(payload.get('timestamp', datetime.now().isoformat()))
        
        # Add to buffer
        self.processor.add_reading(patient_id, timestamp, hr, spo2, temp)
        
        # Get buffer data
        buffer_len = self.processor.get_buffer_length(patient_id)
        hr_history = self.processor.get_hr_history(patient_id)
        
        # === SIGNAL PROCESSING ===
        hr_smoothed = moving_average(hr_history, window_size=5)
        autocorr = compute_autocorrelation(hr_history, lag=1) if len(hr_history) >= 3 else 0.0
        
        # === ML PREDICTION (internal, no labels exposed) ===
        logistic_status, logistic_confidence = self.detector.predict_status(hr, spo2, temp)
        
        # === LSTM PREDICTION ===
        predicted_hr = None
        if buffer_len >= MIN_BUFFER_SIZE:
            predicted_hr = self.detector.predict_next_hr(hr_history)
            self.stats['predictions'] += 1
        
        # === CARDIAC RISK ANALYSIS ===
        final_status, final_confidence, clinical_message, alert_type = self.cardiac_analyzer.analyze(
            hr=hr,
            autocorr=autocorr,
            hr_history=hr_history,
            logistic_status=logistic_status,
            logistic_confidence=logistic_confidence
        )
        
        # === DATABASE STORAGE ===
        insert_vital_reading(
            patient_id=patient_id,
            device_id=device_id,
            timestamp=timestamp,
            heart_rate=hr,
            spo2=spo2,
            temperature=temp,
            heart_rate_smoothed=hr_smoothed,
            predicted_next_hr=predicted_hr,
            logistic_confidence=final_confidence,
            autocorrelation=autocorr,
            status=final_status
        )
        
        # Update patient status
        update_patient_status(
            patient_id=patient_id,
            status=final_status,
            hr=hr,
            spo2=spo2,
            temperature=temp,
            predicted_hr=predicted_hr,
            confidence=final_confidence,
            autocorrelation=autocorr
        )
        
        # === ALERT GENERATION WITH SUPPRESSION ===
        if final_status == 'Critical' and alert_type:
            # Check if a similar alert already exists within the suppression window
            if not recent_alert_exists(patient_id, alert_type, ALERT_SUPPRESSION_WINDOW):
                # No recent alert exists - create new alert
                insert_alert(
                    patient_id=patient_id,
                    timestamp=timestamp,
                    alert_type=alert_type,
                    severity='HIGH',
                    message=clinical_message
                )
                self.stats['alerts_created'] += 1
                
                # Publish alert via MQTT
                alert_msg = {
                    'patient_id': patient_id,
                    'status': final_status,
                    'confidence': final_confidence,
                    'message': clinical_message,
                    'timestamp': timestamp.isoformat()
                }
                self.client.publish(TOPIC_ALERTS, json.dumps(alert_msg))
                
                print(f"[ALERT] {patient_id}: {clinical_message}")
            else:
                # Alert suppressed - already exists
                self.stats['alerts_suppressed'] += 1
        
        # Standard logging
        pred_str = f", PredHR={predicted_hr:.1f}" if predicted_hr else ""
        status_indicator = "⚠️" if final_status == "Critical" else "✓"
        print(f"[{patient_id}] {status_indicator} HR={hr:.1f}, SpO2={spo2:.1f}, Temp={temp:.1f} | "
              f"Status={final_status} (conf={final_confidence:.2f}){pred_str}")
    
    def start(self):
        """Start the MQTT listener."""
        self.client = mqtt.Client(client_id=f"listener_{int(time.time())}")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        
        try:
            print(f"[MQTT] Connecting to {MQTT_BROKER}:{MQTT_PORT}...")
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            print("[MQTT] Health Monitoring Network - Backend Active")
            print(f"[MQTT] Alert suppression window: {ALERT_SUPPRESSION_WINDOW}s")
            print("=" * 60)
            self.client.loop_forever()
        except ConnectionRefusedError:
            print(f"[MQTT] Cannot connect to broker. Is it running?")
        except KeyboardInterrupt:
            print("\n[MQTT] Shutting down...")
            self.stop()
    
    def stop(self):
        """Stop the listener."""
        if self.client:
            self.client.disconnect()
        print(f"\n[MQTT] Final Stats:")
        print(f"  Messages processed: {self.stats['messages']}")
        print(f"  HR predictions: {self.stats['predictions']}")
        print(f"  Alerts created: {self.stats['alerts_created']}")
        print(f"  Alerts suppressed: {self.stats['alerts_suppressed']}")
        print(f"  Auth failures: {self.stats['auth_failures']}")


def main():
    print("=" * 60)
    print("Health Monitoring Network - Backend Service")
    print("=" * 60)
    
    listener = MQTTListener()
    
    # Print model status
    model_status = listener.detector.get_model_status()
    print(f"Classification Model: {'✓' if model_status['logistic_loaded'] else '✗'}")
    print(f"Prediction Model: {'✓' if model_status['lstm_loaded'] else '✗'}")
    print("=" * 60)
    
    listener.start()


if __name__ == "__main__":
    main()