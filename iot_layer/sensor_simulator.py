#python iot_layer/sensor_simulator.py
"""
sensor_simulator.py
IoT sensor simulator with device authentication tokens.
Generates multi-patient vital signs at 1 Hz.
"""

import os
import json
import time
import math
import random
from datetime import datetime

import paho.mqtt.client as mqtt

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuration
MQTT_BROKER = os.environ.get('MQTT_BROKER', 'localhost')
MQTT_PORT = int(os.environ.get('MQTT_PORT', 1883))

TOPIC_VITALS = "hospital/vitals/{patient_id}"

# Device authentication tokens - MUST match mqtt_listener.py
DEVICE_TOKENS = {
    "sensor_001": os.environ.get("SENSOR_001_TOKEN", "secure_token_001"),
    "sensor_002": os.environ.get("SENSOR_002_TOKEN", "secure_token_002"),
    "sensor_003": os.environ.get("SENSOR_003_TOKEN", "secure_token_003"),
    "sensor_004": os.environ.get("SENSOR_004_TOKEN", "secure_token_004"),
    "sensor_005": os.environ.get("SENSOR_005_TOKEN", "secure_token_005")
}

# Patient configurations
PATIENTS = {
    "P001": {
        "device_id": "sensor_001",
        "baseline_hr": 72,
        "baseline_spo2": 98,
        "baseline_temp": 37.0,
        "condition": "normal"
    },
    "P002": {
        "device_id": "sensor_002",
        "baseline_hr": 85,
        "baseline_spo2": 96,
        "baseline_temp": 37.3,
        "condition": "normal"
    },
    "P003": {
        "device_id": "sensor_003",
        "baseline_hr": 65,
        "baseline_spo2": 97,
        "baseline_temp": 36.8,
        "condition": "normal"
    },
    "P004": {
        "device_id": "sensor_004",
        "baseline_hr": 95,
        "baseline_spo2": 94,
        "baseline_temp": 38.2,
        "condition": "tachycardia"
    },
    "P005": {
        "device_id": "sensor_005",
        "baseline_hr": 55,
        "baseline_spo2": 99,
        "baseline_temp": 36.5,
        "condition": "bradycardia"
    }
}


class VitalGenerator:
    """Generate realistic vital signs for a patient."""
    
    def __init__(self, config: dict):
        self.config = config
        self.time = 0
        self.phase = random.uniform(0, 2 * math.pi)
        self.event = None
        self.event_duration = 0
    
    def _generate_event(self):
        """Randomly trigger physiological events."""
        if self.event:
            self.event_duration -= 1
            if self.event_duration <= 0:
                self.event = None
            return
        
        # 0.5% chance per reading
        if random.random() < 0.005:
            events = ['exertion', 'anxiety', 'fever']
            self.event = random.choice(events)
            self.event_duration = random.randint(20, 60)
    
    def generate(self) -> dict:
        """Generate a vital signs reading."""
        self.time += 1
        self._generate_event()
        
        # Base values
        hr = self.config['baseline_hr']
        spo2 = self.config['baseline_spo2']
        temp = self.config['baseline_temp']
        
        # Respiratory variation
        hr += 3 * math.sin(0.25 * self.time + self.phase)
        
        # Condition modifiers
        if self.config['condition'] == 'tachycardia':
            hr += 15
        elif self.config['condition'] == 'bradycardia':
            hr -= 10
        
        # Event modifiers
        if self.event == 'exertion':
            hr += 25 + random.uniform(0, 10)
            spo2 -= random.uniform(0, 2)
        elif self.event == 'anxiety':
            hr += 15 + random.uniform(0, 5)
        elif self.event == 'fever':
            temp += 1.5 + random.uniform(0, 0.5)
            hr += 10
        
        # Add noise
        hr += random.gauss(0, 2)
        spo2 += random.gauss(0, 0.3)
        temp += random.gauss(0, 0.05)
        
        # Clamp values
        hr = max(35, min(180, hr))
        spo2 = max(80, min(100, spo2))
        temp = max(35, min(41, temp))
        
        return {
            'heart_rate': round(hr, 1),
            'spo2': round(spo2, 1),
            'temperature': round(temp, 2)
        }


class SensorSimulator:
    """Multi-patient sensor simulator."""
    
    def __init__(self):
        self.client = None
        self.generators = {}
        self.running = False
        
        # Initialize generators
        for patient_id, config in PATIENTS.items():
            self.generators[patient_id] = VitalGenerator(config)
    
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"[SIMULATOR] Connected to {MQTT_BROKER}:{MQTT_PORT}")
        else:
            print(f"[SIMULATOR] Connection failed: {rc}")
    
    def _publish_reading(self, patient_id: str):
        """Publish a reading for a patient."""
        config = PATIENTS[patient_id]
        device_id = config['device_id']
        token = DEVICE_TOKENS[device_id]
        
        # Generate vitals
        vitals = self.generators[patient_id].generate()
        
        # Create payload WITH authentication token
        payload = {
            'device_id': device_id,
            'token': token,  # Authentication token included
            'patient_id': patient_id,
            'heart_rate': vitals['heart_rate'],
            'spo2': vitals['spo2'],
            'temperature': vitals['temperature'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Publish
        topic = TOPIC_VITALS.format(patient_id=patient_id)
        self.client.publish(topic, json.dumps(payload), qos=1)
        
        print(f"[{patient_id}] HR={vitals['heart_rate']:.1f}, "
              f"SpO2={vitals['spo2']:.1f}, Temp={vitals['temperature']:.2f}")
    
    def run(self):
        """Run the simulation loop."""
        self.client = mqtt.Client(client_id=f"simulator_{int(time.time())}")
        self.client.on_connect = self._on_connect
        
        try:
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.client.loop_start()
            
            time.sleep(1)  # Wait for connection
            
            self.running = True
            print("[SIMULATOR] Starting 1 Hz simulation...")
            print("=" * 60)
            
            while self.running:
                loop_start = time.time()
                
                # Publish for all patients
                for patient_id in PATIENTS:
                    self._publish_reading(patient_id)
                
                # Maintain 1 Hz
                elapsed = time.time() - loop_start
                sleep_time = max(0, 1.0 - elapsed)
                time.sleep(sleep_time)
                
        except ConnectionRefusedError:
            print(f"[SIMULATOR] Cannot connect to broker at {MQTT_BROKER}:{MQTT_PORT}")
        except KeyboardInterrupt:
            print("\n[SIMULATOR] Stopping...")
        finally:
            self.running = False
            if self.client:
                self.client.loop_stop()
                self.client.disconnect()
            print("[SIMULATOR] Stopped")


def main():
    print("=" * 60)
    print("Hospital IoT Sensor Simulator")
    print("=" * 60)
    print(f"Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print(f"Patients: {len(PATIENTS)}")
    print("=" * 60)
    
    simulator = SensorSimulator()
    simulator.run()


if __name__ == "__main__":
    main()