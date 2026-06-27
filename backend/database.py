"""
database.py
SQLite persistence with alert suppression support.
"""

import sqlite3
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import threading
import json

# Database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'patient_monitoring.db')

# Thread-local storage
_local = threading.local()


def get_connection():
    """Get thread-local database connection."""
    if not hasattr(_local, 'connection') or _local.connection is None:
        _local.connection = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.connection.row_factory = sqlite3.Row
    return _local.connection


def init_database():
    """Initialize database schema with all required columns."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Vital signs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vital_signs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            device_id TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            heart_rate REAL,
            spo2 REAL,
            temperature REAL,
            heart_rate_smoothed REAL,
            predicted_next_hr REAL,
            logistic_confidence REAL,
            autocorrelation REAL,
            status TEXT DEFAULT 'Normal',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Patients table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT UNIQUE NOT NULL,
            name TEXT,
            room_number TEXT,
            current_status TEXT DEFAULT 'Normal',
            last_hr REAL,
            last_spo2 REAL,
            last_temperature REAL,
            last_predicted_hr REAL,
            last_confidence REAL,
            last_autocorrelation REAL,
            last_update DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Alerts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT,
            acknowledged INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Devices table for authentication
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS devices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT UNIQUE NOT NULL,
            auth_token TEXT NOT NULL,
            patient_id TEXT,
            is_active INTEGER DEFAULT 1,
            last_seen DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_vitals_patient ON vital_signs(patient_id, timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_vitals_timestamp ON vital_signs(timestamp DESC)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_patient ON alerts(patient_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp DESC)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(patient_id, alert_type, timestamp)')
    
    conn.commit()
    print(f"[DATABASE] Initialized at {DB_PATH}")
    
    # Migrate existing tables if needed
    _migrate_schema(cursor, conn)


def _migrate_schema(cursor, conn):
    """Add new columns to existing tables if they don't exist."""
    columns_to_add = {
        'vital_signs': [
            ('predicted_next_hr', 'REAL'),
            ('logistic_confidence', 'REAL'),
            ('autocorrelation', 'REAL'),
            ('heart_rate_smoothed', 'REAL'),
            ('status', 'TEXT DEFAULT "Normal"')
        ],
        'patients': [
            ('last_predicted_hr', 'REAL'),
            ('last_confidence', 'REAL'),
            ('last_autocorrelation', 'REAL')
        ]
    }
    
    for table, columns in columns_to_add.items():
        for col_name, col_type in columns:
            try:
                cursor.execute(f'ALTER TABLE {table} ADD COLUMN {col_name} {col_type}')
                conn.commit()
                print(f"[DATABASE] Added column {col_name} to {table}")
            except sqlite3.OperationalError:
                pass  # Column already exists


def recent_alert_exists(patient_id: str, alert_type: str, window_seconds: int = 60) -> bool:
    """
    Check if a similar alert already exists within the time window.
    Prevents duplicate alert spam for persistent conditions.
    
    Args:
        patient_id: Patient identifier
        alert_type: Type of alert (e.g., 'CARDIAC_RISK', 'VITAL_ANOMALY')
        window_seconds: Time window to check for existing alerts
        
    Returns:
        True if a recent similar alert exists, False otherwise
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Calculate the cutoff time
    cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
    
    cursor.execute('''
        SELECT COUNT(*) as count FROM alerts
        WHERE patient_id = ?
        AND alert_type = ?
        AND timestamp >= ?
        AND acknowledged = 0
    ''', (patient_id, alert_type, cutoff_time.isoformat()))
    
    row = cursor.fetchone()
    return row['count'] > 0 if row else False


def get_unique_active_alert_patients() -> List[str]:
    """
    Get list of unique patient IDs with active (unacknowledged) alerts.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT DISTINCT patient_id FROM alerts
        WHERE acknowledged = 0
        ORDER BY patient_id
    ''')
    
    return [row['patient_id'] for row in cursor.fetchall()]


def get_active_alert_summary() -> Dict:
    """
    Get summary of active alerts - unique patients and alert types.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Count unique patients with alerts
    cursor.execute('''
        SELECT COUNT(DISTINCT patient_id) as patient_count FROM alerts
        WHERE acknowledged = 0
    ''')
    patient_count = cursor.fetchone()['patient_count']
    
    # Count unique alert types per patient
    cursor.execute('''
        SELECT patient_id, COUNT(DISTINCT alert_type) as type_count
        FROM alerts
        WHERE acknowledged = 0
        GROUP BY patient_id
    ''')
    patient_alerts = {row['patient_id']: row['type_count'] for row in cursor.fetchall()}
    
    # Get most recent alert per patient
    cursor.execute('''
        SELECT a.* FROM alerts a
        INNER JOIN (
            SELECT patient_id, MAX(timestamp) as max_ts
            FROM alerts
            WHERE acknowledged = 0
            GROUP BY patient_id
        ) b ON a.patient_id = b.patient_id AND a.timestamp = b.max_ts
        WHERE a.acknowledged = 0
    ''')
    recent_alerts = [dict(row) for row in cursor.fetchall()]
    
    return {
        'patient_count': patient_count,
        'patient_alerts': patient_alerts,
        'recent_alerts': recent_alerts
    }


def get_latest_alert_for_patient(patient_id: str) -> Optional[Dict]:
    """
    Get the most recent unacknowledged alert for a patient.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM alerts
        WHERE patient_id = ? AND acknowledged = 0
        ORDER BY timestamp DESC
        LIMIT 1
    ''', (patient_id,))
    
    row = cursor.fetchone()
    return dict(row) if row else None


def acknowledge_patient_alerts(patient_id: str) -> int:
    """
    Acknowledge all alerts for a patient.
    Returns the number of alerts acknowledged.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE alerts
        SET acknowledged = 1
        WHERE patient_id = ? AND acknowledged = 0
    ''', (patient_id,))
    
    conn.commit()
    return cursor.rowcount


def clear_resolved_alerts(patient_id: str) -> int:
    """
    Mark alerts as acknowledged when patient status returns to Normal.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE alerts
        SET acknowledged = 1
        WHERE patient_id = ? AND acknowledged = 0
    ''', (patient_id,))
    
    conn.commit()
    return cursor.rowcount


def insert_vital_reading(patient_id: str, device_id: str, timestamp: datetime,
                         heart_rate: float, spo2: float, temperature: float,
                         heart_rate_smoothed: float = None,
                         predicted_next_hr: float = None,
                         logistic_confidence: float = None,
                         autocorrelation: float = None,
                         status: str = 'Normal') -> int:
    """Insert a vital signs reading with all computed values."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO vital_signs 
        (patient_id, device_id, timestamp, heart_rate, spo2, temperature,
         heart_rate_smoothed, predicted_next_hr, logistic_confidence, 
         autocorrelation, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (patient_id, device_id, timestamp, heart_rate, spo2, temperature,
          heart_rate_smoothed, predicted_next_hr, logistic_confidence,
          autocorrelation, status))
    
    conn.commit()
    return cursor.lastrowid


def update_patient_status(patient_id: str, status: str, hr: float, spo2: float,
                          temperature: float, predicted_hr: float = None,
                          confidence: float = None, autocorrelation: float = None):
    """Update patient's current status and latest values."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Upsert patient record
    cursor.execute('''
        INSERT INTO patients (patient_id, name, current_status, last_hr, last_spo2,
                             last_temperature, last_predicted_hr, last_confidence,
                             last_autocorrelation, last_update)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(patient_id) DO UPDATE SET
            current_status = excluded.current_status,
            last_hr = excluded.last_hr,
            last_spo2 = excluded.last_spo2,
            last_temperature = excluded.last_temperature,
            last_predicted_hr = excluded.last_predicted_hr,
            last_confidence = excluded.last_confidence,
            last_autocorrelation = excluded.last_autocorrelation,
            last_update = excluded.last_update
    ''', (patient_id, f"Patient {patient_id}", status, hr, spo2, temperature,
          predicted_hr, confidence, autocorrelation, datetime.now()))
    
    conn.commit()
    
    # If status is Normal, clear any existing alerts (condition resolved)
    if status == 'Normal':
        clear_resolved_alerts(patient_id)


def insert_alert(patient_id: str, timestamp: datetime, alert_type: str,
                 severity: str, message: str) -> int:
    """Insert an alert record."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO alerts (patient_id, timestamp, alert_type, severity, message)
        VALUES (?, ?, ?, ?, ?)
    ''', (patient_id, timestamp, alert_type, severity, message))
    
    conn.commit()
    return cursor.lastrowid


def register_device(device_id: str, auth_token: str, patient_id: str = None) -> bool:
    """Register a device with authentication token."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO devices (device_id, auth_token, patient_id, is_active, last_seen)
            VALUES (?, ?, ?, 1, ?)
        ''', (device_id, auth_token, patient_id, datetime.now()))
        conn.commit()
        return True
    except Exception as e:
        print(f"[DATABASE] Error registering device: {e}")
        return False


def validate_device_token(device_id: str, token: str) -> bool:
    """Validate device authentication token."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT auth_token, is_active FROM devices WHERE device_id = ?
    ''', (device_id,))
    
    row = cursor.fetchone()
    if row and row['auth_token'] == token and row['is_active']:
        # Update last seen
        cursor.execute('UPDATE devices SET last_seen = ? WHERE device_id = ?',
                       (datetime.now(), device_id))
        conn.commit()
        return True
    return False


def get_all_patients() -> List[Dict]:
    """Get all registered patients."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM patients ORDER BY patient_id')
    return [dict(row) for row in cursor.fetchall()]


def get_critical_patients() -> List[Dict]:
    """Get patients with Critical status."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM patients 
        WHERE current_status = 'Critical' 
        ORDER BY last_update DESC
    ''')
    return [dict(row) for row in cursor.fetchall()]


def get_patient_vitals(patient_id: str, limit: int = 100) -> List[Dict]:
    """Get recent vital signs for a patient."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM vital_signs 
        WHERE patient_id = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (patient_id, limit))
    return [dict(row) for row in cursor.fetchall()]


def get_patient_latest(patient_id: str) -> Optional[Dict]:
    """Get latest record for a patient."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM patients WHERE patient_id = ?
    ''', (patient_id,))
    row = cursor.fetchone()
    return dict(row) if row else None


def get_active_alerts(limit: int = 50) -> List[Dict]:
    """Get unacknowledged alerts."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM alerts 
        WHERE acknowledged = 0 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (limit,))
    return [dict(row) for row in cursor.fetchall()]


if __name__ == "__main__":
    init_database()
    print("[DATABASE] Setup complete")