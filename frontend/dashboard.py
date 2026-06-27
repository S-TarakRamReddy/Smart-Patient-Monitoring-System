#streamlit run frontend/dashboard.py
"""
dashboard.py
Health Monitoring Network Dashboard
High-contrast dark theme for ICU-style clarity.
"""

import os
import sys
from datetime import datetime
import time

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import (
    init_database, get_all_patients, get_critical_patients,
    get_patient_vitals, get_patient_latest, get_active_alerts,
    get_active_alert_summary, get_latest_alert_for_patient
)

# Page config
st.set_page_config(
    page_title="Health Monitoring Network",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# High-contrast dark theme CSS
st.markdown("""
<style>
/* ============================================
   MODERN FLAT DARK THEME (SLATE)
=========================================== */
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    background-color: #0f172a;
    color: #f8fafc;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

.status-normal {
    background-color: rgba(16, 185, 129, 0.1);
    color: #10b981;
    padding: 12px 24px;
    border-radius: 8px;
    font-size: 1.1rem;
    font-weight: 600;
    text-align: center;
    border: 1px solid rgba(16, 185, 129, 0.2);
    display: inline-block;
}

.status-critical {
    background-color: rgba(239, 68, 68, 0.1);
    color: #ef4444;
    padding: 12px 24px;
    border-radius: 8px;
    font-size: 1.1rem;
    font-weight: 600;
    text-align: center;
    border: 1px solid rgba(239, 68, 68, 0.3);
    animation: pulse-red 2s infinite;
    display: inline-block;
}

@keyframes pulse-red {
    0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
    70% { box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }
    100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
}

.alert-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 9999px;
    font-weight: 500;
    font-size: 0.9rem;
    margin: 4px 0;
    background-color: #1e293b;
}

.alert-badge-critical { color: #fca5a5; background-color: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.08); }
.alert-badge-bradycardia { color: #c4b5fd; background-color: rgba(139, 92, 246, 0.15); border: 1px solid rgba(139, 92, 246, 0.2); }
.alert-badge-tachycardia { color: #fdba74; background-color: rgba(249, 115, 22, 0.05); border: 1px solid rgba(249, 115, 22, 0.2); }
.alert-badge-warning { color: #fcd34d; background-color: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.2); }
.alert-badge-decline { color: #f9a8d4; background-color: rgba(236, 72, 153, 0.15); border: 1px solid rgba(236, 72, 153, 0.2); }
.alert-badge-unstable { color: #f8fafc; background-color: rgba(71, 85, 105, 0.4); border: 1px solid #ef4444; }

.metric-box {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #334155;
    margin: 8px 0;
    display: flex;
    flex-direction: column;
}

.metric-label {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-bottom: 8px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metric-value-container {
    display: flex;
    align-items: baseline;
    gap: 6px;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1.1;
}

.metric-unit {
    font-size: 0.9rem;
    color: #64748b;
    font-weight: 500;
}

.prediction-box {
    background-color: rgba(14, 165, 233, 0.05);
    border: 1px solid rgba(14, 165, 233, 0.2);
    padding: 20px;
    border-radius: 12px;
    margin: 8px 0;
    display: flex;
    flex-direction: column;
}

.prediction-label {
    font-size: 0.85rem;
    color: #38bdf8;
    margin-bottom: 8px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.prediction-value {
    font-size: 2rem;
    font-weight: 700;
    color: #f0f9ff;
    line-height: 1.1;
}

.trend-container {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
}

.trend-up { color: #ef4444; font-size: 0.9rem; font-weight: 600; background: rgba(239, 68, 68, 0.1); padding: 2px 6px; border-radius: 4px; }
.trend-down { color: #10b981; font-size: 0.9rem; font-weight: 600; background: rgba(16, 185, 129, 0.1); padding: 2px 6px; border-radius: 4px; }
.trend-stable { color: #64748b; font-size: 0.9rem; font-weight: 500; }

.critical-card {
    background-color: #1e293b;
    border-left: 4px solid #ef4444;
    padding: 16px;
    border-radius: 8px;
    margin: 12px 0;
    border-top: 1px solid #334155;
    border-right: 1px solid #334155;
    border-bottom: 1px solid #334155;
}

.critical-card-header { font-weight: 600; color: #f8fafc; font-size: 1.05rem; margin-bottom: 8px; }
.critical-card-content { color: #cbd5e1; font-size: 0.9rem; line-height: 1.5; }
.critical-card-content strong { color: #f8fafc; }

.condition-panel { background-color: #1e293b; border-radius: 12px; padding: 24px; margin: 12px 0; border: 1px solid #334155; }
.condition-panel-critical { border-left: 4px solid #ef4444; }
.condition-panel-warning { border-left: 4px solid #f59e0b; }
.condition-panel-moderate { border-left: 4px solid #8b5cf6; }
.condition-panel-stable { border-left: 4px solid #10b981; }

.condition-panel-header { font-size: 1.15rem; font-weight: 600; color: #f8fafc; margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }
.condition-panel-content { color: #cbd5e1; font-size: 0.95rem; line-height: 1.6; }
.condition-panel-time { color: #64748b; font-size: 0.85rem; margin-top: 12px; font-weight: 500; }

.active-condition { background-color: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.08); padding: 8px 12px; border-radius: 6px; margin-top: 12px; }
.active-condition-text { color: #fca5a5; font-size: 0.85rem; font-weight: 500; }

.confidence-high { color: #10b981; }
.confidence-medium { color: #f59e0b; }
.confidence-low { color: #ef4444; }

.rhythm-stable { color: #10b981; font-weight: 600; }
.rhythm-variable { color: #f59e0b; font-weight: 600; }
.rhythm-irregular { color: #ef4444; font-weight: 600; }

.sidebar-alert-summary { background-color: rgba(239, 68, 68, 0.1); color: #ef4444; padding: 12px 16px; border-radius: 8px; font-weight: 600; font-size: 0.95rem; text-align: center; border: 1px solid rgba(239, 68, 68, 0.08); margin: 10px 0; }
.sidebar-no-alerts { background-color: rgba(16, 185, 129, 0.1); color: #10b981; padding: 12px 16px; border-radius: 8px; font-weight: 500; font-size: 0.95rem; text-align: center; border: 1px solid rgba(16, 185, 129, 0.2); margin: 10px 0; }

.footer-item { color: #64748b; font-size: 0.85rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


def safe_format(value, fmt=".1f", default="N/A"):
    """Safely format a value, returning default if None."""
    if value is None:
        return default
    try:
        return f"{value:{fmt}}"
    except (ValueError, TypeError):
        return default


def get_clinical_badge(message: str) -> str:
    """Generate high-contrast clinical alert badge."""
    if not message:
        return ""

    msg_lower = message.lower()

    if "unstable" in msg_lower:
        badge_class = "alert-badge alert-badge-unstable"
        icon = "⚠️"
    elif "bradycardia" in msg_lower:
        badge_class = "alert-badge alert-badge-bradycardia"
        icon = "💜"
    elif "tachycardia" in msg_lower:
        badge_class = "alert-badge alert-badge-tachycardia"
        icon = "🔥"
    elif "decline" in msg_lower or "drop" in msg_lower:
        badge_class = "alert-badge alert-badge-decline"
        icon = "📉"
    elif "threshold" in msg_lower or "breach" in msg_lower:
        badge_class = "alert-badge alert-badge-critical"
        icon = "🚨"
    else:
        badge_class = "alert-badge alert-badge-warning"
        icon = "⚠️"

    clean_message = message
    model_terms = ['logistic', 'lstm', 'model', 'ml', 'algorithm', 'classifier', 'neural']
    for term in model_terms:
        if term.lower() in message.lower():
            clean_message = "Cardiac Condition Alert"
            break

    return f'<span class="{badge_class}">{icon} {clean_message}</span>'


def get_trend_indicator(current_hr, predicted_hr) -> str:
    """Generate high-contrast trend arrow."""
    if current_hr is None or predicted_hr is None:
        return '<span class="trend-stable">→</span>'

    try:
        current = float(current_hr)
        predicted = float(predicted_hr)
        diff = predicted - current

        if diff > 3:
            return f'<span class="trend-up">↑ +{diff:.1f}</span>'
        elif diff < -3:
            return f'<span class="trend-down">↓ {diff:.1f}</span>'
        else:
            return f'<span class="trend-stable">→ {diff:+.1f}</span>'
    except (ValueError, TypeError):
        return '<span class="trend-stable">→</span>'


def get_confidence_class(confidence) -> str:
    """Get CSS class based on confidence level."""
    if confidence is None:
        return "confidence-low"
    try:
        conf = float(confidence)
        if conf >= 0.8:
            return "confidence-high"
        elif conf >= 0.5:
            return "confidence-medium"
        else:
            return "confidence-low"
    except (ValueError, TypeError):
        return "confidence-low"


def get_rhythm_class(autocorr) -> tuple:
    """Get rhythm status class and label."""
    if autocorr is None:
        return "rhythm-variable", "Unknown"
    try:
        ac = float(autocorr)
        if ac > 0.5:
            return "rhythm-stable", "Stable"
        elif ac > 0.2:
            return "rhythm-variable", "Variable"
        else:
            return "rhythm-irregular", "Irregular"
    except (ValueError, TypeError):
        return "rhythm-variable", "Unknown"


def render_sidebar():
    """Render high-contrast sidebar. Returns (selected_patient, auto_refresh)."""
    st.sidebar.title("🏥 Health Monitor")
    st.sidebar.markdown("---")

    # Get patients
    patients = get_all_patients()

    if not patients:
        st.sidebar.warning("No patients found")
        return None, False

    # Patient selector
    patient_options = {p['patient_id']: f"{p['patient_id']} - {p.get('name', 'Unknown')}"
                       for p in patients}

    selected = st.sidebar.selectbox(
        "Select Patient",
        options=list(patient_options.keys()),
        format_func=lambda x: patient_options.get(x, x)
    )

    st.sidebar.markdown("---")

    # Critical patients section
    st.sidebar.subheader("🚨 Active Conditions")

    critical = get_critical_patients()

    if critical:
        for p in critical:
            hr = p.get('last_hr')
            predicted = p.get('last_predicted_hr')

            hr_str = safe_format(hr, ".0f", "N/A")
            predicted_str = safe_format(predicted, ".0f", "N/A")
            trend = get_trend_indicator(hr, predicted)

            # Get latest alert for this patient
            latest_alert = get_latest_alert_for_patient(p.get('patient_id'))
            alert_msg = ""
            if latest_alert:
                msg = latest_alert.get('message', '')
                alert_msg = msg if msg else "Cardiac Condition"

            st.sidebar.markdown(f"""
            <div class="critical-card">
                <div class="critical-card-header">🔴 {p.get('patient_id', 'Unknown')}</div>
                <div class="critical-card-content">
                    HR: <strong>{hr_str}</strong> BPM {trend}<br>
                    Predicted: <strong>{predicted_str}</strong> BPM
                </div>
                {f'<div class="active-condition"><span class="active-condition-text">⚠️ {alert_msg}</span></div>' if alert_msg else ''}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div class="sidebar-no-alerts">
            ✓ All Patients Stable
        </div>
        """, unsafe_allow_html=True)

    st.sidebar.markdown("---")

    # Alert summary
    alert_summary = get_active_alert_summary()
    patient_count = alert_summary.get('patient_count', 0)

    if patient_count > 0:
        plural = "s" if patient_count > 1 else ""
        st.sidebar.markdown(f"""
        <div class="sidebar-alert-summary">
            🔔 {patient_count} Patient{plural} Require Attention
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div class="sidebar-no-alerts">
            🔕 No Active Conditions
        </div>
        """, unsafe_allow_html=True)

    st.sidebar.markdown("---")

    # ── FIX: checkbox only here, no sleep/rerun ──
    auto_refresh = st.sidebar.checkbox("Auto-refresh (3s)", value=True)

    return selected, auto_refresh


def render_status_indicator(status: str, alert_message: str = ""):
    """Render high-contrast status indicator."""
    col1, col2 = st.columns([1, 2])

    with col1:
        if status == 'Critical':
            st.markdown('<div class="status-critical">🚨 ATTENTION REQUIRED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-normal">✓ STABLE</div>', unsafe_allow_html=True)

    with col2:
        if alert_message:
            st.markdown(get_clinical_badge(alert_message), unsafe_allow_html=True)


def render_metrics(patient_data: dict):
    """Render modern vital sign metrics."""
    hr = patient_data.get('last_hr')
    spo2 = patient_data.get('last_spo2')
    temp = patient_data.get('last_temperature')
    predicted_hr = patient_data.get('last_predicted_hr')
    confidence = patient_data.get('last_confidence')
    autocorr = patient_data.get('last_autocorrelation')

    hr_str = safe_format(hr, ".1f", "N/A")
    spo2_str = safe_format(spo2, ".1f", "N/A")
    temp_str = safe_format(temp, ".2f", "N/A")
    predicted_hr_str = safe_format(predicted_hr, ".1f", "N/A")

    if confidence is not None:
        try:
            confidence_pct = float(confidence) * 100
            confidence_str = f"{confidence_pct:.1f}"
        except (ValueError, TypeError):
            confidence_str = "N/A"
    else:
        confidence_str = "N/A"

    trend = get_trend_indicator(hr, predicted_hr)
    conf_class = get_confidence_class(confidence)
    rhythm_class, rhythm_label = get_rhythm_class(autocorr)

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">❤️ Heart Rate</div>
            <div class="metric-value-container">
                <div class="metric-value">{hr_str}</div>
                <div class="metric-unit">BPM</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="prediction-box">
            <div class="prediction-label">🔮 Predicted HR</div>
            <div class="metric-value-container">
                <div class="prediction-value">{predicted_hr_str}</div>
                <div class="metric-unit">BPM</div>
            </div>
            <div class="trend-container">{trend}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">🫁 SpO2</div>
            <div class="metric-value-container">
                <div class="metric-value">{spo2_str}</div>
                <div class="metric-unit">%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">🌡️ Temperature</div>
            <div class="metric-value-container">
                <div class="metric-value">{temp_str}</div>
                <div class="metric-unit">°C</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">📊 Confidence</div>
            <div class="metric-value-container">
                <div class="metric-value {conf_class}">{confidence_str}</div>
                <div class="metric-unit">%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">📈 Rhythm</div>
            <div class="metric-value-container">
                <div class="metric-value {rhythm_class}">{rhythm_label}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_hr_chart(vitals: list, patient_id: str):
    """Render heart rate time-series chart with dark theme."""
    if not vitals:
        st.info("No vital signs data available")
        return

    df = pd.DataFrame(vitals)
    df = df.sort_values('timestamp')

    if 'heart_rate' not in df.columns or df['heart_rate'].dropna().empty:
        st.info("No heart rate data available")
        return

    fig = make_subplots(rows=1, cols=1)

    # Raw HR
    hr_data = df.dropna(subset=['heart_rate'])
    if not hr_data.empty:
        fig.add_trace(go.Scatter(
            x=hr_data['timestamp'],
            y=hr_data['heart_rate'],
            mode='lines',
            name='Heart Rate',
            line=dict(color='rgba(56, 189, 248, 0.6)', width=1.5),
            hovertemplate='HR: %{y:.1f} BPM<br>Time: %{x}<extra></extra>'
        ))

    # Smoothed HR
    if 'heart_rate_smoothed' in df.columns:
        smoothed_data = df.dropna(subset=['heart_rate_smoothed'])
        if not smoothed_data.empty:
            fig.add_trace(go.Scatter(
                x=smoothed_data['timestamp'],
                y=smoothed_data['heart_rate_smoothed'],
                mode='lines',
                name='Smoothed',
                line=dict(color='#0ea5e9', width=3)
            ))

    # Predicted HR
    if 'predicted_next_hr' in df.columns:
        predicted_data = df.dropna(subset=['predicted_next_hr'])
        if not predicted_data.empty:
            fig.add_trace(go.Scatter(
                x=predicted_data['timestamp'],
                y=predicted_data['predicted_next_hr'],
                mode='lines+markers',
                name='Predicted',
                line=dict(color='#f59e0b', width=2.5, dash='dash'),
                marker=dict(size=6, color='#f59e0b')
            ))

    # Threshold zones with better visibility
    fig.add_hrect(y0=60, y1=100, fillcolor="rgba(16, 185, 129, 0.05)", line_width=0)
    fig.add_hrect(y0=50, y1=60, fillcolor="rgba(245, 158, 11, 0.05)", line_width=0)
    fig.add_hrect(y0=40, y1=50, fillcolor="rgba(249, 115, 22, 0.05)", line_width=0)
    fig.add_hrect(y0=0, y1=40, fillcolor="rgba(239, 68, 68, 0.08)", line_width=0)
    fig.add_hrect(y0=100, y1=120, fillcolor="rgba(245, 158, 11, 0.05)", line_width=0)
    fig.add_hrect(y0=120, y1=150, fillcolor="rgba(249, 115, 22, 0.05)", line_width=0)
    fig.add_hrect(y0=150, y1=250, fillcolor="rgba(239, 68, 68, 0.08)", line_width=0)

    # Threshold lines
    fig.add_hline(y=40, line_dash="dash", line_color="#EF4444", line_width=2,
                  annotation_text="Critical Low", annotation_font_color="#EF4444")
    fig.add_hline(y=150, line_dash="dash", line_color="#EF4444", line_width=2,
                  annotation_text="Critical High", annotation_font_color="#EF4444")

    # Dark theme layout
    fig.update_layout(
        title=dict(
            text=f"🫀 Cardiac Monitor - {patient_id}",
            font=dict(size=22, color='#E2E8F0')
        ),
        xaxis_title="Time",
        yaxis_title="Heart Rate (BPM)",
        height=480,
        template="plotly_dark",
        paper_bgcolor='rgba(15, 23, 42, 0)',
        plot_bgcolor='rgba(15, 23, 42, 0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='#E2E8F0')
        ),
        hovermode="x unified",
        xaxis=dict(gridcolor='rgba(71, 85, 105, 0.5)', color='#94A3B8'),
        yaxis=dict(gridcolor='rgba(71, 85, 105, 0.5)', color='#94A3B8')
    )

    hr_values = df['heart_rate'].dropna()
    if not hr_values.empty:
        hr_min = hr_values.min()
        hr_max = hr_values.max()
        y_min = max(30, hr_min - 15)
        y_max = min(200, hr_max + 15)
        fig.update_yaxes(range=[y_min, y_max])

    st.plotly_chart(fig, width='stretch')


def render_condition_status(patient_id: str):
    """Render high-contrast condition status panel."""
    latest_alert = get_latest_alert_for_patient(patient_id)

    if not latest_alert:
        st.markdown("""
        <div class="condition-panel condition-panel-stable">
            <div class="condition-panel-header">
                ✓ No Active Conditions
            </div>
            <div class="condition-panel-content">
                All vital signs within normal parameters.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    message = latest_alert.get('message', 'Condition detected')
    timestamp = latest_alert.get('timestamp', 'Unknown time')
    alert_type = latest_alert.get('alert_type', '')

    # Clean message
    clean_message = message
    model_terms = ['logistic', 'lstm', 'model', 'ml', 'algorithm']
    for term in model_terms:
        if term.lower() in message.lower():
            clean_message = "Cardiac condition requiring attention"
            break

    # Determine panel class based on severity
    msg_lower = message.lower()
    if "unstable" in msg_lower or "severe" in msg_lower:
        panel_class = "condition-panel-critical"
        icon = "🚨"
    elif "decline" in msg_lower or "drop" in msg_lower:
        panel_class = "condition-panel-warning"
        icon = "📉"
    elif "bradycardia" in msg_lower or "tachycardia" in msg_lower:
        panel_class = "condition-panel-moderate"
        icon = "💜"
    else:
        panel_class = "condition-panel-warning"
        icon = "⚠️"

    st.markdown(f"""
    <div class="condition-panel {panel_class}">
        <div class="condition-panel-header">
            {icon} Active Cardiac Condition
        </div>
        <div class="condition-panel-content">
            {clean_message}
        </div>
        <div class="condition-panel-time">
            🕐 Since: {timestamp}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_trend_analysis(vitals: list):
    """Render trend analysis section."""
    if not vitals or len(vitals) < 10:
        st.info("Collecting data for trend analysis...")
        return

    df = pd.DataFrame(vitals).sort_values('timestamp')

    if 'heart_rate' not in df.columns:
        return

    hr_values = df['heart_rate'].dropna().tolist()

    if len(hr_values) < 10:
        st.info("Insufficient data for trend analysis")
        return

    recent = hr_values[-10:]
    x = np.arange(len(recent))

    try:
        slope, intercept = np.polyfit(x, recent, 1)
    except Exception:
        return

    if slope < -2:
        trend_text = "📉 Declining"
        trend_color = "#EF4444"
    elif slope < -0.5:
        trend_text = "↘️ Slight Decline"
        trend_color = "#F97316"
    elif slope > 2:
        trend_text = "📈 Rising"
        trend_color = "#EF4444"
    elif slope > 0.5:
        trend_text = "↗️ Slight Rise"
        trend_color = "#F97316"
    else:
        trend_text = "➡️ Stable"
        trend_color = "#22C55E"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Trend", f"{slope:.2f} bpm/reading")

    with col2:
        st.markdown(f"<h4 style='color: {trend_color}; text-shadow: 0 0 10px {trend_color}40;'>{trend_text}</h4>", unsafe_allow_html=True)

    with col3:
        std = np.std(recent)
        variability = "High" if std > 10 else ("Medium" if std > 5 else "Low")
        var_color = "#EF4444" if std > 10 else ("#FBBF24" if std > 5 else "#22C55E")
        st.markdown(f"<div style='color: {var_color};'><strong>Variability:</strong> {variability}</div>", unsafe_allow_html=True)


def main():
    """Main dashboard function."""
    init_database()

    # Header
    st.title("🏥 Health Monitoring Network")
    st.caption("Real-time Patient Vital Signs Monitoring")
    st.markdown("---")

    # Sidebar — now returns (selected, auto_refresh) instead of sleeping internally
    selected_patient, auto_refresh = render_sidebar()

    if not selected_patient:
        st.warning("Please select a patient")
        if auto_refresh:
            time.sleep(3)
            st.rerun()
        return

    # Get patient data
    patient_data = get_patient_latest(selected_patient)
    vitals = get_patient_vitals(selected_patient, limit=200)

    if not patient_data:
        st.info(f"Waiting for data from patient {selected_patient}...")
        if auto_refresh:
            time.sleep(3)
            st.rerun()
        return

    # Get latest alert
    latest_alert = get_latest_alert_for_patient(selected_patient)
    alert_message = ""
    if latest_alert:
        alert_message = latest_alert.get('message', '')

    # Patient header
    patient_name = patient_data.get('name', selected_patient)
    st.subheader(f"Patient: {patient_name}")

    # Status indicator
    current_status = patient_data.get('current_status', 'Normal')
    render_status_indicator(current_status, alert_message)

    st.markdown("---")

    # Metrics
    render_metrics(patient_data)

    st.markdown("---")

    # Main chart
    render_hr_chart(vitals, selected_patient)

    # Additional analysis
    col1, col2 = st.columns(2)

    with col1:
        with st.expander("📊 Trend Analysis", expanded=True):
            render_trend_analysis(vitals)

    with col2:
        with st.expander("🩺 Condition Status", expanded=True):
            render_condition_status(selected_patient)

    # Footer
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    last_update = patient_data.get('last_update', 'N/A')
    vitals_count = len(vitals) if vitals else 0

    with col1:
        st.markdown(f'<span class="footer-item">📅 Last update: {last_update}</span>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<span class="footer-item">📊 Data points: {vitals_count}</span>', unsafe_allow_html=True)

    with col3:
        buffer_status = "Ready" if vitals_count >= 30 else f"Collecting ({vitals_count}/30)"
        st.markdown(f'<span class="footer-item">🔮 Prediction: {buffer_status}</span>', unsafe_allow_html=True)

    with col4:
        st.markdown(f'<span class="footer-item">🕐 {datetime.now().strftime("%H:%M:%S")}</span>', unsafe_allow_html=True)

    # ── THE ONLY REAL CHANGE: sleep + rerun live here at the bottom of main(),
    #    after the full page has rendered. When the checkbox is unchecked,
    #    auto_refresh=False and this block is skipped — loop stops immediately. ──
    if auto_refresh:
        time.sleep(3)
        st.rerun()


if __name__ == "__main__":
    main()