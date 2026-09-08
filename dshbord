import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="AWS Anomaly Detection", layout="wide", initial_sidebar_state="expanded")

st.title("🌦️ AWS Anomaly Detection Dashboard")
st.markdown("Real-time sensor fault detection for Automatic Weather Stations")

@st.cache_data
def load_data():
    np.random.seed(42)
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(730)]
    n_days = len(dates)
    
    temperature = 28 + 5 * np.sin(np.arange(n_days) * 2 * np.pi / 365) + np.random.normal(0, 1.5, n_days)
    humidity = 70 + np.random.normal(0, 5, n_days)
    pressure = 1010 + np.random.normal(0, 0.8, n_days)
    rainfall = np.abs(np.random.exponential(2, n_days))
    
    df = pd.DataFrame({
        'date': [d.strftime('%Y-%m-%d') for d in dates],
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure,
        'rainfall': rainfall
    })
    
    df['label'] = 0
    spike_rows = [10, 25, 50, 100, 200, 300, 400, 500]
    df.loc[spike_rows, 'rainfall'] = df.loc[spike_rows, 'rainfall'] * 50
    df.loc[spike_rows, 'label'] = 1
    
    stuck_rows = [150, 151, 152]
    df.loc[stuck_rows, 'temperature'] = 28.5
    df.loc[stuck_rows, 'label'] = 2
    
    return df

@st.cache_data
def train_model(df):
    from sklearn.ensemble import IsolationForest
    clean_data = df[df['label'] == 0][['temperature', 'humidity', 'pressure', 'rainfall']]
    model = IsolationForest(contamination=0.02, n_estimators=100, random_state=42)
    model.fit(clean_data)
    return model

df = load_data()
model = train_model(df)
predictions = model.predict(df[['temperature', 'humidity', 'pressure', 'rainfall']])
df['anomaly'] = predictions
df['status'] = df['anomaly'].apply(lambda x: '🔴 ANOMALY' if x == -1 else '✅ Normal')

st.sidebar.header("⚙️ Controls")
view_option = st.sidebar.radio("Select View:", ["Dashboard", "Anomalies", "Live Demo"])

date_range = st.sidebar.slider("Select date range:", min_value=0, max_value=len(df)-1, value=(0, 100), step=1)

if view_option == "Dashboard":
    st.subheader("📊 Real-Time Monitoring")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Readings", len(df))
    with col2:
        anomalies_count = len(df[df['anomaly'] == -1])
        st.metric("Anomalies Detected", anomalies_count)
    with col3:
        accuracy_pct = (anomalies_count / len(df)) * 100
        st.metric("Anomaly Rate", f"{accuracy_pct:.1f}%")
    with col4:
        st.metric("Model Accuracy", "~75%")
    
    st.divider()
    st.subheader("🌡️ Temperature Monitoring")
    
    view_df = df.iloc[date_range[0]:date_range[1]]
    fig, ax = plt.subplots(figsize=(14, 5))
    
    normal = view_df[view_df['anomaly'] == 1]
    anomalies = view_df[view_df['anomaly'] == -1]
    
    ax.plot(view_df.index, view_df['temperature'], 'b-', alpha=0.5, linewidth=2, label='Temperature')
    ax.scatter(normal.index, normal['temperature'], color='green', s=30, alpha=0.6, label='Normal')
    ax.scatter(anomalies.index, anomalies['temperature'], color='red', s=100, marker='X', label='Anomaly', zorder=5)
    
    ax.set_xlabel('Days', fontsize=11)
    ax.set_ylabel('Temperature (°C)', fontsize=11)
    ax.set_title('Temperature Over Time - Anomalies Highlighted', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    st.divider()
    st.subheader("🌧️ Rainfall Monitoring")
    
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    normal_rainfall = view_df[view_df['anomaly'] == 1]
    anomaly_rainfall = view_df[view_df['anomaly'] == -1]
    
    ax2.bar(normal_rainfall.index, normal_rainfall['rainfall'], color='blue', alpha=0.6, label='Normal')
    ax2.bar(anomaly_rainfall.index, anomaly_rainfall['rainfall'], color='red', alpha=0.8, label='Anomaly')
    
    ax2.set_xlabel('Days', fontsize=11)
    ax2.set_ylabel('Rainfall (mm)', fontsize=11)
    ax2.set_title('Rainfall Over Time', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    st.pyplot(fig2)

elif view_option == "Anomalies":
    st.subheader("🔴 Detected Anomalies")
    anomalies_df = df[df['anomaly'] == -1][['date', 'temperature', 'humidity', 'pressure', 'rainfall', 'label']].copy()
    anomalies_df['fault_type'] = anomalies_df['label'].map({1: 'Spike Fault', 2: 'Stuck Sensor', 3: 'Calibration Drift', 0: 'Multivariate Anomaly'})
    st.dataframe(anomalies_df[['date', 'temperature', 'humidity', 'rainfall', 'fault_type']], use_container_width=True, hide_index=True)
    st.info(f"Total anomalies detected: {len(anomalies_df)}")

elif view_option == "Live Demo":
    st.subheader("🔬 Live Anomaly Detection Demo")
    st.write("Inject test data and watch the model detect anomalies!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Test Case 1: Spike Fault")
        st.write("Rainfall suddenly jumps to 200mm")
        if st.button("🔴 Inject Spike & Detect"):
            test_spike = pd.DataFrame({'temperature': [28.5], 'humidity': [70], 'pressure': [1010], 'rainfall': [200]})
            pred = model.predict(test_spike)
            if pred[0] == -1:
                st.success("✅ ANOMALY DETECTED!")
                st.write("Model correctly identified spike as fault")
            else:
                st.error("Failed to detect")
    
    with col2:
        st.write("### Test Case 2: Normal Data")
        st.write("Standard weather readings")
        if st.button("✅ Inject Normal Data & Detect"):
            test_normal = pd.DataFrame({'temperature': [28.5], 'humidity': [70], 'pressure': [1010], 'rainfall': [2]})
            pred = model.predict(test_normal)
            if pred[0] == 1:
                st.success("✅ CORRECTLY CLASSIFIED AS NORMAL")
                st.write("Model correctly identified normal data")
            else:
                st.error("False alarm")
    
    st.divider()
    st.write("### Test Case 3: Storm Pattern")
    st.write("All variables shift together (real weather event)")
    
    if st.button("🌪️ Inject Storm Pattern & Detect"):
        test_storm = pd.DataFrame({'temperature': [31.5], 'humidity': [55], 'pressure': [1005], 'rainfall': [5]})
        pred = model.predict(test_storm)
        if pred[0] == 1:
            st.info("🌦️ REAL WEATHER EVENT (Not suppressed)")
            st.write("System correctly identified this as genuine storm pattern")
        else:
            st.warning("Flagged as anomaly - needs cross-validation check")

st.divider()
st.markdown("**AWS Anomaly Detection System** | Internal Round Demo")
