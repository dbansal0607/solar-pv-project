"""
Solar PV Digital Twin - Streamlit Dashboard
Glass-themed UI with live simulation and API integration

Author: [Your Name]
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import io
import time
import matplotlib.pyplot as plt
import altair as alt
from datetime import datetime
import joblib

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Solar PV Digital Twin",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# GLASS-THEMED CSS STYLING
# ============================================================================
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background with gradient */
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e8ba3 100%);
    }
    
    /* Glass effect for main containers */
    .glass-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    /* Yellow solar accent */
    .solar-accent {
        color: #FFD700;
        font-weight: 600;
    }
    
    /* Metric cards with glass effect */
    .metric-card {
        background: rgba(255, 215, 0, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 2px solid rgba(255, 215, 0, 0.3);
        text-align: center;
        box-shadow: 0 4px 16px rgba(255, 215, 0, 0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FFD700;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.8);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 10px;
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px 20px;
        color: rgba(255, 255, 255, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.2);
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 215, 0, 0.3);
        color: #FFD700;
        border: 2px solid #FFD700;
        font-weight: 600;
    }
    
    /* Title styling */
    h1 {
        color: #FFD700;
        text-shadow: 0 2px 10px rgba(255, 215, 0, 0.3);
        font-weight: 700;
        letter-spacing: -1px;
    }
    
    h2, h3 {
        color: rgba(255, 255, 255, 0.95);
        font-weight: 600;
    }
    
    /* Input fields */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 215, 0, 0.3);
        border-radius: 10px;
        color: white;
        padding: 10px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #1e3c72;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 215, 0, 0.6);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(30, 60, 114, 0.8);
        backdrop-filter: blur(10px);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: rgba(0, 255, 0, 0.1);
        border-left: 4px solid #00ff00;
    }
    
    .stError {
        background: rgba(255, 0, 0, 0.1);
        border-left: 4px solid #ff0000;
    }
    
    /* DataFrame styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================
st.markdown("<h1 style='text-align: center;'>🌞 Solar PV Digital Twin Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.7); font-size: 1.1rem;'>Real-time power prediction with ML-powered simulation</p>", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL LOCALLY FOR FEATURE IMPORTANCE
# ============================================================================
@st.cache_resource
def load_model_artifact():
    """Load model artifact for local operations"""
    try:
        artifact = joblib.load('models/pipeline_prod.joblib')
        return artifact
    except:
        return None

model_artifact = load_model_artifact()

# Load metrics
@st.cache_resource
def load_metrics():
    """Load metrics.json"""
    try:
        with open('models/metrics.json', 'r') as f:
            return json.load(f)
    except:
        return None

metrics = load_metrics()

# ============================================================================
# SIDEBAR - MODEL INFO
# ============================================================================
with st.sidebar:
    st.markdown("### 📊 Model Information")
    
    if metrics:
        st.markdown(f"**Version:** `{metrics['model_version']}`")
        st.markdown(f"**Test MAE:** {metrics['test_metrics']['mae']} W")
        st.markdown(f"**Test R²:** {metrics['test_metrics']['r2']}")
        st.markdown(f"**Training Date:** {metrics['training_date'][:10]}")
    else:
        st.warning("⚠️ Metrics file not found. Run training first.")
    
    st.markdown("---")
    st.markdown("### ⚙️ API Configuration")
    api_url = st.text_input("API Base URL", value="http://localhost:8000")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This dashboard demonstrates a **digital twin** of a solar PV panel system.
    
    **Features:**
    - 🎯 Real-time predictions
    - 📈 Live data simulation
    - 📁 Batch CSV processing
    - 🔍 Feature importance analysis
    """)

# ============================================================================
# TOP METRICS ROW
# ============================================================================
if metrics:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Test MAE</div>
            <div class="metric-value">{metrics['test_metrics']['mae']}</div>
            <div class="metric-label">Watts</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Test RMSE</div>
            <div class="metric-value">{metrics['test_metrics']['rmse']}</div>
            <div class="metric-label">Watts</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">R² Score</div>
            <div class="metric-value">{metrics['test_metrics']['r2']}</div>
            <div class="metric-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Training Samples</div>
            <div class="metric-value">{metrics['data_split']['train_samples']}</div>
            <div class="metric-label">Rows</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# MAIN TABS
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Manual Predictor", "📊 Feature Importance", "🔴 Live Simulation", "📁 Batch Predictions"])

# ============================================================================
# TAB 1: MANUAL PREDICTOR
# ============================================================================
with tab1:
    st.markdown("### 🎯 Single Prediction (API-Powered)")
    st.markdown("Enter environmental parameters and get instant power output predictions from the deployed model.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Environmental Parameters")
        solar_irr = st.number_input("Solar Irradiance (kWh/m²)", min_value=0.0, max_value=1.5, value=0.75, step=0.01)
        temp_c = st.number_input("Temperature (°C)", min_value=-20.0, max_value=50.0, value=25.0, step=0.5)
        wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=20.0, value=3.0, step=0.1)
        humidity = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
    
    with col2:
        st.markdown("#### Panel Configuration")
        panel_tilt = st.number_input("Panel Tilt (degrees)", min_value=0.0, max_value=90.0, value=30.0, step=1.0)
        panel_azimuth = st.number_input("Panel Azimuth (degrees)", min_value=0.0, max_value=360.0, value=180.0, step=1.0)
        poa_irr = st.number_input("POA Irradiance", min_value=0.0, max_value=1500.0, value=800.0, step=10.0)
        cell_temp = st.number_input("Cell Temperature (°C)", min_value=-20.0, max_value=80.0, value=35.0, step=0.5)
    
    if st.button("🚀 Predict Power Output", use_container_width=True):
        # Prepare request payload
        payload = {
            "Solar_Irradiance_kWh_m2": solar_irr,
            "Temperature_C": temp_c,
            "Wind_Speed_mps": wind_speed,
            "Relative_Humidity_%": humidity,
            "Panel_Tilt_deg": panel_tilt,
            "Panel_Azimuth_deg": panel_azimuth,
            "Plane_of_Array_Irradiance": poa_irr,
            "Cell_Temperature_C": cell_temp
        }
        
        try:
            with st.spinner("🔄 Calling API..."):
                response = requests.post(f"{api_url}/predict", json=payload, timeout=10)
                response.raise_for_status()
                
                result = response.json()
                
                # Display result in glass container
                st.markdown(f"""
                <div class="glass-container">
                    <h2 style='text-align: center; color: #FFD700;'>⚡ Predicted Power Output</h2>
                    <div style='text-align: center; font-size: 4rem; font-weight: 700; color: #FFD700; margin: 20px 0;'>
                        {result['predicted_power']:.2f} W
                    </div>
                    <div style='text-align: center; color: rgba(255,255,255,0.7);'>
                        Model Version: {result['model_version']}<br>
                        Timestamp: {result['timestamp'][:19]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.success("✅ Prediction successful!")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ **Connection Error**: Could not connect to API. Make sure the server is running:\n```\nuvicorn src.server:app --reload\n```")
        except requests.exceptions.Timeout:
            st.error("❌ **Timeout Error**: API request took too long. Check server health.")
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ **HTTP Error**: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            st.error(f"❌ **Error**: {str(e)}")

# ============================================================================
# TAB 2: FEATURE IMPORTANCE
# ============================================================================
with tab2:
    st.markdown("### 📊 Feature Importance Analysis")
    st.markdown("Understand which environmental factors have the most impact on power output predictions.")
    
    if metrics and 'feature_importance' in metrics:
        # Prepare data for chart
        importance_df = pd.DataFrame({
            'Feature': list(metrics['feature_importance'].keys()),
            'Importance': list(metrics['feature_importance'].values())
        }).sort_values('Importance', ascending=False)
        
        # Create Altair chart with solar theme
        chart = alt.Chart(importance_df).mark_bar(color='#FFD700').encode(
            x=alt.X('Importance:Q', title='Importance Score'),
            y=alt.Y('Feature:N', sort='-x', title='Feature'),
            tooltip=['Feature', 'Importance']
        ).properties(
            height=400,
            title='Feature Importance (RandomForest)'
        ).configure_axis(
            labelColor='white',
            titleColor='white'
        ).configure_title(
            color='#FFD700',
            fontSize=20
        ).configure_view(
            strokeWidth=0
        )
        
        st.altair_chart(chart, use_container_width=True)
        
        # Interpretation
        st.markdown("#### 💡 Interpretation")
        top_feature = importance_df.iloc[0]
        st.markdown(f"""
        <div class="glass-container">
            <p>The most important feature is <span class="solar-accent">{top_feature['Feature']}</span> 
            with an importance score of <span class="solar-accent">{top_feature['Importance']:.4f}</span>.</p>
            
            <p><strong>What this means:</strong> This feature has the strongest influence on the model's 
            power output predictions. Changes in this parameter will have the largest impact on predicted values.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Feature importance data not available. Train the model first.")

# ============================================================================
# TAB 3: LIVE SIMULATION
# ============================================================================
with tab3:
    st.markdown("### 🔴 Live Digital Twin Simulation")
    st.markdown("Watch real-time simulated sensor data and power predictions streaming live.")
    
    if model_artifact:
        # Initialize session state for simulation
        if 'simulation_running' not in st.session_state:
            st.session_state.simulation_running = False
        if 'simulation_data' not in st.session_state:
            st.session_state.simulation_data = []
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("▶️ Start Simulation" if not st.session_state.simulation_running else "⏸️ Stop Simulation"):
                st.session_state.simulation_running = not st.session_state.simulation_running
            
            if st.button("🔄 Reset Data"):
                st.session_state.simulation_data = []
        
        # Simulation loop
        if st.session_state.simulation_running:
            chart_placeholder = st.empty()
            metrics_placeholder = st.empty()
            
            for i in range(50):  # Run 50 iterations
                if not st.session_state.simulation_running:
                    break
                
                # Generate simulated sensor data
                simulated_data = {
                    'Solar_Irradiance_kWh_m2': np.random.uniform(0.3, 1.0),
                    'Temperature_C': np.random.uniform(15, 35),
                    'Wind_Speed_mps': np.random.uniform(1, 8),
                    'Relative_Humidity_%': np.random.uniform(30, 80),
                    'Panel_Tilt_deg': 30.0,
                    'Panel_Azimuth_deg': 180.0,
                    'Plane_of_Array_Irradiance': np.random.uniform(400, 1000),
                    'Cell_Temperature_C': np.random.uniform(25, 45)
                }
                
                # Make prediction
                input_df = pd.DataFrame([simulated_data])
                prediction = model_artifact['pipeline'].predict(input_df)[0]
                
                # Store data point
                st.session_state.simulation_data.append({
                    'timestamp': datetime.now(),
                    'power': prediction,
                    'irradiance': simulated_data['Solar_Irradiance_kWh_m2']
                })
                
                # Keep only last 30 points
                if len(st.session_state.simulation_data) > 30:
                    st.session_state.simulation_data.pop(0)
                
                # Create live chart
                if len(st.session_state.simulation_data) > 0:
                    sim_df = pd.DataFrame(st.session_state.simulation_data)
                    
                    # Power chart
                    power_chart = alt.Chart(sim_df.reset_index()).mark_line(
                        color='#FFD700',
                        strokeWidth=3,
                        point=alt.OverlayMarkDef(color='#FFA500', size=50)
                    ).encode(
                        x=alt.X('index:Q', title='Time Step'),
                        y=alt.Y('power:Q', title='Predicted Power (W)', scale=alt.Scale(zero=False)),
                        tooltip=['power']
                    ).properties(
                        height=300,
                        title='Real-Time Power Output'
                    )
                    
                    chart_placeholder.altair_chart(power_chart, use_container_width=True)
                    
                    # Current metrics
                    current_power = st.session_state.simulation_data[-1]['power']
                    avg_power = np.mean([d['power'] for d in st.session_state.simulation_data])
                    
                    col1, col2, col3 = metrics_placeholder.columns(3)
                    col1.metric("Current Power", f"{current_power:.2f} W", f"{current_power - avg_power:.2f} W")
                    col2.metric("Average Power", f"{avg_power:.2f} W")
                    col3.metric("Data Points", len(st.session_state.simulation_data))
                
                time.sleep(0.5)  # Update every 0.5 seconds
        
        else:
            st.info("▶️ Click 'Start Simulation' to begin live streaming.")
    
    else:
        st.error("❌ Model not loaded. Train the model first.")

# ============================================================================
# TAB 4: BATCH PREDICTIONS
# ============================================================================
with tab4:
    st.markdown("### 📁 Batch CSV Predictions")
    st.markdown("Upload a CSV file with multiple samples for bulk predictions via the API.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Display uploaded data preview
        df_upload = pd.read_csv(uploaded_file)
        st.markdown("#### 📋 Uploaded Data Preview (first 5 rows)")
        st.dataframe(df_upload.head())
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", len(df_upload))
        with col2:
            st.metric("Total Columns", len(df_upload.columns))
        
        # Predict button
        if st.button("🚀 Run Batch Prediction", use_container_width=True):
            try:
                with st.spinner("🔄 Sending to API for prediction..."):
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Send to API
                    files = {'file': ('uploaded.csv', uploaded_file, 'text/csv')}
                    params = {'return_csv': 'true'}
                    
                    response = requests.post(
                        f"{api_url}/predict_batch",
                        files=files,
                        params=params,
                        timeout=30
                    )
                    response.raise_for_status()
                    
                    # Parse response as CSV
                    result_df = pd.read_csv(io.StringIO(response.text))
                    
                    st.success(f"✅ Successfully predicted {len(result_df)} samples!")
                    
                    # Display results
                    st.markdown("#### 📊 Prediction Results")
                    st.dataframe(result_df)
                    
                    # Download button
                    csv_data = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Predictions CSV",
                        data=csv_data,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Statistics
                    st.markdown("#### 📈 Prediction Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Mean Power", f"{result_df['Predicted_Power_W'].mean():.2f} W")
                    col2.metric("Max Power", f"{result_df['Predicted_Power_W'].max():.2f} W")
                    col3.metric("Min Power", f"{result_df['Predicted_Power_W'].min():.2f} W")
                    col4.metric("Std Dev", f"{result_df['Predicted_Power_W'].std():.2f} W")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ **Connection Error**: API server not reachable.")
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ **API Error**: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                st.error(f"❌ **Error**: {str(e)}")
    else:
        st.info("📤 Upload a CSV file to get started.")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: rgba(255,255,255,0.5); padding: 20px;'>
    <p>🌞 Solar PV Digital Twin Dashboard | Powered by FastAPI + Streamlit + scikit-learn</p>
    <p>Built for College Final Year Project | 2024</p>
</div>
""", unsafe_allow_html=True)