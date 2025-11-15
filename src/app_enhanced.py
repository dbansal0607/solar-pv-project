"""
Solar PV Digital Twin - Enhanced Streamlit Dashboard
With EDA, Advanced Metrics, and Comprehensive Visualizations

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
import os

# ---------- Robust image display helper ----------
def show_image(path, *, fallback_width=700, caption=None):
    """
    Display image with best-supported kwarg across Streamlit versions.
    Tries, in order:
      1. use_container_width=True (newer Streamlit)
      2. use_column_width=True (older Streamlit)
      3. explicit width=fallback_width
    If caption is provided, it will be passed to st.image when supported;
    otherwise leave caption handling to st.caption() calls after showing the image.
    """
    if not os.path.exists(path):
        return False

    # Try the newer API first; if it raises TypeError, fall back gracefully.
    try:
        if caption is None:
            st.image(path, use_container_width=True)
        else:
            st.image(path, use_container_width=True, caption=caption)
        return True
    except TypeError:
        pass
    except Exception:
        # If some other unexpected error occurs (corrupted image etc.), fallback to width
        try:
            if caption is None:
                st.image(path, width=fallback_width)
            else:
                st.image(path, width=fallback_width, caption=caption)
            return True
        except Exception:
            return False

    try:
        if caption is None:
            st.image(path, use_column_width=True)
        else:
            st.image(path, use_column_width=True, caption=caption)
        return True
    except TypeError:
        pass
    except Exception:
        # fallback to explicit width
        try:
            if caption is None:
                st.image(path, width=fallback_width)
            else:
                st.image(path, width=fallback_width, caption=caption)
            return True
        except Exception:
            return False

    # Final fallback: explicit width (should always work)
    try:
        if caption is None:
            st.image(path, width=fallback_width)
        else:
            st.image(path, width=fallback_width, caption=caption)
        return True
    except Exception:
        return False
# ------------------------------------------------

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Solar PV Digital Twin - Enhanced",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# GLASS-THEMED CSS STYLING
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e8ba3 100%);
    }
    
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
    
    .solar-accent {
        color: #FFD700;
        font-weight: 600;
    }
    
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
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================
st.markdown("<h1 style='text-align: center;'>🌞 Solar PV Digital Twin - Enhanced Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.7); font-size: 1.1rem;'>Complete ML Pipeline with EDA, Advanced Metrics & Visualization</p>", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND METRICS
# ============================================================================
@st.cache_resource
def load_model_artifact():
    try:
        artifact = joblib.load('models/pipeline_prod.joblib')
        return artifact
    except:
        return None

@st.cache_resource
def load_metrics():
    try:
        with open('models/metrics.json', 'r') as f:
            return json.load(f)
    except:
        return None

model_artifact = load_model_artifact()
metrics = load_metrics()

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### 📊 Model Information")
    
    if metrics:
        st.markdown(f"**Version:** `{metrics['model_version']}`")
        st.markdown(f"**Test MAE:** {metrics['test_metrics']['mae']} W")
        st.markdown(f"**Test MAPE:** {metrics['test_metrics']['mape']} %")
        st.markdown(f"**Test R²:** {metrics['test_metrics']['r2']}")
        st.markdown(f"**Max Error:** {metrics['test_metrics']['max_error']} W")
        st.markdown(f"**Training Date:** {metrics['training_date'][:10]}")
    else:
        st.warning("⚠️ Run enhanced training first!")
    
    st.markdown("---")
    st.markdown("### ⚙️ API Configuration")
    api_url = st.text_input("API Base URL", value="http://localhost:8000")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Enhanced Features:**
    - 📊 Comprehensive EDA
    - 📈 Advanced Metrics (MAPE, Max Error)
    - 🔍 Residual Analysis
    - 📉 Learning Curves
    - 🎨 10+ Visualizations
    """)

# ============================================================================
# TOP METRICS ROW
# ============================================================================
if metrics:
    col1, col2, col3, col4, col5 = st.columns(5)
    
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
            <div class="metric-label">Test MAPE</div>
            <div class="metric-value">{metrics['test_metrics']['mape']}</div>
            <div class="metric-label">Percent</div>
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
            <div class="metric-label">Max Error</div>
            <div class="metric-value">{metrics['test_metrics']['max_error']}</div>
            <div class="metric-label">Watts</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Explained Var</div>
            <div class="metric-value">{metrics['test_metrics']['explained_variance']}</div>
            <div class="metric-label">Score</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# MAIN TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 EDA & Visualizations", 
    "🎯 Manual Predictor", 
    "📈 Advanced Metrics",
    "🔴 Live Simulation", 
    "📁 Batch Predictions",
    "🔍 Model Analysis"
])

# ============================================================================
# TAB 1: EDA & VISUALIZATIONS
# ============================================================================
with tab1:
    st.markdown("### 📊 Exploratory Data Analysis & Visualizations")
    st.markdown("Comprehensive analysis of the dataset and model predictions")
    
    viz_dir = 'models/visualizations'
    
    if os.path.exists(viz_dir):
        # Section 1: Data Distribution
        st.markdown("#### 1️⃣ Data Distribution Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists(f'{viz_dir}/01_target_distribution.png'):
                show_image(f'{viz_dir}/01_target_distribution.png')
                st.caption("Power Output Distribution: Shows the distribution and outliers in target variable")
        
        with col2:
            if os.path.exists(f'{viz_dir}/02_feature_distributions.png'):
                show_image(f'{viz_dir}/02_feature_distributions.png')
                st.caption("Feature Distributions: Histograms of all input features")
        
        # Section 2: Correlation Analysis
        st.markdown("#### 2️⃣ Feature Correlation Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists(f'{viz_dir}/03_correlation_heatmap.png'):
                show_image(f'{viz_dir}/03_correlation_heatmap.png')
                st.caption("Correlation Heatmap: Shows relationships between all features")
                
                if metrics and 'correlation_with_target' in metrics:
                    st.markdown("**Top Correlations with Power Output:**")
                    corr_sorted = sorted(metrics['correlation_with_target'].items(), 
                                       key=lambda x: abs(x[1]), reverse=True)
                    for i, (feat, corr) in enumerate(corr_sorted[:5], 1):
                        st.markdown(f"{i}. **{feat}**: {corr:.4f}")
        
        with col2:
            if os.path.exists(f'{viz_dir}/05_feature_vs_target.png'):
                show_image(f'{viz_dir}/05_feature_vs_target.png')
                st.caption("Feature vs Target: Scatter plots showing relationships")
        
        # Section 3: Pairplot
        st.markdown("#### 3️⃣ Feature Relationships (Top 3 Features)")
        if os.path.exists(f'{viz_dir}/04_pairplot_top_features.png'):
            show_image(f'{viz_dir}/04_pairplot_top_features.png')
            st.caption("Pairplot: Pairwise relationships between most important features and target")
        
        # Section 4: Model Performance
        st.markdown("#### 4️⃣ Model Performance Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists(f'{viz_dir}/07_predicted_vs_actual.png'):
                show_image(f'{viz_dir}/07_predicted_vs_actual.png')
                st.caption("Predicted vs Actual: How well predictions match reality (closer to red line = better)")
        
        with col2:
            if os.path.exists(f'{viz_dir}/06_residual_analysis.png'):
                show_image(f'{viz_dir}/06_residual_analysis.png')
                st.caption("Residual Analysis: Distribution of prediction errors (centered at 0 = unbiased)")
        
        # Section 5: Error Analysis
        st.markdown("#### 5️⃣ Error Distribution Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists(f'{viz_dir}/08_error_distribution.png'):
                show_image(f'{viz_dir}/08_error_distribution.png')
                st.caption("Error Distribution: Absolute and percentage errors across predictions")
        
        with col2:
            if os.path.exists(f'{viz_dir}/09_feature_importance.png'):
                show_image(f'{viz_dir}/09_feature_importance.png')
                st.caption("Feature Importance: Which features matter most for predictions")
        
        # Section 6: Learning Curve
        st.markdown("#### 6️⃣ Learning Curve Analysis")
        if os.path.exists(f'{viz_dir}/10_learning_curve.png'):
            show_image(f'{viz_dir}/10_learning_curve.png')
            st.caption("Learning Curve: Model performance vs training set size (converging lines = well-trained)")
        
        # Interpretation Guide
        st.markdown("---")
        st.markdown("### 💡 How to Interpret These Visualizations")
        
        with st.expander("📖 Click to read interpretation guide"):
            st.markdown("""
            **1. Target Distribution**
            - Shows if power output data is normally distributed or skewed
            - Box plot reveals outliers (points outside whiskers)
            
            **2. Correlation Heatmap**
            - Red colors = strong positive correlation
            - Values close to 1.0 or -1.0 indicate strong relationships
            - Look for features highly correlated with Power_Output_W
            
            **3. Predicted vs Actual**
            - Points close to red diagonal line = accurate predictions
            - Spread around line = prediction variability
            - R² score shows % of variance explained
            
            **4. Residual Plot**
            - Should be randomly scattered around 0
            - Patterns indicate model bias or non-linear relationships
            - Normal distribution of residuals = good model
            
            **5. Error Distribution**
            - Most errors should be small (left side of histogram)
            - Few large errors = robust model
            - MAPE shows average percentage error
            
            **6. Feature Importance**
            - Taller bars = more important features
            - Top features drive predictions most
            - Validates domain knowledge (e.g., irradiance matters most)
            
            **7. Learning Curve**
            - Gap between training/validation = overfitting
            - Converging lines = model generalizes well
            - Plateau = model reached max capacity
            """)
    
    else:
        st.warning("⚠️ Visualizations not found. Please run the enhanced training script first:")
        st.code("python src/train_production_enhanced.py", language="bash")

# ============================================================================
# TAB 2: MANUAL PREDICTOR
# ============================================================================
with tab2:
    st.markdown("### 🎯 Single Prediction (API-Powered)")
    st.markdown("Enter environmental parameters and get instant power output predictions.")
    
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
            st.error("❌ **Connection Error**: Could not connect to API.")
        except Exception as e:
            st.error(f"❌ **Error**: {str(e)}")

# ============================================================================
# TAB 3: ADVANCED METRICS
# ============================================================================
with tab3:
    st.markdown("### 📈 Advanced Performance Metrics")
    
    if metrics:
        # Metrics comparison table
        st.markdown("#### 📊 Model Performance Across All Sets")
        
        metrics_df = pd.DataFrame({
            'Metric': ['MAE (W)', 'RMSE (W)', 'R² Score', 'MAPE (%)', 'Explained Variance', 'Max Error (W)'],
            'Training': [
                metrics['training_metrics']['mae'],
                metrics['training_metrics']['rmse'],
                metrics['training_metrics']['r2'],
                metrics['training_metrics']['mape'],
                metrics['training_metrics']['explained_variance'],
                metrics['training_metrics']['max_error']
            ],
            'Validation': [
                metrics['validation_metrics']['mae'],
                metrics['validation_metrics']['rmse'],
                metrics['validation_metrics']['r2'],
                metrics['validation_metrics']['mape'],
                metrics['validation_metrics']['explained_variance'],
                metrics['validation_metrics']['max_error']
            ],
            'Test': [
                metrics['test_metrics']['mae'],
                metrics['test_metrics']['rmse'],
                metrics['test_metrics']['r2'],
                metrics['test_metrics']['mape'],
                metrics['test_metrics']['explained_variance'],
                metrics['test_metrics']['max_error']
            ]
        })
        
        st.dataframe(metrics_df, use_container_width=True, height=250)
        
        # Metric explanations
        st.markdown("#### 💡 Metric Interpretations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **MAE (Mean Absolute Error)**
            - Average prediction error in Watts
            - Lower is better
            - Your MAE: {:.2f}W means predictions off by ~{:.2f}W on average
            
            **RMSE (Root Mean Squared Error)**
            - Similar to MAE but penalizes large errors more
            - Always ≥ MAE
            - Large difference suggests outlier errors exist
            
            **R² Score (R-squared)**
            - Proportion of variance explained (0 to 1)
            - Your R²: {:.4f} means {:.1f}% variance explained
            - Above 0.9 is excellent for regression
            """.format(
                metrics['test_metrics']['mae'],
                metrics['test_metrics']['mae'],
                metrics['test_metrics']['r2'],
                metrics['test_metrics']['r2'] * 100
            ))
        
        with col2:
            st.markdown("""
            **MAPE (Mean Absolute Percentage Error)**
            - Average error as a percentage
            - Your MAPE: {:.2f}% means typical error is {:.2f}% of actual value
            - Below 10% is generally considered accurate
            
            **Explained Variance Score**
            - How much variance model captures
            - Close to R² but calculated differently
            - Higher values indicate better fit
            
            **Max Error**
            - Worst single prediction error
            - Your Max: {:.2f}W
            - Important for understanding model limitations
            """.format(
                metrics['test_metrics']['mape'],
                metrics['test_metrics']['mape'],
                metrics['test_metrics']['max_error']
            ))
        
        # Training vs Test comparison
        st.markdown("#### 🔍 Overfitting Analysis")
        
        train_r2 = metrics['training_metrics']['r2']
        test_r2 = metrics['test_metrics']['r2']
        r2_gap = train_r2 - test_r2
        
        if r2_gap < 0.05:
            st.success(f"✅ **Excellent Generalization**: Training R² ({train_r2:.4f}) and Test R² ({test_r2:.4f}) are very close (gap: {r2_gap:.4f}). Model generalizes well to unseen data!")
        elif r2_gap < 0.10:
            st.info(f"ℹ️ **Good Generalization**: Small gap between Training R² ({train_r2:.4f}) and Test R² ({test_r2:.4f}). Gap: {r2_gap:.4f}. Acceptable performance.")
        else:
            st.warning(f"⚠️ **Potential Overfitting**: Significant gap between Training R² ({train_r2:.4f}) and Test R² ({test_r2:.4f}). Gap: {r2_gap:.4f}. Consider regularization or more data.")
        
        # Feature importance ranking
        st.markdown("#### 🏆 Feature Importance Ranking")
        
        if 'feature_importance' in metrics:
            importance_df = pd.DataFrame({
                'Rank': range(1, len(metrics['feature_importance']) + 1),
                'Feature': list(metrics['feature_importance'].keys()),
                'Importance': list(metrics['feature_importance'].values())
            }).sort_values('Importance', ascending=False).reset_index(drop=True)
            importance_df['Rank'] = range(1, len(importance_df) + 1)
            
            st.dataframe(importance_df, use_container_width=True, height=300)
            
            st.markdown(f"""
            **Key Insight**: The top feature **{importance_df.iloc[0]['Feature']}** 
            contributes {importance_df.iloc[0]['Importance']:.2%} to predictions, 
            {importance_df.iloc[0]['Importance'] / importance_df.iloc[-1]['Importance']:.1f}x more than 
            the least important feature.
            """)
    
    else:
        st.warning("⚠️ Metrics not loaded. Run enhanced training first.")

# ============================================================================
# TAB 4: LIVE SIMULATION
# ============================================================================
with tab4:
    st.markdown("### 🔴 Live Digital Twin Simulation")
    st.markdown("Watch real-time simulated sensor data and power predictions.")
    
    if model_artifact:
        if 'simulation_running' not in st.session_state:
            st.session_state.simulation_running = False
        if 'simulation_data' not in st.session_state:
            st.session_state.simulation_data = []
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("▶️ Start" if not st.session_state.simulation_running else "⏸️ Stop"):
                st.session_state.simulation_running = not st.session_state.simulation_running
            
            if st.button("🔄 Reset"):
                st.session_state.simulation_data = []
        
        if st.session_state.simulation_running:
            chart_placeholder = st.empty()
            metrics_placeholder = st.empty()
            
            for i in range(50):
                if not st.session_state.simulation_running:
                    break
                
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
                
                input_df = pd.DataFrame([simulated_data])
                prediction = model_artifact['pipeline'].predict(input_df)[0]
                
                st.session_state.simulation_data.append({
                    'timestamp': datetime.now(),
                    'power': prediction,
                    'irradiance': simulated_data['Solar_Irradiance_kWh_m2']
                })
                
                if len(st.session_state.simulation_data) > 30:
                    st.session_state.simulation_data.pop(0)
                
                if len(st.session_state.simulation_data) > 0:
                    sim_df = pd.DataFrame(st.session_state.simulation_data)
                    
                    power_chart = alt.Chart(sim_df.reset_index()).mark_line(
                        color='#FFD700',
                        strokeWidth=3,
                        point=alt.OverlayMarkDef(color='#FFA500', size=50)
                    ).encode(
                        x=alt.X('index:Q', title='Time Step'),
                        y=alt.Y('power:Q', title='Predicted Power (W)', scale=alt.Scale(zero=False)),
                        tooltip=['power']
                    ).properties(height=300, title='Real-Time Power Output')
                    
                    chart_placeholder.altair_chart(power_chart, use_container_width=True)
                    
                    current_power = st.session_state.simulation_data[-1]['power']
                    avg_power = np.mean([d['power'] for d in st.session_state.simulation_data])
                    
                    col1, col2, col3 = metrics_placeholder.columns(3)
                    col1.metric("Current Power", f"{current_power:.2f} W", f"{current_power - avg_power:.2f} W")
                    col2.metric("Average Power", f"{avg_power:.2f} W")
                    col3.metric("Data Points", len(st.session_state.simulation_data))
                
                time.sleep(0.5)
        else:
            st.info("▶️ Click 'Start' to begin simulation.")
    else:
        st.error("❌ Model not loaded.")

# ============================================================================
# TAB 5: BATCH PREDICTIONS
# ============================================================================
with tab5:
    st.markdown("### 📁 Batch CSV Predictions")
    st.markdown("Upload a CSV file for bulk predictions via API.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        st.markdown("#### 📋 Uploaded Data Preview")
        st.dataframe(df_upload.head())
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", len(df_upload))
        with col2:
            st.metric("Total Columns", len(df_upload.columns))
        
        if st.button("🚀 Run Batch Prediction", use_container_width=True):
            try:
                with st.spinner("🔄 Sending to API..."):
                    uploaded_file.seek(0)
                    files = {'file': ('uploaded.csv', uploaded_file, 'text/csv')}
                    params = {'return_csv': 'true'}
                    
                    response = requests.post(
                        f"{api_url}/predict_batch",
                        files=files,
                        params=params,
                        timeout=30
                    )
                    response.raise_for_status()
                    
                    result_df = pd.read_csv(io.StringIO(response.text))
                    
                    st.success(f"✅ Successfully predicted {len(result_df)} samples!")
                    
                    st.markdown("#### 📊 Prediction Results")
                    st.dataframe(result_df)
                    
                    csv_data = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Predictions CSV",
                        data=csv_data,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    st.markdown("#### 📈 Prediction Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Mean Power", f"{result_df['Predicted_Power_W'].mean():.2f} W")
                    col2.metric("Max Power", f"{result_df['Predicted_Power_W'].max():.2f} W")
                    col3.metric("Min Power", f"{result_df['Predicted_Power_W'].min():.2f} W")
                    col4.metric("Std Dev", f"{result_df['Predicted_Power_W'].std():.2f} W")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ **Connection Error**: API server not reachable.")
            except Exception as e:
                st.error(f"❌ **Error**: {str(e)}")
    else:
        st.info("📤 Upload a CSV file to get started.")

# ============================================================================
# TAB 6: MODEL ANALYSIS
# ============================================================================
with tab6:
    st.markdown("### 🔍 Deep Model Analysis")
    
    if metrics:
        # Hyperparameters
        st.markdown("#### ⚙️ Best Hyperparameters (GridSearchCV)")
        
        if 'best_hyperparameters' in metrics:
            params_df = pd.DataFrame({
                'Parameter': [k.replace('model__', '') for k in metrics['best_hyperparameters'].keys()],
                'Value': list(metrics['best_hyperparameters'].values())
            })
            st.dataframe(params_df, use_container_width=True)
            
            st.markdown("""
            **What these mean:**
            - **n_estimators**: Number of trees in the Random Forest
            - **max_depth**: Maximum depth each tree can grow
            - **min_samples_split**: Minimum samples required to split a node
            - **min_samples_leaf**: Minimum samples required in a leaf node
            """)
        
        # Data split info
        st.markdown("#### 📦 Dataset Split Information")
        
        if 'data_split' in metrics:
            split_df = pd.DataFrame({
                'Set': ['Training', 'Validation', 'Test'],
                'Samples': [
                    metrics['data_split']['train_samples'],
                    metrics['data_split']['validation_samples'],
                    metrics['data_split']['test_samples']
                ],
                'Percentage': [
                    metrics['data_split']['train_percentage'],
                    metrics['data_split']['validation_percentage'],
                    metrics['data_split']['test_percentage']
                ]
            })
            
            st.dataframe(split_df, use_container_width=True)
            
            # Pie chart of split
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#FFD700', '#FFA500', '#FF8C00']
            ax.pie(split_df['Samples'], labels=split_df['Set'], autopct='%1.1f%%',
                   colors=colors, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
            ax.set_title('Data Split Distribution', fontsize=14, fontweight='bold')
            st.pyplot(fig)
            plt.close()
        
        # Correlation analysis
        st.markdown("#### 🔗 Feature Correlation with Target")
        
        if 'correlation_with_target' in metrics:
            corr_df = pd.DataFrame({
                'Feature': list(metrics['correlation_with_target'].keys()),
                'Correlation': list(metrics['correlation_with_target'].values())
            }).sort_values('Correlation', ascending=False, key=abs)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#FF6B6B' if x < 0 else '#4ECDC4' for x in corr_df['Correlation']]
            ax.barh(corr_df['Feature'], corr_df['Correlation'], color=colors, edgecolor='black')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax.set_xlabel('Correlation Coefficient', fontsize=12)
            ax.set_title('Feature Correlations with Power Output', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            st.pyplot(fig)
            plt.close()
            
            st.markdown("""
            **Interpretation:**
            - Positive correlations (green): Feature increases → Power increases
            - Negative correlations (red): Feature increases → Power decreases
            - Correlation closer to ±1.0 = stronger relationship
            """)
        
        # Outlier information
        st.markdown("#### 🎯 Outlier Detection Summary")
        
        if 'outlier_counts' in metrics:
            outlier_df = pd.DataFrame({
                'Feature': list(metrics['outlier_counts'].keys()),
                'Outliers': list(metrics['outlier_counts'].values())
            }).sort_values('Outliers', ascending=False)
            
            total_samples = (metrics['data_split']['train_samples'] + 
                           metrics['data_split']['validation_samples'] + 
                           metrics['data_split']['test_samples'])
            
            outlier_df['Percentage'] = (outlier_df['Outliers'] / total_samples * 100).round(2)
            
            st.dataframe(outlier_df, use_container_width=True)
            
            total_outliers = outlier_df['Outliers'].sum()
            if total_outliers == 0:
                st.success("✅ No significant outliers detected in any feature!")
            else:
                st.info(f"ℹ️ Total outliers detected: {total_outliers} across all features")
        
        # Model metadata
        st.markdown("#### 📋 Model Metadata")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Model Version:** `{metrics['model_version']}`
            
            **Training Date:** {metrics['training_date'][:19].replace('T', ' ')}
            
            **Total Training Time:** GridSearchCV tested 81 combinations
            
            **Algorithm:** RandomForest Regressor
            
            **Preprocessing:** StandardScaler (normalize features)
            """)
        
        with col2:
            st.markdown(f"""
            **scikit-learn Version:** 1.3.0
            
            **Random State:** 42 (reproducible results)
            
            **Cross-Validation:** 3-fold CV during hyperparameter tuning
            
            **Scoring Metric:** Negative MAE (minimize error)
            
            **Pipeline Steps:** Scaler → Model
            """)
        
        # Download metrics
        st.markdown("#### 💾 Export Metrics")
        
        metrics_json = json.dumps(metrics, indent=2)
        st.download_button(
            label="⬇️ Download Full Metrics JSON",
            data=metrics_json,
            file_name=f"model_metrics_{metrics['model_version']}.json",
            mime="application/json",
            use_container_width=True
        )
    
    else:
        st.warning("⚠️ Metrics not loaded. Run enhanced training first.")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: rgba(255,255,255,0.5); padding: 20px;'>
    <p>🌞 Solar PV Digital Twin - Enhanced Dashboard</p>
    <p>With Comprehensive EDA, Advanced Metrics & 10+ Visualizations</p>
    <p>Built for College Final Year Project | 2024</p>
</div>
""", unsafe_allow_html=True)
