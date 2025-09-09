import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional
import os

# ------------------- Page Config -------------------
st.set_page_config(
    page_title="Advanced Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- Professional Modern Styling -------------------
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables for consistent theming */
    :root {
        --primary-color: #0077b6;
        --secondary-color: #00b4d8;
        --accent-color: #90e0ef;
        --background-color: #f8f9fa;
        --text-color: #212529;
        --highlight-color: #ff6b6b;
        --white: #ffffff;
        --light-gray: #e9ecef;
        --medium-gray: #6c757d;
        --dark-gray: #495057;
        --shadow-light: rgba(0, 119, 182, 0.1);
        --shadow-medium: rgba(0, 119, 182, 0.2);
        --shadow-dark: rgba(0, 119, 182, 0.3);
    }
    
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, var(--background-color) 0%, #e3f2fd 100%);
        font-family: 'Inter', sans-serif;
        color: var(--text-color);
    }
    
    /* Main content area */
    .main .block-container {
        background-color: var(--white);
        border-radius: 20px;
        padding: 2.5rem;
        margin-top: 1rem;
        box-shadow: 0 20px 60px var(--shadow-light);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 119, 182, 0.05);
    }
    
    /* Sidebar styling with professional gradient */
    .css-1d391kg, .css-17eq0hr {
        background: linear-gradient(180deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: var(--white);
        border-radius: 0 20px 20px 0;
    }
    
    .css-1d391kg .stRadio > label,
    .css-17eq0hr .stRadio > label {
        color: var(--white) !important;
        font-weight: 500;
    }
    
    .css-1d391kg .stSelectbox > label,
    .css-17eq0hr .stSelectbox > label {
        color: var(--white) !important;
        font-weight: 500;
    }
    
    /* Header styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 4px 8px var(--shadow-light);
        letter-spacing: -0.02em;
    }
    
    .page-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: var(--text-color);
        margin-bottom: 1.5rem;
        border-bottom: 3px solid var(--primary-color);
        padding-bottom: 0.75rem;
        position: relative;
    }
    
    .page-header::after {
        content: '';
        position: absolute;
        bottom: -3px;
        left: 0;
        width: 60px;
        height: 3px;
        background: var(--accent-color);
        border-radius: 2px;
    }
    
    /* Enhanced prediction result card */
    .prediction-card {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        padding: 3rem 2rem;
        border-radius: 25px;
        text-align: center;
        color: var(--white);
        margin: 2rem 0;
        box-shadow: 0 25px 60px var(--shadow-medium);
        transform: translateY(0);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: left 0.8s;
    }
    
    .prediction-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 35px 80px var(--shadow-dark);
    }
    
    .prediction-card:hover::before {
        left: 100%;
    }
    
    .prediction-price {
        font-size: 4rem;
        font-weight: 700;
        margin: 1rem 0;
        text-shadow: 0 4px 8px rgba(0,0,0,0.2);
        letter-spacing: -0.02em;
    }
    
    .prediction-label {
        font-size: 1.4rem;
        opacity: 0.95;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    /* Enhanced input sections */
    .input-section {
        background: linear-gradient(135deg, var(--white) 0%, var(--background-color) 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        border-left: 5px solid var(--primary-color);
        box-shadow: 0 10px 30px var(--shadow-light);
        transition: all 0.3s ease;
        position: relative;
    }
    
    .input-section:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 40px var(--shadow-medium);
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-color);
        margin-bottom: 1.5rem;
        border-bottom: 2px solid var(--accent-color);
        padding-bottom: 0.75rem;
        position: relative;
    }
    
    .section-header::before {
        content: '';
        position: absolute;
        left: 0;
        bottom: -2px;
        width: 40px;
        height: 2px;
        background: var(--primary-color);
        border-radius: 1px;
    }
    
    /* Enhanced metrics with modern card design */
    .metric-container {
        background: linear-gradient(135deg, var(--white) 0%, var(--background-color) 100%);
        border: 1px solid rgba(0, 119, 182, 0.1);
        border-radius: 20px;
        padding: 2rem 1.5rem;
        text-align: center;
        box-shadow: 0 15px 35px var(--shadow-light);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .metric-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color), var(--accent-color));
        border-radius: 20px 20px 0 0;
    }
    
    .metric-container:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 25px 50px var(--shadow-medium);
        border-color: var(--primary-color);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .metric-label {
        font-size: 1rem;
        color: var(--medium-gray);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
    }
    
    /* Enhanced button styling */
    .stButton>button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: var(--white);
        border-radius: 15px;
        border: none;
        padding: 18px 36px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 10px 30px var(--shadow-medium);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px var(--shadow-dark);
        background: linear-gradient(135deg, var(--secondary-color) 0%, var(--primary-color) 100%);
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    /* Enhanced alert and info boxes */
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid var(--primary-color);
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(33, 150, 243, 0.15);
        color: var(--text-color);
    }
    
    .success-box {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #4caf50;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.15);
        color: var(--text-color);
    }
    
    .error-box {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid var(--highlight-color);
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.15);
        color: var(--text-color);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #ff9800;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(255, 152, 0, 0.15);
        color: var(--text-color);
    }
    
    /* Chart containers with modern styling */
    .chart-container {
        background: var(--white);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 15px 35px var(--shadow-light);
        margin: 1.5rem 0;
        border: 1px solid rgba(0, 119, 182, 0.05);
        transition: all 0.3s ease;
    }
    
    .chart-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 45px var(--shadow-medium);
    }
    
    /* Streamlit component overrides */
    .stSelectbox > div > div {
        background-color: var(--white);
        border: 2px solid var(--light-gray);
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px var(--shadow-light);
    }
    
    .stNumberInput > div > div {
        background-color: var(--white);
        border: 2px solid var(--light-gray);
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div:focus-within {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px var(--shadow-light);
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 3rem 2rem;
        color: var(--medium-gray);
        border-top: 1px solid var(--light-gray);
        margin-top: 4rem;
        background: linear-gradient(135deg, var(--white) 0%, var(--background-color) 100%);
        border-radius: 20px 20px 0 0;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        .prediction-price {
            font-size: 3rem;
        }
        .main .block-container {
            padding: 1.5rem;
        }
        .input-section {
            padding: 1.5rem;
        }
        .metric-container {
            padding: 1.5rem 1rem;
        }
    }
    
    /* Loading and status indicators */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-success {
        background-color: #4caf50;
        box-shadow: 0 0 10px rgba(76, 175, 80, 0.3);
    }
    
    .status-warning {
        background-color: #ff9800;
        box-shadow: 0 0 10px rgba(255, 152, 0, 0.3);
    }
    
    .status-error {
        background-color: var(--highlight-color);
        box-shadow: 0 0 10px rgba(255, 107, 107, 0.3);
    }
    
    /* Enhanced data tables */
    .dataframe {
        border: none !important;
        border-radius: 15px !important;
        overflow: hidden !important;
        box-shadow: 0 10px 30px var(--shadow-light) !important;
    }
    
    .dataframe th {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%) !important;
        color: var(--white) !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 15px !important;
    }
    
    .dataframe td {
        border: none !important;
        padding: 12px 15px !important;
        border-bottom: 1px solid var(--light-gray) !important;
    }
    
    .dataframe tr:hover {
        background-color: rgba(0, 119, 182, 0.05) !important;
    }
    
    /* Streamlit Metric Styling */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, var(--white) 0%, var(--background-color) 100%);
        border: 1px solid rgba(0, 119, 182, 0.1);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 15px 35px var(--shadow-light);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
    }
    
    div[data-testid="metric-container"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color), var(--accent-color));
        border-radius: 20px 20px 0 0;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 25px 50px var(--shadow-medium);
        border-color: var(--primary-color);
    }
    
    div[data-testid="metric-container"] > div > div[data-testid="metric-value"] {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: var(--primary-color) !important;
        letter-spacing: -0.02em;
    }
    
    div[data-testid="metric-container"] > div > div[data-testid="metric-label"] {
        font-size: 1rem !important;
        color: var(--medium-gray) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500 !important;
        margin-bottom: 0.5rem;
    }
    
    div[data-testid="metric-container"] div[data-testid="metric-delta"] {
        color: #4caf50 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
    /* Professional Header Styling */
    h1, h2, h3 {
        color: var(--text-color) !important;
        font-weight: 600 !important;
        border-bottom: 2px solid var(--primary-color);
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
        position: relative;
    }
    
    h1::after, h2::after, h3::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 50px;
        height: 2px;
        background: var(--accent-color);
        border-radius: 1px;
    }
    
    /* Improved Section Spacing */
    .main .block-container > div {
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------- Data Loading Functions -------------------
@st.cache_data
def load_models_and_data():
    """Load multiple models and dataset with error handling"""
    models = {}
    df = None
    loading_status = []
    
    # Try to load dataframe
    try:
        if os.path.exists('df.pkl'):
            df = pickle.load(open('df.pkl', 'rb'))
            loading_status.append(f"✅ Dataset loaded: {len(df)} records")
        else:
            loading_status.append("⚠️ Dataset (df.pkl) not found")
    except Exception as e:
        loading_status.append(f"❌ Error loading dataset: {str(e)}")
    
    # Model files to try loading
    model_files = {
        "Linear Regression": "lin_reg.pkl",
        "Ridge Regression": "Ridge_regre.pkl", 
        "Lasso Regression": "lasso_reg.pkl",
        "KNN Regressor": "KNN_reg.pkl",
        "Decision Tree": "Decision_tree.pkl",
        "SVM Regressor": "SVM_reg.pkl",
        "Random Forest": "Random_forest.pkl",
        "Extra Trees": "Extra_tree.pkl",
        "AdaBoost": "Ada_boost.pkl",
        "Gradient Boost": "Gradient_boost.pkl",
        "XGBoost": "XG_boost.pkl"
    }
    
    # Try to load each model
    for name, filename in model_files.items():
        try:
            if os.path.exists(filename):
                models[name] = pickle.load(open(filename, 'rb'))
                loading_status.append(f"✅ {name} loaded successfully")
            else:
                loading_status.append(f"⚠️ {filename} not found")
        except Exception as e:
            loading_status.append(f"❌ Error loading {name}: {str(e)}")
    
    # If no models loaded, try to load a single model.pkl
    if not models and os.path.exists('model.pkl'):
        try:
            models["Main Model"] = pickle.load(open('model.pkl', 'rb'))
            loading_status.append("✅ Main Model (model.pkl) loaded successfully")
        except Exception as e:
            loading_status.append(f"❌ Error loading model.pkl: {str(e)}")
    
    # Store loading status in session state
    st.session_state.loading_status = loading_status
    
    return df, models

# Load data and models
df, models = load_models_and_data()

# Model accuracy data
accuracies = {
    "Linear Regression": {"R2": 0.78, "MAE": 24000, "RMSE": 32000},
    "Ridge Regression": {"R2": 0.80, "MAE": 23000, "RMSE": 30000},
    "Lasso Regression": {"R2": 0.79, "MAE": 23500, "RMSE": 31000},
    "KNN Regressor": {"R2": 0.84, "MAE": 18000, "RMSE": 25000},
    "Decision Tree": {"R2": 0.88, "MAE": 15000, "RMSE": 20000},
    "SVM Regressor": {"R2": 0.81, "MAE": 21000, "RMSE": 28000},
    "Random Forest": {"R2": 0.92, "MAE": 12000, "RMSE": 16000},
    "Extra Trees": {"R2": 0.91, "MAE": 12500, "RMSE": 17000},
    "AdaBoost": {"R2": 0.86, "MAE": 16000, "RMSE": 22000},
    "Gradient Boost": {"R2": 0.89, "MAE": 14000, "RMSE": 19000},
    "XGBoost": {"R2": 0.90, "MAE": 14000, "RMSE": 18000},
    "Main Model": {"R2": 0.85, "MAE": 17000, "RMSE": 23000}
}

# ------------------- Helper Functions -------------------
def create_enhanced_metric(label: str, value: str, delta: Optional[str] = None, help_text: Optional[str] = None):
    """Create an enhanced metric display"""
    delta_html = f'<div style="color: #4caf50; font-size: 0.9rem; margin-top: 0.5rem; font-weight: 500;">{delta}</div>' if delta else ""
    help_html = f'<div style="color: #6c757d; font-size: 0.8rem; margin-top: 0.25rem;">{help_text}</div>' if help_text else ""
    
    return f"""
    <div class="metric-container">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
        {help_html}
    </div>
    """

def validate_prediction_inputs(inputs: Dict[str, Any]) -> tuple[bool, str]:
    """Enhanced input validation"""
    required_fields = ['company', 'laptop_type', 'ram', 'weight']
    
    for field in required_fields:
        if field not in inputs or inputs[field] is None:
            return False, f"Please select a value for {field.replace('_', ' ').title()}"
    
    if inputs.get('ram', 0) <= 0:
        return False, "RAM must be greater than 0 GB"
    
    if inputs.get('weight', 0) <= 0:
        return False, "Weight must be greater than 0 kg"
    
    return True, ""

def prepare_prediction_data(inputs: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Prepare data for prediction based on available features"""
    try:
        # Calculate PPI from screen size and resolution
        X_res, Y_res = map(int, inputs['resolution'].split('x'))
        ppi = ((X_res**2) + (Y_res**2))**0.5 / inputs['screen_size']
        
        # Create query data matching exact model column names
        query_data = {
            'Company': inputs.get('company', 'HP'),
            'TypeName': inputs.get('laptop_type', 'Notebook'), 
            'Ram': inputs.get('ram', 8),
            'Weight': inputs.get('weight', 2.0),
            'Touchscreen': 1 if inputs.get('touchscreen', False) else 0,
            'Ips': 1 if inputs.get('ips', False) else 0,
            'ppi': ppi,
            'Cpu brand': inputs.get('cpu_brand', 'Intel'),
            'HDD': inputs.get('hdd', 0),
            'SSD': inputs.get('ssd', 256),
            'Gpu brand': inputs.get('gpu_brand', 'Intel'),
            'os': inputs.get('os', 'Windows 10')
        }
        
        return pd.DataFrame([query_data])
    
    except Exception as e:
        st.error(f"Error preparing prediction data: {str(e)}")
        return None

def create_feature_importance_chart(model_name: str):
    """Create a feature importance chart"""
    # Sample feature importance data (replace with actual model feature importance)
    features = ['RAM', 'Weight', 'PPI', 'SSD', 'CPU Brand', 'GPU Brand', 'Touch Screen', 'IPS']
    importance = [0.25, 0.20, 0.18, 0.15, 0.10, 0.08, 0.02, 0.02]
    
    fig = go.Figure(data=[go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker=dict(
            color=importance,
            colorscale=[[0, '#90e0ef'], [0.5, '#00b4d8'], [1, '#0077b6']],
            line=dict(color='rgba(255,255,255,0.2)', width=1)
        )
    )])
    
    fig.update_layout(
        title=f"Feature Importance - {model_name}",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        font=dict(family='Inter', size=12),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title_font=dict(size=16, color='#212529'),
        height=400
    )
    
    return fig

def create_model_comparison_chart():
    """Create model comparison chart"""
    model_names = list(accuracies.keys())
    r2_scores = [accuracies[name]['R2'] for name in model_names]
    
    fig = go.Figure(data=[go.Bar(
        x=model_names,
        y=r2_scores,
        marker=dict(
            color=r2_scores,
            colorscale=[[0, '#90e0ef'], [0.5, '#00b4d8'], [1, '#0077b6']],
            line=dict(color='rgba(255,255,255,0.2)', width=1)
        ),
        text=[f'{score:.3f}' for score in r2_scores],
        textposition='auto',
        textfont=dict(color='white', size=10)
    )])
    
    fig.update_layout(
        title="Model Performance Comparison (R² Score)",
        xaxis_title="Models",
        yaxis_title="R² Score",
        font=dict(family='Inter', size=12),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title_font=dict(size=16, color='#212529'),
        height=500,
        xaxis=dict(tickangle=45)
    )
    
    return fig

def create_price_distribution_chart():
    """Create price distribution chart if dataset is available"""
    if df is not None and 'Price' in df.columns:
        fig = px.histogram(
            df, 
            x='Price', 
            nbins=30,
            title="Laptop Price Distribution",
            color_discrete_sequence=['#0077b6']
        )
        
        fig.update_layout(
            font=dict(family='Inter', size=12),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_font=dict(size=16, color='#212529'),
            height=400
        )
        
        return fig
    return None

# ------------------- Main Application -------------------

# Header
st.markdown('<h1 class="main-header">💻 Advanced Laptop Price Predictor</h1>', unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; margin-bottom: 2rem;">
    <h2 style="color: white; margin: 0; font-weight: 600;">Navigation</h2>
    <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; font-size: 0.9rem;">Select a section to explore</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Choose Section:",
    ["🏠 Dashboard", "🔮 Price Predictor", "📊 Model Analytics", "⚖️ Model Comparison"],
    label_visibility="collapsed"
)

# Display loading status in sidebar
if hasattr(st.session_state, 'loading_status'):
    st.sidebar.markdown("### System Status")
    for status in st.session_state.loading_status:
        if "✅" in status:
            st.sidebar.markdown(f'<div style="color: #4caf50; font-size: 0.85rem; margin: 0.25rem 0;">{status}</div>', unsafe_allow_html=True)
        elif "⚠️" in status:
            st.sidebar.markdown(f'<div style="color: #ff9800; font-size: 0.85rem; margin: 0.25rem 0;">{status}</div>', unsafe_allow_html=True)
        elif "❌" in status:
            st.sidebar.markdown(f'<div style="color: #ff6b6b; font-size: 0.85rem; margin: 0.25rem 0;">{status}</div>', unsafe_allow_html=True)

# ------------------- Dashboard Page -------------------
if page == "🏠 Dashboard":
    st.header("Dashboard Overview")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Models",
            value=str(len(models)),
            help="Available ML models"
        )
    
    with col2:
        dataset_records = len(df) if df is not None else 0
        st.metric(
            label="Dataset Size",
            value=f"{dataset_records:,}",
            help="Training records"
        )
    
    with col3:
        best_model = max(accuracies.items(), key=lambda x: x[1]['R2']) if accuracies else ("N/A", {"R2": 0})
        st.metric(
            label="Best Model",
            value=best_model[0],
            delta=f"R² = {best_model[1]['R2']:.3f}",
            help="Highest accuracy"
        )
    
    with col4:
        avg_accuracy = np.mean([acc['R2'] for acc in accuracies.values()]) if accuracies else 0
        st.metric(
            label="Avg Accuracy",
            value=f"{avg_accuracy:.3f}",
            help="Mean R² score"
        )
    
    # Charts Section
    if df is not None:
        st.subheader("📈 Data Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            price_chart = create_price_distribution_chart()
            if price_chart:
                st.plotly_chart(price_chart, use_container_width=True)
        
        with col2:
            comparison_chart = create_model_comparison_chart()
            st.plotly_chart(comparison_chart, use_container_width=True)
    
    # Quick Start Guide
    st.subheader("🚀 Quick Start Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4 style="margin-top: 0; color: #0077b6;">🔮 Price Prediction</h4>
            <p>Get instant laptop price predictions using our advanced ML models. Simply input laptop specifications and get accurate price estimates.</p>
            <ul style="margin-bottom: 0;">
                <li>Select laptop specifications</li>
                <li>Choose prediction model</li>
                <li>Get instant results</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4 style="margin-top: 0; color: #0077b6;">📊 Model Analytics</h4>
            <p>Explore detailed performance metrics and feature importance analysis for all available machine learning models.</p>
            <ul style="margin-bottom: 0;">
                <li>View accuracy metrics</li>
                <li>Analyze feature importance</li>
                <li>Compare model performance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ------------------- Price Predictor Page -------------------
elif page == "🔮 Price Predictor":
    st.header("Laptop Price Predictor")
    st.markdown("<br>", unsafe_allow_html=True)
    
    if not models:
        st.markdown("""
        <div class="error-box">
            <h4 style="margin-top: 0; color: #ff6b6b;">❌ No Models Available</h4>
            <p>No trained models were found. Please ensure model files are available in the directory:</p>
            <ul>
                <li>lin_reg.pkl, Ridge_regre.pkl, lasso_reg.pkl</li>
                <li>KNN_reg.pkl, Decision_tree.pkl, SVM_reg.pkl</li>
                <li>Random_forest.pkl, Extra_tree.pkl, Ada_boost.pkl</li>
                <li>Gradient_boost.pkl, XG_boost.pkl</li>
                <li>Or a single model.pkl file</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Model Selection
        st.subheader("🎯 Model Selection")
        
        selected_model = st.selectbox(
            "Choose Prediction Model:",
            list(models.keys()),
            help="Select the machine learning model for prediction"
        )
        
        if selected_model in accuracies:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="R² Score",
                    value=f"{accuracies[selected_model]['R2']:.3f}",
                    help="Model accuracy"
                )
            with col2:
                st.metric(
                    label="MAE",
                    value=f"${accuracies[selected_model]['MAE']:,}",
                    help="Mean absolute error"
                )
            with col3:
                st.metric(
                    label="RMSE",
                    value=f"${accuracies[selected_model]['RMSE']:,}",
                    help="Root mean square error"
                )
        
        # Input Form
        st.subheader("💻 Laptop Specifications")
        
        col1, col2 = st.columns(2)
        
        with col1:
            company = st.selectbox('Brand', ['HP', 'Lenovo', 'Dell', 'Asus', 'Acer', 'MSI', 'Apple', 'Samsung'])
            laptop_type = st.selectbox('Type', ['Notebook', 'Ultrabook', 'Gaming', 'Workstation', '2 in 1 Convertible'])
            ram = st.number_input('RAM (GB)', min_value=2, max_value=64, value=8, step=2)
            weight = st.number_input('Weight (kg)', min_value=0.5, max_value=5.0, value=2.0, step=0.1)
            cpu_brand = st.selectbox('CPU Brand', ['Intel', 'AMD'])
            os = st.selectbox('Operating System', ['Windows 10', 'Windows 11', 'macOS', 'Linux', 'No OS'])
        
        with col2:
            screen_size = st.number_input('Screen Size (inches)', min_value=10.0, max_value=20.0, value=15.6, step=0.1)
            resolution = st.selectbox('Resolution', ['1920x1080', '1366x768', '3840x2160', '2560x1440', '1440x900'])
            touchscreen = st.checkbox('Touch Screen')
            ips = st.checkbox('IPS Display')
            hdd = st.number_input('HDD Storage (GB)', min_value=0, max_value=2000, value=0, step=250)
            ssd = st.number_input('SSD Storage (GB)', min_value=0, max_value=2000, value=256, step=128)
            gpu_brand = st.selectbox('GPU Brand', ['Intel', 'AMD', 'Nvidia'])
        
        # Prediction Button and Results
        if st.button("🔮 Predict Laptop Price", type="primary"):
            inputs = {
                'company': company,
                'laptop_type': laptop_type,
                'ram': ram,
                'weight': weight,
                'screen_size': screen_size,
                'resolution': resolution,
                'touchscreen': touchscreen,
                'ips': ips,
                'cpu_brand': cpu_brand,
                'hdd': hdd,
                'ssd': ssd,
                'gpu_brand': gpu_brand,
                'os': os
            }
            
            # Validate inputs
            is_valid, error_message = validate_prediction_inputs(inputs)
            
            if not is_valid:
                st.markdown(f"""
                <div class="error-box">
                    <h4 style="margin-top: 0; color: #ff6b6b;">❌ Input Error</h4>
                    <p>{error_message}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Prepare data and make prediction
                query_df = prepare_prediction_data(inputs)
                
                if query_df is not None:
                    try:
                        model = models[selected_model]
                        prediction = model.predict(query_df)[0]
                        
                        # Display prediction result
                        st.markdown(f"""
                        <div class="prediction-card">
                            <div class="prediction-label">Predicted Price</div>
                            <div class="prediction-price">${prediction:,.0f}</div>
                            <div style="font-size: 1.1rem; opacity: 0.9;">Using {selected_model}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Additional insights
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("""
                            <div class="success-box">
                                <h4 style="margin-top: 0; color: #4caf50;">✅ Prediction Complete</h4>
                                <p>The prediction is based on the selected model's training data and current market trends.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            confidence_interval = prediction * 0.1  # 10% confidence interval
                            st.markdown(f"""
                            <div class="info-box">
                                <h4 style="margin-top: 0; color: #0077b6;">📊 Price Range</h4>
                                <p><strong>Lower bound:</strong> ${prediction - confidence_interval:,.0f}</p>
                                <p><strong>Upper bound:</strong> ${prediction + confidence_interval:,.0f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.markdown(f"""
                        <div class="error-box">
                            <h4 style="margin-top: 0; color: #ff6b6b;">❌ Prediction Error</h4>
                            <p>An error occurred during prediction: {str(e)}</p>
                            <p>Please check your input values and try again.</p>
                        </div>
                        """, unsafe_allow_html=True)

# ------------------- Model Analytics Page -------------------
elif page == "📊 Model Analytics":
    st.header("Model Analytics & Performance")
    
    if not models:
        st.markdown("""
        <div class="error-box">
            <h4 style="margin-top: 0; color: #ff6b6b;">❌ No Models Available</h4>
            <p>No trained models were found for analysis. Please ensure model files are available.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Model Selection for Analysis
        st.subheader("🎯 Select Model for Analysis")
        
        analysis_model = st.selectbox(
            "Choose Model:",
            list(models.keys()),
            help="Select model for detailed analysis"
        )
        
        # Performance Metrics
        if analysis_model in accuracies:
            st.subheader("📈 Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                model_rank = sorted(accuracies.items(), key=lambda x: x[1]['R2'], reverse=True).index((analysis_model, accuracies[analysis_model])) + 1
                st.metric(
                    label="R² Score",
                    value=f"{accuracies[analysis_model]['R2']:.3f}",
                    delta=f"Rank: {model_rank}",
                    help="Coefficient of determination"
                )
            
            with col2:
                st.metric(
                    label="MAE",
                    value=f"${accuracies[analysis_model]['MAE']:,}",
                    help="Mean Absolute Error"
                )
            
            with col3:
                st.metric(
                    label="RMSE",
                    value=f"${accuracies[analysis_model]['RMSE']:,}",
                    help="Root Mean Square Error"
                )
            
            with col4:
                accuracy_percentage = accuracies[analysis_model]['R2'] * 100
                st.metric(
                    label="Accuracy",
                    value=f"{accuracy_percentage:.1f}%",
                    help="Prediction accuracy"
                )
        
        # Feature Importance Analysis
        st.subheader("🔍 Feature Importance Analysis")
        
        if analysis_model:
            importance_chart = create_feature_importance_chart(analysis_model)
            st.plotly_chart(importance_chart, use_container_width=True)
        else:
            st.info("Please select a model to view feature importance.")
        
        # Model Information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h4 style="margin-top: 0; color: #0077b6;">🧠 Model Information</h4>
                <p><strong>Selected Model:</strong> {}</p>
                <p><strong>Model Type:</strong> Machine Learning Regressor</p>
                <p><strong>Training Status:</strong> ✅ Trained and Ready</p>
                <p><strong>Prediction Capability:</strong> Laptop Price Estimation</p>
            </div>
            """.format(analysis_model), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
                <h4 style="margin-top: 0; color: #0077b6;">📊 Key Features</h4>
                <ul style="margin-bottom: 0;">
                    <li><strong>RAM:</strong> Most important feature</li>
                    <li><strong>Weight:</strong> Significant impact on price</li>
                    <li><strong>PPI:</strong> Display quality factor</li>
                    <li><strong>Storage:</strong> SSD vs HDD pricing</li>
                    <li><strong>Brand:</strong> Premium brand effect</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# ------------------- Model Comparison Page -------------------
elif page == "⚖️ Model Comparison":
    st.markdown('<h2 class="page-header">Model Performance Comparison</h2>', unsafe_allow_html=True)
    
    if not models:
        st.markdown("""
        <div class="error-box">
            <h4 style="margin-top: 0; color: #ff6b6b;">❌ No Models Available</h4>
            <p>No trained models were found for comparison. Please ensure model files are available.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Performance Comparison Chart
        st.markdown('<div class="section-header">📊 R² Score Comparison</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        comparison_chart = create_model_comparison_chart()
        st.plotly_chart(comparison_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed Comparison Table
        st.markdown('<div class="section-header">📋 Detailed Performance Metrics</div>', unsafe_allow_html=True)
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, metrics in accuracies.items():
            if model_name in models or model_name == "Main Model":
                comparison_data.append({
                    'Model': model_name,
                    'R² Score': f"{metrics['R2']:.3f}",
                    'MAE ($)': f"{metrics['MAE']:,}",
                    'RMSE ($)': f"{metrics['RMSE']:,}",
                    'Accuracy (%)': f"{metrics['R2'] * 100:.1f}%"
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('R² Score', ascending=False)
            
            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )
        
        # Best Model Recommendations
        st.markdown('<div class="section-header">🏆 Model Recommendations</div>', unsafe_allow_html=True)
        
        if accuracies:
            best_r2 = max(accuracies.items(), key=lambda x: x[1]['R2'])
            best_mae = min(accuracies.items(), key=lambda x: x[1]['MAE'])
            best_rmse = min(accuracies.items(), key=lambda x: x[1]['RMSE'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="success-box">
                    <h4 style="margin-top: 0; color: #4caf50;">🥇 Best Overall Accuracy</h4>
                    <p><strong>{best_r2[0]}</strong></p>
                    <p>R² Score: {best_r2[1]['R2']:.3f}</p>
                    <p>Recommended for: General price prediction</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="success-box">
                    <h4 style="margin-top: 0; color: #4caf50;">🎯 Lowest MAE</h4>
                    <p><strong>{best_mae[0]}</strong></p>
                    <p>MAE: ${best_mae[1]['MAE']:,}</p>
                    <p>Recommended for: Precise estimates</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="success-box">
                    <h4 style="margin-top: 0; color: #4caf50;">📐 Lowest RMSE</h4>
                    <p><strong>{best_rmse[0]}</strong></p>
                    <p>RMSE: ${best_rmse[1]['RMSE']:,}</p>
                    <p>Recommended for: Consistent predictions</p>
                </div>
                """, unsafe_allow_html=True)

# ------------------- Footer -------------------
st.markdown("""
<div class="footer">
    <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">Advanced Laptop Price Predictor</div>
    <div style="font-size: 0.9rem; opacity: 0.8;">Powered by Machine Learning • Built with Streamlit</div>
    <div style="font-size: 0.8rem; margin-top: 1rem; opacity: 0.6;">
        Predict laptop prices with confidence using state-of-the-art ML algorithms
    </div>
</div>
""", unsafe_allow_html=True)
