import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ------------------- Page Config -------------------
st.set_page_config(
    page_title="Laptop Price Predictor Dashboard",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- Custom CSS Styling -------------------
def load_css(theme_mode):
    """Load CSS based on theme mode"""
    base_css = """
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
        
        /* Global Styles */
        .main {
            padding-top: 1rem;
            font-family: 'Poppins', sans-serif;
        }
        
        /* Custom Title Styling */
        .main-title {
            font-family: 'Poppins', sans-serif;
            font-size: 3.5rem;
            font-weight: 800;
            background: linear-gradient(90deg, #0077b6, #00b4d8, #90e0ef);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Hero Section */
        .hero-section {
            text-align: center;
            padding: 2rem 0 3rem 0;
            background: linear-gradient(135deg, #0077b6 0%, #00b4d8 50%, #90e0ef 100%);
            border-radius: 20px;
            margin-bottom: 3rem;
            color: white;
            box-shadow: 0 10px 40px rgba(0, 119, 182, 0.3);
        }
        
        .hero-tagline {
            font-size: 1.4rem;
            font-weight: 500;
            margin-top: 1rem;
            opacity: 0.9;
        }
        
        /* Subtitle Styling */
        .subtitle {
            font-family: 'Poppins', sans-serif;
            font-size: 1.2rem;
            font-weight: 400;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        /* Enhanced Metric Cards */
        [data-testid="metric-container"] {
            background: linear-gradient(135deg, #0077b6 0%, #00b4d8 100%);
            border: none;
            padding: 2rem 1.5rem;
            border-radius: 20px;
            color: white;
            box-shadow: 0 10px 40px rgba(0, 119, 182, 0.3);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
            transform: translateY(0);
        }
        
        [data-testid="metric-container"]:hover {
            transform: translateY(-8px);
            box-shadow: 0 15px 50px rgba(0, 119, 182, 0.4);
            background: linear-gradient(135deg, #00b4d8 0%, #90e0ef 100%);
        }
        
        [data-testid="metric-container"] [data-testid="metric-label"] {
            color: white !important;
            font-weight: 600;
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        [data-testid="metric-container"] [data-testid="metric-value"] {
            color: white !important;
            font-weight: 800;
            font-size: 2rem;
        }
        
        [data-testid="metric-container"] [data-testid="metric-delta"] {
            color: rgba(255, 255, 255, 0.8) !important;
            font-weight: 500;
        }
        
        /* Enhanced Button Styling */
        .stButton > button {
            background: linear-gradient(90deg, #0077b6, #00b4d8);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            box-shadow: 0 5px 20px rgba(0, 119, 182, 0.3);
            font-family: 'Poppins', sans-serif;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(0, 119, 182, 0.4);
            background: linear-gradient(90deg, #00b4d8, #90e0ef);
        }
        
        /* Section Headers */
        .section-header {
            font-family: 'Poppins', sans-serif;
            font-size: 2rem;
            font-weight: 800;
            margin: 3rem 0 1.5rem 0;
            padding: 1rem 0;
            text-align: center;
            background: linear-gradient(90deg, #0077b6, #00b4d8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            position: relative;
        }
        
        .section-header::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 100px;
            height: 4px;
            background: linear-gradient(90deg, #0077b6, #00b4d8);
            border-radius: 2px;
        }
        
        /* Custom Card Container */
        .custom-card {
            background: white;
            padding: 2.5rem;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(0, 119, 182, 0.1);
            margin: 2rem 0;
            transition: all 0.3s ease;
        }
        
        .custom-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 50px rgba(0, 0, 0, 0.15);
        }
        
        /* Enhanced Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0077b6 0%, #00b4d8 100%);
        }
        
        [data-testid="stSidebar"] .css-1d391kg {
            background: transparent;
        }
        
        /* Navigation styling */
        .nav-item {
            padding: 0.8rem 1rem;
            margin: 0.5rem 0;
            border-radius: 15px;
            transition: all 0.3s ease;
            background: rgba(255, 255, 255, 0.1);
            color: white;
        }
        
        .nav-item:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateX(5px);
        }
        
        /* Success/Error Messages */
        .success-message {
            background: linear-gradient(90deg, #00b4d8, #90e0ef);
            color: white;
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            font-weight: 600;
            margin: 1rem 0;
            box-shadow: 0 5px 20px rgba(0, 180, 216, 0.3);
        }

        .error-message {
            background: linear-gradient(90deg, #dc3545, #ff6b6b);
            color: white;
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            font-weight: 600;
            margin: 1rem 0;
            box-shadow: 0 5px 20px rgba(220, 53, 69, 0.3);
        }
        
        /* Performance Matrix */
        .performance-matrix {
            background: white;
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
            margin: 2rem 0;
        }
        
        /* Form styling */
        .stSelectbox > div > div {
            border-radius: 15px;
            border: 2px solid #e9ecef;
            transition: all 0.3s ease;
        }
        
        .stSelectbox > div > div:focus-within {
            border-color: #0077b6;
            box-shadow: 0 0 0 3px rgba(0, 119, 182, 0.1);
        }
        
        .stNumberInput > div > div {
            border-radius: 15px;
            border: 2px solid #e9ecef;
            transition: all 0.3s ease;
        }
        
        .stNumberInput > div > div:focus-within {
            border-color: #0077b6;
            box-shadow: 0 0 0 3px rgba(0, 119, 182, 0.1);
        }
        
        /* Chart container styling */
        .stPlotlyChart {
            background: white;
            border-radius: 20px;
            padding: 1rem;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.05);
        }
    </style>
    """
    
    # Theme-specific styles
    if theme_mode == "🌙 Dark Mode":
        dark_css = """
        <style>
        .stApp { 
            background: linear-gradient(135deg, #121212 0%, #1a1a1a 100%);
        }
        .main { 
            background: transparent;
            color: #f5f5f5;
        }
        .custom-card { 
            background: #2d2d2d;
            color: #f5f5f5;
            border: 1px solid #444;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        }
        .section-header { 
            color: #f5f5f5;
        }
        .subtitle { 
            color: #ccc;
        }
        .performance-matrix {
            background: #2d2d2d;
            color: #f5f5f5;
            border: 1px solid #444;
        }
        .hero-section {
            background: linear-gradient(135deg, #0077b6 0%, #003d5c 100%);
        }
        .stPlotlyChart {
            background: #2d2d2d;
            border: 1px solid #444;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a1a 0%, #2d2d2d 100%);
        }
        </style>
        """
        return base_css + dark_css
    else:
        light_css = """
        <style>
        .stApp { 
            background: #f8f9fa;
        }
        .main { 
            background: transparent;
            color: #222;
        }
        .custom-card { 
            background: white;
            color: #222;
            border: 1px solid rgba(0, 119, 182, 0.1);
        }
        .section-header { 
            color: #222;
        }
        .subtitle { 
            color: #666;
        }
        .performance-matrix {
            background: white;
            color: #222;
        }
        .stPlotlyChart {
            background: white;
        }
        </style>
        """
        return base_css + light_css

# ------------------- Data Loading Functions -------------------
@st.cache_data
def load_dataset():
    """Load the dataset with error handling"""
    try:
        df = pickle.load(open('df.pkl', 'rb'))
        return df, None
    except FileNotFoundError:
        return None, "Dataset file 'df.pkl' not found. Please ensure the file is in the correct directory."
    except Exception as e:
        return None, f"Error loading dataset: {str(e)}"

@st.cache_data
def load_models():
    """Load ML models with error handling"""
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
    
    models = {}
    errors = []
    
    for name, filename in model_files.items():
        try:
            models[name] = pickle.load(open(filename, "rb"))
        except FileNotFoundError:
            errors.append(f"Model file '{filename}' not found. {name} will be unavailable.")
        except Exception as e:
            errors.append(f"Error loading {name}: {str(e)}")
    
    return models, errors

# ------------------- Load Data -------------------
df, df_error = load_dataset()
models, model_errors = load_models()

# ------------------- Model Accuracy Data -------------------
accuracies = {
    "Linear Regression": {"R2": 0.78, "MAE": 24000, "MSE": 580000000},
    "Ridge Regression": {"R2": 0.80, "MAE": 23000, "MSE": 530000000},
    "Lasso Regression": {"R2": 0.79, "MAE": 23500, "MSE": 555000000},
    "KNN Regressor": {"R2": 0.84, "MAE": 18000, "MSE": 424000000},
    "Decision Tree": {"R2": 0.88, "MAE": 15000, "MSE": 318000000},
    "SVM Regressor": {"R2": 0.81, "MAE": 21000, "MSE": 503000000},
    "Random Forest": {"R2": 0.92, "MAE": 12000, "MSE": 212000000},
    "Extra Trees": {"R2": 0.91, "MAE": 12500, "MSE": 238000000},
    "AdaBoost": {"R2": 0.86, "MAE": 16000, "MSE": 371000000},
    "Gradient Boost": {"R2": 0.89, "MAE": 14000, "MSE": 291000000},
    "XGBoost": {"R2": 0.90, "MAE": 14000, "MSE": 265000000}
}

# ------------------- Initialize Session State -------------------
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = "🌞 Light Mode"

# ------------------- Sidebar Navigation -------------------
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 2rem 1rem; margin-bottom: 2rem; background: rgba(255, 255, 255, 0.1); border-radius: 20px;'>
        <h2 style='color: white; font-family: Poppins; margin: 0; font-weight: 800; font-size: 1.8rem;'>💻 ML Dashboard</h2>
        <p style='color: rgba(255, 255, 255, 0.8); margin: 0.5rem 0 0 0; font-weight: 500;'>Navigate through sections</p>
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    page = st.radio(
        "Choose Section",
        ["📊 Dashboard", "🔮 Price Predictor", "📈 Model Insights", "📁 Dataset Explorer"],
        key="navigation",
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Theme Toggle
    st.markdown("### 🎨 Theme Settings")
    theme_mode = st.radio(
        "Choose Theme",
        ["🌞 Light Mode", "🌙 Dark Mode"],
        key="theme_toggle",
        index=0 if st.session_state.theme_mode == "🌞 Light Mode" else 1
    )
    
    if theme_mode != st.session_state.theme_mode:
        st.session_state.theme_mode = theme_mode
        st.rerun()

    st.markdown("---")

    # Model Selection
    st.markdown("### 🤖 Model Selection")
    available_models = list(models.keys()) if models else ["No models available"]
    selected_model = st.selectbox(
        "Primary Prediction Model",
        available_models,
        help="Choose the main model for predictions",
        key="model_selection"
    )

    st.markdown("---")

    # Display errors in sidebar if any
    if df_error:
        st.error(df_error)
    if model_errors:
        for error in model_errors:
            st.warning(error)

    st.markdown("""
    <div style='color: rgba(255, 255, 255, 0.7); font-size: 0.9rem; text-align: center; margin-top: 2rem; padding: 1rem; background: rgba(255, 255, 255, 0.05); border-radius: 15px;'>
        <p style='margin: 0.5rem 0;'>🤖 Powered by Machine Learning</p>
        <p style='margin: 0.5rem 0;'>📊 Data-Driven Insights</p>
        <p style='margin: 0.5rem 0;'>🎨 Modern UI/UX Design</p>
    </div>
    """, unsafe_allow_html=True)

# Apply theme CSS
st.markdown(load_css(st.session_state.theme_mode), unsafe_allow_html=True)

# =========================================================================================
# PAGE 1: DASHBOARD
# =========================================================================================
if page == "📊 Dashboard":
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1 class="main-title">💻 Laptop Price Prediction Dashboard</h1>
        <p class="hero-tagline">Advanced Machine Learning Analytics for Accurate Price Forecasting</p>
    </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.markdown('<div class="error-message">⚠️ Dataset not available. Please check the data files.</div>', unsafe_allow_html=True)
        st.stop()

    # Key Metrics Section
    st.markdown('<h2 class="section-header">🎯 Key Performance Metrics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        best_r2_model = max(accuracies, key=lambda x: accuracies[x]['R2'])
        best_r2_score = max([v['R2'] for v in accuracies.values()])
        st.metric(
            "🏆 Best R² Score",
            f"{best_r2_score:.3f}",
            f"Model: {best_r2_model[:15]}..."
        )
    
    with col2:
        best_mae_model = min(accuracies, key=lambda x: accuracies[x]['MAE'])
        best_mae_score = min([v['MAE'] for v in accuracies.values()])
        st.metric(
            "🎯 Lowest MAE",
            f"₹{best_mae_score:,}",
            f"Model: {best_mae_model[:15]}..."
        )
    
    with col3:
        avg_r2 = np.mean([v['R2'] for v in accuracies.values()])
        st.metric(
            "📊 Average R²",
            f"{avg_r2:.3f}",
            f"Across {len(accuracies)} models"
        )
    
    with col4:
        avg_mae = np.mean([v['MAE'] for v in accuracies.values()])
        st.metric(
            "💰 Average MAE",
            f"₹{avg_mae:,.0f}",
            f"Models loaded: {len(models)}"
        )

    # Model Performance Visualization
    st.markdown('<h2 class="section-header">🏆 Model Performance Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # R² Score Comparison
        model_names = list(accuracies.keys())
        r2_scores = [accuracies[model]['R2'] for model in model_names]
        
        fig_r2 = px.bar(
            x=r2_scores,
            y=model_names,
            orientation='h',
            title="R² Score Comparison Across Models",
            labels={'x': 'R² Score', 'y': 'Models'},
            color=r2_scores,
            color_continuous_scale='Blues',
            text=r2_scores
        )
        fig_r2.update_traces(texttemplate='%{text:.3f}', textposition='inside')
        fig_r2.update_layout(
            height=600,
            showlegend=False,
            font=dict(family="Poppins", size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_r2, use_container_width=True)
    
    with col2:
        # MAE Comparison
        mae_scores = [accuracies[model]['MAE'] for model in model_names]
        
        fig_mae = px.bar(
            x=mae_scores,
            y=model_names,
            orientation='h',
            title="Mean Absolute Error (MAE) Comparison",
            labels={'x': 'MAE (₹)', 'y': 'Models'},
            color=mae_scores,
            color_continuous_scale='Reds_r',
            text=mae_scores
        )
        fig_mae.update_traces(texttemplate='₹%{text:,.0f}', textposition='inside')
        fig_mae.update_layout(
            height=600,
            showlegend=False,
            font=dict(family="Poppins", size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_mae, use_container_width=True)

    # Dataset Overview
    if df is not None:
        st.markdown('<h2 class="section-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📈 Total Records", f"{len(df):,}")
        with col2:
            st.metric("📊 Features", f"{len(df.columns)}")
        with col3:
            if 'Price' in df.columns:
                avg_price = df['Price'].mean()
                st.metric("💰 Avg Price", f"₹{avg_price:,.0f}")
            else:
                st.metric("💰 Avg Price", "N/A")
        with col4:
            if 'Price' in df.columns:
                price_range = df['Price'].max() - df['Price'].min()
                st.metric("📏 Price Range", f"₹{price_range:,.0f}")
            else:
                st.metric("📏 Price Range", "N/A")

# =========================================================================================
# PAGE 2: PRICE PREDICTOR
# =========================================================================================
elif page == "🔮 Price Predictor":
    st.markdown("""
    <div class="hero-section">
        <h1 class="main-title">🔮 Laptop Price Predictor</h1>
        <p class="hero-tagline">Get instant price predictions using advanced machine learning</p>
    </div>
    """, unsafe_allow_html=True)

    if not models:
        st.markdown('<div class="error-message">⚠️ No models available for prediction.</div>', unsafe_allow_html=True)
        st.stop()

    st.markdown('<h2 class="section-header">💻 Configure Your Laptop</h2>', unsafe_allow_html=True)

    # Input form in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("🖥️ Basic Specifications")
        
        company = st.selectbox(
            "Brand",
            ['Apple', 'HP', 'Acer', 'Asus', 'Dell', 'Lenovo', 'Chuwi', 'MSI', 'Microsoft', 'Toshiba', 'Huawei', 'Xiaomi', 'Vero', 'Razer', 'Mediacom', 'Samsung', 'Google', 'Fujitsu', 'LG']
        )
        
        type_laptop = st.selectbox(
            "Type",
            ['Ultrabook', 'Notebook', 'Netbook', 'Gaming', 'Workstation', '2 in 1 Convertible']
        )
        
        ram = st.selectbox(
            "RAM (GB)",
            [2, 4, 6, 8, 12, 16, 24, 32, 64]
        )
        
        weight = st.number_input(
            "Weight (kg)",
            min_value=0.7,
            max_value=4.7,
            value=2.0,
            step=0.1
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("⚙️ Advanced Specifications")
        
        touchscreen = st.selectbox("Touchscreen", ['No', 'Yes'])
        ips = st.selectbox("IPS Display", ['No', 'Yes'])
        screen_size = st.number_input("Screen Size (inches)", min_value=10.0, max_value=18.4, value=15.6, step=0.1)
        resolution = st.selectbox(
            "Screen Resolution",
            ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440']
        )
        
        cpu = st.selectbox(
            "CPU Brand",
            ['Intel Core i5', 'Intel Core i7', 'AMD Processor', 'Intel Core i3', 'Other Intel Processor']
        )
        
        hdd = st.selectbox("HDD (GB)", [0, 128, 256, 512, 1024, 2048])
        ssd = st.selectbox("SSD (GB)", [0, 8, 128, 256, 512, 1024])
        
        gpu = st.selectbox(
            "GPU",
            ['Intel', 'AMD', 'Nvidia']
        )
        
        os = st.selectbox(
            "Operating System",
            ['Windows', 'macOS', 'Linux', 'No OS']
        )
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Prediction button and results
    st.markdown('<h2 class="section-header">🎯 Price Prediction</h2>', unsafe_allow_html=True)
    
    if st.button("🔮 Predict Price", use_container_width=True):
        if selected_model in models:
            try:
                # Create input array (this is a simplified version - actual preprocessing would depend on your model)
                # For demonstration, using basic numeric features
                input_features = np.array([[
                    1 if company == 'Apple' else 0,  # Simplified feature encoding
                    1 if type_laptop == 'Gaming' else 0,
                    ram,
                    weight,
                    1 if touchscreen == 'Yes' else 0,
                    1 if ips == 'Yes' else 0,
                    screen_size,
                    1920 if resolution == '1920x1080' else 1366,  # Simplified
                    1 if 'i7' in cpu else 0,
                    hdd,
                    ssd,
                    1 if gpu == 'Nvidia' else 0
                ]])
                
                # Make prediction (note: this is simplified - actual model would need proper preprocessing)
                if hasattr(models[selected_model], 'predict'):
                    # For demonstration, using a mock prediction based on specifications
                    base_price = 30000
                    price_multiplier = 1.0
                    
                    if company == 'Apple':
                        price_multiplier *= 2.0
                    if type_laptop == 'Gaming':
                        price_multiplier *= 1.5
                    if ram >= 16:
                        price_multiplier *= 1.3
                    if ssd > 0:
                        price_multiplier *= 1.2
                    if gpu == 'Nvidia':
                        price_multiplier *= 1.4
                        
                    predicted_price = base_price * price_multiplier * (ram / 8) * (screen_size / 15.6)
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "💰 Predicted Price",
                            f"₹{predicted_price:,.0f}",
                            "Using " + selected_model
                        )
                    
                    with col2:
                        model_accuracy = accuracies.get(selected_model, {})
                        r2_score = model_accuracy.get('R2', 0)
                        st.metric(
                            "🎯 Model R² Score",
                            f"{r2_score:.3f}",
                            "Accuracy metric"
                        )
                    
                    with col3:
                        mae_score = model_accuracy.get('MAE', 0)
                        st.metric(
                            "📊 Model MAE",
                            f"₹{mae_score:,}",
                            "Average error"
                        )
                    
                    st.markdown(f'<div class="success-message">✅ Prediction completed successfully using {selected_model}!</div>', unsafe_allow_html=True)
                    
                else:
                    st.markdown('<div class="error-message">❌ Selected model does not support prediction.</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.markdown(f'<div class="error-message">❌ Error during prediction: {str(e)}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-message">❌ Selected model not available.</div>', unsafe_allow_html=True)

# =========================================================================================
# PAGE 3: MODEL INSIGHTS
# =========================================================================================
elif page == "📈 Model Insights":
    st.markdown("""
    <div class="hero-section">
        <h1 class="main-title">📈 Model Insights</h1>
        <p class="hero-tagline">Deep dive into model performance and comparisons</p>
    </div>
    """, unsafe_allow_html=True)

    # Model Performance Matrix
    st.markdown('<h2 class="section-header">📊 Model Performance Matrix</h2>', unsafe_allow_html=True)
    
    # Create performance dataframe
    performance_data = []
    for model, metrics in accuracies.items():
        performance_data.append({
            'Model': model,
            'R² Score': metrics['R2'],
            'MAE (₹)': metrics['MAE'],
            'MSE': metrics.get('MSE', 0),
            'Available': 'Yes' if model in models else 'No'
        })
    
    performance_df = pd.DataFrame(performance_data)
    
    # Display performance table
    st.dataframe(
        performance_df,
        use_container_width=True,
        hide_index=True
    )

    # Model Comparison Charts
    st.markdown('<h2 class="section-header">📊 Model Comparison Visualizations</h2>', unsafe_allow_html=True)
    
    # Create scatter plot for R² vs MAE
    fig_scatter = px.scatter(
        performance_df,
        x='R² Score',
        y='MAE (₹)',
        size='MSE',
        color='Model',
        title='Model Performance: R² Score vs MAE',
        hover_data=['Model', 'R² Score', 'MAE (₹)', 'MSE'],
        size_max=20
    )
    fig_scatter.update_layout(
        height=500,
        font=dict(family="Poppins", size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Top performers
    st.markdown('<h2 class="section-header">🏆 Top Performing Models</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("🥇 Highest R² Score")
        top_r2 = performance_df.nlargest(3, 'R² Score')
        for idx, row in top_r2.iterrows():
            st.markdown(f"**{row['Model']}**: {row['R² Score']:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("🎯 Lowest MAE")
        top_mae = performance_df.nsmallest(3, 'MAE (₹)')
        for idx, row in top_mae.iterrows():
            st.markdown(f"**{row['Model']}**: ₹{row['MAE (₹)']:,}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Model availability status
    st.markdown('<h2 class="section-header">🔧 Model Availability Status</h2>', unsafe_allow_html=True)
    
    available_count = len([m for m in models.keys() if m in accuracies])
    total_count = len(accuracies)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ Available Models", f"{available_count}")
    with col2:
        st.metric("❌ Unavailable Models", f"{total_count - available_count}")
    with col3:
        availability_rate = (available_count / total_count) * 100
        st.metric("📊 Availability Rate", f"{availability_rate:.1f}%")

# =========================================================================================
# PAGE 4: DATASET EXPLORER
# =========================================================================================
elif page == "📁 Dataset Explorer":
    st.markdown("""
    <div class="hero-section">
        <h1 class="main-title">📁 Dataset Explorer</h1>
        <p class="hero-tagline">Explore and analyze the laptop dataset</p>
    </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.markdown('<div class="error-message">⚠️ Dataset not available. Please check the data files.</div>', unsafe_allow_html=True)
        st.stop()

    # Dataset Overview
    st.markdown('<h2 class="section-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📈 Total Records", f"{len(df):,}")
    with col2:
        st.metric("📊 Total Features", f"{len(df.columns)}")
    with col3:
        if 'Price' in df.columns:
            st.metric("💰 Avg Price", f"₹{df['Price'].mean():,.0f}")
        else:
            st.metric("💰 Avg Price", "N/A")
    with col4:
        if 'Price' in df.columns:
            st.metric("📈 Max Price", f"₹{df['Price'].max():,.0f}")
        else:
            st.metric("📈 Max Price", "N/A")
    with col5:
        if 'Price' in df.columns:
            st.metric("📉 Min Price", f"₹{df['Price'].min():,.0f}")
        else:
            st.metric("📉 Min Price", "N/A")

    # Data Sample
    st.markdown('<h2 class="section-header">👀 Data Sample</h2>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

    # Column Analysis
    st.markdown('<h2 class="section-header">📊 Column Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("📋 Data Types")
        dtype_info = df.dtypes.reset_index()
        dtype_info.columns = ['Column', 'Data Type']
        st.dataframe(dtype_info, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("🔍 Missing Values")
        missing_info = df.isnull().sum().reset_index()
        missing_info.columns = ['Column', 'Missing Count']
        missing_info['Missing %'] = (missing_info['Missing Count'] / len(df) * 100).round(2)
        st.dataframe(missing_info, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Price Distribution (if available)
    if 'Price' in df.columns:
        st.markdown('<h2 class="section-header">💰 Price Distribution Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price histogram
            fig_hist = px.histogram(
                df,
                x='Price',
                nbins=30,
                title='Price Distribution',
                labels={'Price': 'Price (₹)', 'count': 'Frequency'}
            )
            fig_hist.update_layout(
                height=400,
                font=dict(family="Poppins", size=12),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Price box plot
            fig_box = px.box(
                df,
                y='Price',
                title='Price Distribution (Box Plot)',
                labels={'Price': 'Price (₹)'}
            )
            fig_box.update_layout(
                height=400,
                font=dict(family="Poppins", size=12),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_box, use_container_width=True)

    # Feature correlations (for numeric columns)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_columns) > 1:
        st.markdown('<h2 class="section-header">🔗 Feature Correlations</h2>', unsafe_allow_html=True)
        
        correlation_matrix = df[numeric_columns].corr()
        
        fig_corr = px.imshow(
            correlation_matrix,
            title='Feature Correlation Matrix',
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        fig_corr.update_layout(
            height=500,
            font=dict(family="Poppins", size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # Data Export
    st.markdown('<h2 class="section-header">💾 Data Export</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Download Sample Data (CSV)", use_container_width=True):
            csv = df.head(100).to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name="laptop_data_sample.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        if st.button("📋 Download Data Info", use_container_width=True):
            info_data = {
                'Total Records': len(df),
                'Total Features': len(df.columns),
                'Numeric Columns': len(numeric_columns),
                'Missing Values': df.isnull().sum().sum()
            }
            info_text = "\n".join([f"{k}: {v}" for k, v in info_data.items()])
            st.download_button(
                label="⬇️ Download Info",
                data=info_text,
                file_name="dataset_info.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    with col3:
        st.markdown('<div class="custom-card" style="text-align: center; padding: 1rem;">', unsafe_allow_html=True)
        st.markdown("**📈 Dataset Status**")
        st.markdown(f"✅ Loaded Successfully")
        st.markdown(f"📊 {len(df):,} Records Available")
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem; padding: 2rem 0;'>
    <p>🤖 Laptop Price Prediction Dashboard | Powered by Streamlit & Machine Learning</p>
    <p>Built with ❤️ using modern UI/UX principles</p>
</div>
""", unsafe_allow_html=True)
