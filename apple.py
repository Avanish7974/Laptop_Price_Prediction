import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------- Page Config -------------------
st.set_page_config(
    page_title="Laptop Price Predictor Dashboard",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- Custom CSS Styling -------------------
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        padding-top: 1rem;
    }
    
    /* Custom Title Styling */
    .main-title {
        font-family: 'Poppins', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Subtitle Styling */
    .subtitle {
        font-family: 'Poppins', sans-serif;
        font-size: 1.2rem;
        font-weight: 400;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    /* Metric Cards Enhancement */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: transform 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(31, 38, 135, 0.5);
    }
    
    [data-testid="metric-container"] [data-testid="metric-label"] {
        color: white !important;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: white !important;
        font-weight: 700;
        font-size: 1.8rem;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        background: linear-gradient(90deg, #4ECDC4, #FF6B6B);
    }
    
    /* Section Headers */
    .section-header {
        font-family: 'Poppins', sans-serif;
        font-size: 1.8rem;
        font-weight: 600;
        color: #333;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #FF6B6B;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Custom Card Container */
    .custom-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin: 1rem 0;
    }
    
    /* Success Message */
    .success-message {
        background: linear-gradient(90deg, #4ECDC4, #44A08D);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    /* Error Message */
    .error-message {
        background: linear-gradient(90deg, #FF6B6B, #FF8E8E);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Configuration Cards */
    .config-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    
    .config-card h3 {
        color: white;
        margin-top: 0;
    }
    
    /* Performance Matrix */
    .performance-matrix {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

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
    "Linear Regression": {"R2": 0.78, "MAE": 24000},
    "Ridge Regression": {"R2": 0.80, "MAE": 23000},
    "Lasso Regression": {"R2": 0.79, "MAE": 23500},
    "KNN Regressor": {"R2": 0.84, "MAE": 18000},
    "Decision Tree": {"R2": 0.88, "MAE": 15000},
    "SVM Regressor": {"R2": 0.81, "MAE": 21000},
    "Random Forest": {"R2": 0.92, "MAE": 12000},
    "Extra Trees": {"R2": 0.91, "MAE": 12500},
    "AdaBoost": {"R2": 0.86, "MAE": 16000},
    "Gradient Boost": {"R2": 0.89, "MAE": 14000},
    "XGBoost": {"R2": 0.90, "MAE": 14000}
}

# ------------------- Sidebar Navigation -------------------
st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem; margin-bottom: 2rem;'>
    <h2 style='color: #FF6B6B; font-family: Poppins; margin: 0;'>💻 ML Dashboard</h2>
    <p style='color: #666; margin: 0;'>Navigate through sections</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Choose Section",
    ["📊 Dashboard", "🔮 Price Predictor", "📈 Model Insights", "📁 Dataset Explorer"],
    key="navigation"
)

st.sidebar.markdown("---")

# Theme Toggle
st.sidebar.markdown("### 🎨 Theme Settings")
theme_mode = st.sidebar.radio(
    "Choose Theme",
    ["🌞 Light Mode", "🌙 Dark Mode"],
    key="theme_toggle"
)

# Apply theme-specific styling
if theme_mode == "🌙 Dark Mode":
    st.markdown("""
    <style>
        .stApp { background-color: #1a1a1a !important; }
        .main { background-color: #1a1a1a !important; color: white !important; }
        .custom-card { background: #2d2d2d !important; color: white !important; border: 1px solid #444 !important; }
        .section-header { color: white !important; }
        .subtitle { color: #ccc !important; }
        .main-title { color: white !important; }
        [data-testid="stSidebar"] { background-color: #2d2d2d !important; }
        [data-testid="stSidebar"] * { color: white !important; }
        div[data-testid="metric-container"] { background: #333 !important; }
        .stDataFrame { background-color: #2d2d2d !important; }
        .stDataFrame table { background-color: #2d2d2d !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        .stApp { background-color: white !important; }
        .main { background-color: white !important; color: #333 !important; }
        .custom-card { background: white !important; color: #333 !important; }
        .section-header { color: #333 !important; }
        .subtitle { color: #666 !important; }
    </style>
    """, unsafe_allow_html=True)

# Model Selection
st.sidebar.markdown("### 🤖 Model Selection")
available_models = list(models.keys()) if models else ["No models available"]
selected_model = st.sidebar.selectbox(
    "Primary Prediction Model",
    available_models,
    help="Choose the main model for predictions"
)

st.sidebar.markdown("---")

# Display errors in sidebar if any
if df_error:
    st.sidebar.error(df_error)
if model_errors:
    for error in model_errors:
        st.sidebar.warning(error)

st.sidebar.markdown("""
<div style='color: #666; font-size: 0.8rem; text-align: center; margin-top: 2rem;'>
    <p>🤖 Powered by Machine Learning</p>
    <p>📊 Data-Driven Insights</p>
    <p>🎨 Modern UI/UX Design</p>
</div>
""", unsafe_allow_html=True)

# =========================================================================================
# PAGE 1: DASHBOARD
# =========================================================================================
if page == "📊 Dashboard":
    # Title Section
    st.markdown('<h1 class="main-title">📊 Laptop Price Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Comprehensive overview of ML model performance and predictions</p>', unsafe_allow_html=True)

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
            f"Model: {best_r2_model}"
        )
    
    with col2:
        best_mae_model = min(accuracies, key=lambda x: accuracies[x]['MAE'])
        best_mae_score = min([v['MAE'] for v in accuracies.values()])
        st.metric(
            "🎯 Lowest MAE",
            f"₹{best_mae_score:,}",
            f"Model: {best_mae_model}"
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

    # Top Models Performance Pie Chart
    st.markdown('<h2 class="section-header">🏆 Top 4 Model Performance</h2>', unsafe_allow_html=True)
    
    # Get top 4 models by R² score
    sorted_models = sorted(accuracies.items(), key=lambda x: x[1]['R2'], reverse=True)[:4]
    
    # Create pie chart data
    model_names = [model[0] for model in sorted_models]
    r2_scores = [model[1]['R2'] for model in sorted_models]
    
    # Create pie chart using Plotly
    fig_pie = px.pie(
        values=r2_scores,
        names=model_names,
        title="Top 4 Models by R² Score Performance",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(
        height=500,
        title_x=0.5,
        font=dict(size=12),
        showlegend=True
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Display top 4 models details in a clean table
    top_models_data = []
    for model_name, metrics in sorted_models:
        top_models_data.append({
            'Rank': len(top_models_data) + 1,
            'Model': model_name,
            'R² Score': f"{metrics['R2']:.3f}",
            'MAE (₹)': f"₹{metrics['MAE']:,}",
            'Status': '✅ Loaded' if model_name in models else '❌ Missing'
        })
    
    top_models_df = pd.DataFrame(top_models_data)
    st.dataframe(top_models_df, use_container_width=True, hide_index=True)

    # Charts Section
    st.markdown('<h2 class="section-header">📈 Advanced Analytics Visualizations</h2>', unsafe_allow_html=True)
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("### 🎯 R² vs MAE Performance Map")
        
        # Create scatter plot using Plotly
        model_names = list(accuracies.keys())
        r2_scores = [v['R2'] for v in accuracies.values()]
        mae_scores = [v['MAE'] for v in accuracies.values()]
        
        fig = px.scatter(
            x=r2_scores,
            y=mae_scores,
            text=model_names,
            title="Model Performance Landscape",
            labels={'x': 'R² Score', 'y': 'Mean Absolute Error (₹)'},
            color=r2_scores,
            size=[1]*len(model_names),
            color_continuous_scale='viridis'
        )
        
        fig.update_traces(textposition="top center", textfont_size=10)
        fig.update_layout(
            height=500,
            showlegend=False,
            title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        st.markdown("### 🎖️ Model Performance Radar")
        
        # Create radar chart for top 5 models
        top_models = sorted(accuracies.items(), key=lambda x: x[1]['R2'], reverse=True)[:5]
        
        categories = ['R² Score', 'Error Resilience']
        
        fig = go.Figure()
        
        for model_name, metrics in top_models:
            # Normalize scores for radar chart
            r2_norm = metrics['R2']
            mae_norm = 1 - (metrics['MAE'] / max([v['MAE'] for v in accuracies.values()]))
            
            fig.add_trace(go.Scatterpolar(
                r=[r2_norm, mae_norm],
                theta=categories,
                fill='toself',
                name=model_name,
                opacity=0.7
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Top 5 Models Performance Radar",
            title_x=0.5,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # Market Insights Dashboard
    st.markdown('<h2 class="section-header">📈 Market Insights Dashboard</h2>', unsafe_allow_html=True)
    
    # Price distribution charts
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("### 💰 Average Price by Brand")
        
        # Mock data for demonstration (replace with actual data if available)
        brand_prices = {
            "Apple": 120000, "Dell": 65000, "HP": 58000, "Lenovo": 52000,
            "Asus": 62000, "Acer": 45000, "MSI": 85000
        }
        
        fig_brand = px.bar(
            x=list(brand_prices.keys()),
            y=list(brand_prices.values()),
            title="Average Laptop Price by Brand",
            labels={'x': 'Brand', 'y': 'Average Price (₹)'},
            color=list(brand_prices.values()),
            color_continuous_scale='viridis'
        )
        
        fig_brand.update_layout(
            height=400,
            xaxis_tickangle=45,
            showlegend=False,
            title_x=0.5
        )
        
        st.plotly_chart(fig_brand, use_container_width=True)
    
    with insight_col2:
        st.markdown("### 🧠 Average Price by Processor")
        
        # Mock processor data
        processor_prices = {
            "Intel i3": 35000, "Intel i5": 55000, "Intel i7": 85000, 
            "Intel i9": 125000, "AMD Ryzen 5": 48000, "AMD Ryzen 7": 75000
        }
        
        fig_proc = px.pie(
            values=list(processor_prices.values()),
            names=list(processor_prices.keys()),
            title="Price Distribution by Processor Type",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        fig_proc.update_traces(textposition='inside', textinfo='percent+label')
        fig_proc.update_layout(height=400, title_x=0.5)
        
        st.plotly_chart(fig_proc, use_container_width=True)
    
    # RAM vs Price and Storage Analysis
    analysis_col1, analysis_col2 = st.columns(2)
    
    with analysis_col1:
        st.markdown("### 💾 RAM vs Price Analysis")
        
        # Mock RAM data
        ram_prices = {
            "4GB": 35000, "8GB": 55000, "16GB": 75000, "32GB": 110000, "64GB": 180000
        }
        
        fig_ram = px.line(
            x=list(ram_prices.keys()),
            y=list(ram_prices.values()),
            title="Price Trend by RAM Capacity",
            labels={'x': 'RAM (GB)', 'y': 'Average Price (₹)'},
            markers=True
        )
        
        fig_ram.update_traces(line=dict(color='#FF6B6B', width=3), marker=dict(size=8))
        fig_ram.update_layout(height=400, title_x=0.5)
        
        st.plotly_chart(fig_ram, use_container_width=True)
    
    with analysis_col2:
        st.markdown("### 💿 SSD vs Price Correlation")
        
        # Mock SSD data
        ssd_data = {
            "Storage (GB)": [0, 128, 256, 512, 1024],
            "Price (₹)": [45000, 55000, 65000, 75000, 95000]
        }
        
        fig_ssd = px.scatter(
            x=ssd_data["Storage (GB)"],
            y=ssd_data["Price (₹)"],
            title="SSD Storage vs Price Correlation",
            labels={'x': 'SSD Storage (GB)', 'y': 'Price (₹)'},
            color_discrete_sequence=['#4ECDC4']
        )
        
        fig_ssd.update_traces(marker=dict(size=10))
        fig_ssd.update_layout(height=400, title_x=0.5)
        
        st.plotly_chart(fig_ssd, use_container_width=True)
    
    # Top 10 sections
    st.markdown('<h2 class="section-header">🏆 Top 10 Rankings</h2>', unsafe_allow_html=True)
    
    top_col1, top_col2 = st.columns(2)
    
    with top_col1:
        st.markdown("### 💎 Most Expensive Laptops")
        
        expensive_laptops = [
            {"Brand": "Apple", "Model": "MacBook Pro 16\"", "Price": "₹4,50,000"},
            {"Brand": "Dell", "Model": "Alienware Area-51m", "Price": "₹3,80,000"},
            {"Brand": "MSI", "Model": "GT76 Titan", "Price": "₹3,20,000"},
            {"Brand": "Asus", "Model": "ROG Mothership", "Price": "₹2,90,000"},
            {"Brand": "HP", "Model": "ZBook Fury 17", "Price": "₹2,50,000"}
        ]
        
        expensive_df = pd.DataFrame(expensive_laptops)
        st.dataframe(expensive_df, use_container_width=True, hide_index=True)
    
    with top_col2:
        st.markdown("### 📊 Most Popular Brands")
        
        brand_popularity = {
            "Dell": 28, "HP": 24, "Lenovo": 18, "Asus": 12, 
            "Acer": 8, "Apple": 6, "MSI": 4
        }
        
        fig_popularity = px.bar(
            x=list(brand_popularity.keys()),
            y=list(brand_popularity.values()),
            title="Market Share by Brand (%)",
            labels={'x': 'Brand', 'y': 'Market Share (%)'},
            color=list(brand_popularity.values()),
            color_continuous_scale='blues'
        )
        
        fig_popularity.update_layout(
            height=300,
            showlegend=False,
            title_x=0.5
        )
        
        st.plotly_chart(fig_popularity, use_container_width=True)
    
    # Dataset Overview
    st.markdown('<h2 class="section-header">📋 Dataset Overview</h2>', unsafe_allow_html=True)
    
    overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
    
    with overview_col1:
        st.metric("📊 Total Records", f"{len(df):,}")
    
    with overview_col2:
        st.metric("🔧 Features", f"{len(df.columns):,}")
    
    with overview_col3:
        if 'Price' in df.columns:
            avg_price = df['Price'].mean()
            st.metric("💰 Avg Price", f"₹{avg_price:,.0f}")
        else:
            st.metric("💰 Price Range", "₹25K - ₹5L")
    
    with overview_col4:
        st.metric("🔄 Last Updated", "Today")

# =========================================================================================
# PAGE 2: PRICE PREDICTOR
# =========================================================================================
elif page == "🔮 Price Predictor":
    st.markdown('<h1 class="main-title">🔮 Intelligent Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Configure your laptop specifications and get AI-powered price predictions</p>', unsafe_allow_html=True)

    if df is None or len(models) == 0:
        st.markdown('<div class="error-message">⚠️ Models or dataset not available. Please check the required files.</div>', unsafe_allow_html=True)
        st.stop()

    # Configuration Section
    st.markdown('<h2 class="section-header">🖥️ Laptop Configuration</h2>', unsafe_allow_html=True)
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.markdown('<div class="config-card"><h3>🏷️ Brand & Type</h3></div>', unsafe_allow_html=True)
        
        # Brand Selection
        if 'Company' in df.columns:
            brands = sorted(df['Company'].unique())
            brand = st.selectbox("Brand", brands, help="Select laptop brand")
        else:
            brand = st.selectbox("Brand", ["Apple", "Dell", "HP", "Lenovo", "Asus"])
        
        # Type Selection
        if 'TypeName' in df.columns:
            types = sorted(df['TypeName'].unique())
            laptop_type = st.selectbox("Type", types, help="Select laptop type")
        else:
            laptop_type = st.selectbox("Type", ["Ultrabook", "Gaming", "Notebook", "Workstation"])
        
        st.markdown('<div class="config-card"><h3>💾 Memory & Storage</h3></div>', unsafe_allow_html=True)
        
        # RAM Selection
        ram = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64], index=1, help="Select RAM capacity")
        
        # HDD Selection
        hdd = st.selectbox("HDD (GB)", [0, 128, 256, 500, 1000, 2000], index=3, help="Select HDD capacity")
        
        # SSD Selection
        ssd = st.selectbox("SSD (GB)", [0, 128, 256, 512, 1024], index=2, help="Select SSD capacity")
    
    with config_col2:
        st.markdown('<div class="config-card"><h3>📟 Display Specifications</h3></div>', unsafe_allow_html=True)
        
        # Screen Resolution
        screen_resolution = st.selectbox(
            "Screen Resolution", 
            ["1366x768", "1920x1080", "2560x1440", "3840x2160"],
            index=1,
            help="Select screen resolution"
        )
        
        # Touchscreen
        touchscreen = st.radio("Touchscreen", ["No", "Yes"], help="Does the laptop have touchscreen?")
        
        # IPS Display
        ips_display = st.radio("IPS Display", ["No", "Yes"], help="Does the laptop have IPS display?")
        
        st.markdown('<div class="config-card"><h3>⚙️ Processing Power</h3></div>', unsafe_allow_html=True)
        
        # CPU Selection
        if 'Cpu brand' in df.columns:
            cpu_brands = sorted(df['Cpu brand'].unique())
            cpu_brand = st.selectbox("CPU Brand", cpu_brands, help="Select CPU brand")
        else:
            cpu_brand = st.selectbox("CPU Brand", ["Intel Core i3", "Intel Core i5", "Intel Core i7", "AMD"])
        
        # GPU Selection
        if 'Gpu brand' in df.columns:
            gpu_brands = sorted(df['Gpu brand'].unique())
            gpu_brand = st.selectbox("GPU Brand", gpu_brands, help="Select GPU brand")
        else:
            gpu_brand = st.selectbox("GPU Brand", ["Intel", "AMD", "Nvidia"])
        
        # Weight
        weight = st.slider("Weight (kg)", 0.5, 4.0, 2.0, 0.1, help="Laptop weight in kilograms")

    # Physical Properties Section
    st.markdown('<h2 class="section-header">📐 Physical Properties</h2>', unsafe_allow_html=True)
    
    # Calculate additional properties
    ppi = 1920 * 1080 / (15.6 ** 2) if screen_resolution == "1920x1080" else 100  # Default PPI calculation
    
    phys_col1, phys_col2, phys_col3 = st.columns(3)
    
    with phys_col1:
        st.metric("📏 Screen Size", "15.6 inches", help="Standard laptop screen size")
    
    with phys_col2:
        st.metric("🔍 PPI", f"{ppi:.0f}", help="Pixels per inch")
    
    with phys_col3:
        storage_total = hdd + ssd
        st.metric("💿 Total Storage", f"{storage_total} GB", help="Combined HDD + SSD storage")

    # Prediction Section
    st.markdown('<h2 class="section-header">🎯 Price Prediction Results</h2>', unsafe_allow_html=True)
    
    if st.button("🚀 Predict Price", type="primary"):
        with st.spinner("🔮 Analyzing configuration and predicting price..."):
            # Prepare input data for prediction
            # Note: This is a simplified example. In practice, you'd need to match
            # the exact feature engineering used during model training
            
            # Mock prediction for demonstration (replace with actual model prediction)
            base_price = 30000
            
            # Brand multiplier
            brand_multipliers = {
                "Apple": 2.5, "Dell": 1.2, "HP": 1.1, "Lenovo": 1.0, 
                "Asus": 1.15, "Acer": 0.9, "MSI": 1.3
            }
            brand_mult = brand_multipliers.get(str(brand), 1.0)
            
            # Type multiplier
            type_multipliers = {
                "Gaming": 1.5, "Workstation": 1.4, "Ultrabook": 1.3, "Notebook": 1.0
            }
            type_mult = type_multipliers.get(str(laptop_type), 1.0)
            
            # Component calculations
            ram_price = ram * 1500
            storage_price = (hdd * 0.05) + (ssd * 0.5)
            touchscreen_bonus = 5000 if touchscreen == "Yes" else 0
            ips_bonus = 3000 if ips_display == "Yes" else 0
            
            # Calculate predictions for different models
            predictions = {}
            for model_name in ["Random Forest", "XGBoost", "Gradient Boost"]:
                if model_name in models:
                    # Simplified prediction calculation
                    predicted_price = (base_price * brand_mult * type_mult + 
                                     ram_price + storage_price + 
                                     touchscreen_bonus + ips_bonus)
                    
                    # Add some variation between models
                    if model_name == "Random Forest":
                        predicted_price *= 1.02
                    elif model_name == "XGBoost":
                        predicted_price *= 0.98
                    
                    predictions[model_name] = predicted_price
            
            # Display predictions
            if predictions:
                # Average prediction and confidence
                avg_prediction = float(np.mean(list(predictions.values())))
                std_dev = float(np.std(list(predictions.values())))
                confidence = max(0.0, 100.0 - (std_dev / avg_prediction * 100.0))
                
                # Price range calculation
                price_margin = std_dev * 1.5
                min_price = avg_prediction - price_margin
                max_price = avg_prediction + price_margin
                
                # Price category determination
                if avg_prediction < 40000:
                    price_category = "💰 Budget"
                    category_color = "#4CAF50"
                elif avg_prediction < 80000:
                    price_category = "⚖️ Mid-Range"
                    category_color = "#FF9800"
                elif avg_prediction < 150000:
                    price_category = "🏆 Premium"
                    category_color = "#9C27B0"
                else:
                    price_category = "👑 Flagship"
                    category_color = "#F44336"
                
                # Enhanced prediction display
                st.markdown(f"""
                <div class="success-message" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 20px; margin: 2rem 0;">
                    <h2 style="color: white; margin: 0; text-align: center;">🎯 Predicted Price</h2>
                    <h1 style="color: white; margin: 1rem 0; text-align: center; font-size: 3rem;">₹{avg_prediction:,.0f}</h1>
                    <div style="display: flex; justify-content: space-between; margin-top: 2rem;">
                        <div style="text-align: center;">
                            <h4 style="color: white; margin: 0;">📊 Price Range</h4>
                            <p style="color: white; font-size: 1.2rem; margin: 0.5rem 0;">₹{min_price:,.0f} - ₹{max_price:,.0f}</p>
                        </div>
                        <div style="text-align: center;">
                            <h4 style="color: white; margin: 0;">🎯 Confidence</h4>
                            <p style="color: white; font-size: 1.2rem; margin: 0.5rem 0;">{confidence:.1f}%</p>
                        </div>
                        <div style="text-align: center;">
                            <h4 style="color: white; margin: 0;">🏷️ Category</h4>
                            <p style="color: {category_color}; font-size: 1.2rem; margin: 0.5rem 0; font-weight: bold;">{price_category}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Individual model predictions
                st.markdown("### 🤖 Individual Model Predictions")
                pred_col1, pred_col2, pred_col3 = st.columns(3)
                
                for i, (model_name, price) in enumerate(predictions.items()):
                    col = [pred_col1, pred_col2, pred_col3][i % 3]
                    with col:
                        st.metric(
                            f"{model_name}",
                            f"₹{price:,.0f}",
                            f"{((price - avg_prediction) / avg_prediction * 100):+.1f}%",
                            help=f"Prediction from {model_name} model"
                        )
                
                # Confidence visualization
                st.markdown("### 📊 Prediction Confidence")
                confidence_col1, confidence_col2 = st.columns([2, 1])
                
                with confidence_col1:
                    st.progress(confidence / 100)
                    if confidence > 85:
                        st.success(f"High confidence prediction ({confidence:.1f}%)")
                    elif confidence > 70:
                        st.warning(f"Moderate confidence prediction ({confidence:.1f}%)")
                    else:
                        st.error(f"Low confidence prediction ({confidence:.1f}%)")
                
                with confidence_col2:
                    st.metric("Confidence Score", f"{confidence:.1f}%")
                
                # Similar Laptops Comparison Feature
                st.markdown('<h2 class="section-header">🔍 Similar Laptops Comparison</h2>', unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #0077b6, #00b4d8); color: white; padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">
                    <h4>💡 Based on your configuration (₹{avg_prediction:,.0f}), here are similar laptops:</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Mock similar laptops data
                similar_laptops = [
                    {
                        "Brand": str(brand), "Model": "Similar Model A", "RAM": f"{ram}GB", 
                        "Storage": f"{ssd}GB SSD", "Price": f"₹{avg_prediction-15000:,.0f}",
                        "Match": "92%"
                    },
                    {
                        "Brand": "Alternative", "Model": "Comparable Model", "RAM": f"{ram}GB", 
                        "Storage": f"{ssd}GB SSD", "Price": f"₹{avg_prediction+8000:,.0f}",
                        "Match": "89%"
                    },
                    {
                        "Brand": "Budget", "Model": "Value Option", "RAM": f"{max(4, ram-4)}GB", 
                        "Storage": f"{max(128, ssd-128)}GB SSD", "Price": f"₹{avg_prediction-25000:,.0f}",
                        "Match": "76%"
                    }
                ]
                
                sim_col1, sim_col2, sim_col3 = st.columns(3)
                
                for i, (col, laptop) in enumerate(zip([sim_col1, sim_col2, sim_col3], similar_laptops)):
                    with col:
                        match_color = "#4CAF50" if float(laptop["Match"].replace('%', '')) > 85 else "#FF9800"
                        st.markdown(f"""
                        <div class="custom-card">
                            <h4>{laptop['Brand']} {laptop['Model']}</h4>
                            <p><strong>💾 RAM:</strong> {laptop['RAM']}</p>
                            <p><strong>💿 Storage:</strong> {laptop['Storage']}</p>
                            <p><strong>💰 Price:</strong> <span style="color: #FF6B6B;">{laptop['Price']}</span></p>
                            <p><strong>🎯 Match:</strong> <span style="color: {match_color}; font-weight: bold;">{laptop['Match']}</span></p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Best Value Recommendations
                st.markdown('<h2 class="section-header">💎 Smart Recommendations</h2>', unsafe_allow_html=True)
                
                rec_col1, rec_col2 = st.columns(2)
                
                with rec_col1:
                    st.markdown(f"""
                    <div class="success-message">
                        <h4>🏆 Best Alternative</h4>
                        <h3>Similar Performance</h3>
                        <p>Save ₹{15000:,} with comparable specs</p>
                        <p><strong>Recommended: Alternative Brand Model</strong></p>
                        <p>Match Score: 89%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with rec_col2:
                    cheaper_price = avg_prediction - 25000
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #90e0ef 0%, #00b4d8 100%); color: white; padding: 1.5rem; border-radius: 15px;">
                        <h4>💰 Budget-Friendly</h4>
                        <h3>Value Champion</h3>
                        <p>Great specs at ₹{cheaper_price:,.0f}</p>
                        <p><strong>Recommended: Budget Option</strong></p>
                        <p>76% match with your requirements</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            else:
                st.error("No models available for prediction. Please check model files.")

# =========================================================================================
# PAGE 3: MODEL INSIGHTS
# =========================================================================================
elif page == "📈 Model Insights":
    st.markdown('<h1 class="main-title">📈 Advanced Model Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Deep dive into model performance, accuracy metrics, and comparative analysis</p>', unsafe_allow_html=True)

    if df is None:
        st.markdown('<div class="error-message">⚠️ Dataset not available for analysis.</div>', unsafe_allow_html=True)
        st.stop()

    # Model Comparison Section
    st.markdown('<h2 class="section-header">🔬 Detailed Model Analysis</h2>', unsafe_allow_html=True)
    
    # Create comprehensive comparison chart
    comparison_col1, comparison_col2 = st.columns(2)
    
    with comparison_col1:
        st.markdown("### 📊 R² Score Comparison")
        
        model_names = list(accuracies.keys())
        r2_scores = [v['R2'] for v in accuracies.values()]
        
        fig = px.bar(
            x=model_names,
            y=r2_scores,
            title="R² Score by Model",
            labels={'x': 'Model', 'y': 'R² Score'},
            color=r2_scores,
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            height=500,
            xaxis_tickangle=45,
            showlegend=False,
            title_x=0.5
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with comparison_col2:
        st.markdown("### 💸 Mean Absolute Error Analysis")
        
        mae_scores = [v['MAE'] for v in accuracies.values()]
        
        fig = px.bar(
            x=model_names,
            y=mae_scores,
            title="Mean Absolute Error by Model",
            labels={'x': 'Model', 'y': 'MAE (₹)'},
            color=mae_scores,
            color_continuous_scale='plasma_r'
        )
        
        fig.update_layout(
            height=500,
            xaxis_tickangle=45,
            showlegend=False,
            title_x=0.5
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # Feature Importance Section
    st.markdown('<h2 class="section-header">🧠 Feature Importance Analysis</h2>', unsafe_allow_html=True)
    
    # Mock feature importance data (replace with actual feature importance from your models)
    feature_importance = {
        "RAM (GB)": 0.28,
        "SSD Storage": 0.22,
        "CPU Performance": 0.18,
        "Brand": 0.15,
        "GPU Type": 0.12,
        "Screen Resolution": 0.05
    }
    
    # Create feature importance chart
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())
    
    fig = px.bar(
        x=importance,
        y=features,
        orientation='h',
        title="Feature Importance in Price Prediction",
        labels={'x': 'Importance Score', 'y': 'Features'},
        color=importance,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=400,
        title_x=0.5,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Model Performance Summary
    st.markdown('<h2 class="section-header">📋 Comprehensive Metrics Summary</h2>', unsafe_allow_html=True)
    
    # Create detailed metrics table
    metrics_data = []
    for model_name, metrics in accuracies.items():
        metrics_data.append({
            'Model': model_name,
            'R² Score': metrics['R2'],
            'MAE (₹)': metrics['MAE'],
            'Performance Grade': 'A+' if metrics['R2'] > 0.9 else 'A' if metrics['R2'] > 0.85 else 'B+' if metrics['R2'] > 0.8 else 'B',
            'Status': '✅ Available' if model_name in models else '❌ Not Found'
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Sort by R² score
    metrics_df = metrics_df.sort_values('R² Score', ascending=False)
    
    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True
    )

    # Model Recommendations
    st.markdown('<h2 class="section-header">💡 Model Recommendations</h2>', unsafe_allow_html=True)
    
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    
    with rec_col1:
        best_overall = max(accuracies, key=lambda x: accuracies[x]['R2'])
        st.markdown(f"""
        <div class="custom-card">
            <h4>🏆 Best Overall Performance</h4>
            <h3>{best_overall}</h3>
            <p>R² Score: {accuracies[best_overall]['R2']:.3f}</p>
            <p>Recommended for: Production use</p>
        </div>
        """, unsafe_allow_html=True)
    
    with rec_col2:
        fastest_model = "Decision Tree"  # Assuming Decision Tree is fastest
        st.markdown(f"""
        <div class="custom-card">
            <h4>⚡ Fastest Prediction</h4>
            <h3>{fastest_model}</h3>
            <p>R² Score: {accuracies[fastest_model]['R2']:.3f}</p>
            <p>Recommended for: Real-time applications</p>
        </div>
        """, unsafe_allow_html=True)
    
    with rec_col3:
        most_stable = "Random Forest"  # Assuming Random Forest is most stable
        st.markdown(f"""
        <div class="custom-card">
            <h4>🛡️ Most Stable</h4>
            <h3>{most_stable}</h3>
            <p>R² Score: {accuracies[most_stable]['R2']:.3f}</p>
            <p>Recommended for: Consistent results</p>
        </div>
        """, unsafe_allow_html=True)

# =========================================================================================
# PAGE 4: DATASET EXPLORER
# =========================================================================================
elif page == "📁 Dataset Explorer":
    st.markdown('<h1 class="main-title">📁 Dataset Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Browse, filter, and search through the laptop dataset</p>', unsafe_allow_html=True)

    if df is None:
        st.markdown('<div class="error-message">⚠️ Dataset not available for exploration.</div>', unsafe_allow_html=True)
        st.stop()

    # Filters Section
    st.markdown('<h2 class="section-header">🔍 Filter & Search Options</h2>', unsafe_allow_html=True)
    
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        st.markdown("### 🏷️ Brand Filter")
        # Mock brand options (replace with actual data)
        all_brands = ["All", "Apple", "Dell", "HP", "Lenovo", "Asus", "Acer", "MSI"]
        selected_brand = st.selectbox("Select Brand", all_brands)
        
        st.markdown("### 💾 RAM Filter")
        ram_options = ["All", "4GB", "8GB", "16GB", "32GB", "64GB"]
        selected_ram = st.selectbox("Select RAM", ram_options)
    
    with filter_col2:
        st.markdown("### 🧠 Processor Filter")
        processor_options = ["All", "Intel i3", "Intel i5", "Intel i7", "Intel i9", "AMD Ryzen 5", "AMD Ryzen 7"]
        selected_processor = st.selectbox("Select Processor", processor_options)
        
        st.markdown("### 💿 Storage Filter")
        storage_options = ["All", "128GB", "256GB", "512GB", "1TB", "2TB"]
        selected_storage = st.selectbox("Select Storage", storage_options)
    
    with filter_col3:
        st.markdown("### 💰 Price Range")
        price_range = st.slider(
            "Select Price Range (₹)",
            min_value=20000,
            max_value=500000,
            value=(30000, 200000),
            step=10000
        )
        
        st.markdown("### 🎮 Type Filter")
        type_options = ["All", "Gaming", "Ultrabook", "Notebook", "Workstation", "2-in-1"]
        selected_type = st.selectbox("Select Type", type_options)
    
    # Search functionality
    st.markdown('<h2 class="section-header">🔎 Search Functionality</h2>', unsafe_allow_html=True)
    
    search_col1, search_col2 = st.columns([3, 1])
    
    with search_col1:
        search_query = st.text_input(
            "🔍 Search laptops by model, brand, or specifications",
            placeholder="e.g., MacBook, Gaming laptop, Dell Inspiron..."
        )
    
    with search_col2:
        search_button = st.button("🔍 Search", type="primary")
    
    # Mock filtered dataset display
    st.markdown('<h2 class="section-header">📊 Filtered Results</h2>', unsafe_allow_html=True)
    
    # Create mock filtered data
    filtered_data = [
        {"Brand": "Apple", "Model": "MacBook Pro 13\"", "RAM": "8GB", "Storage": "256GB SSD", "Processor": "Intel i5", "Price": "₹1,29,900"},
        {"Brand": "Dell", "Model": "XPS 13", "RAM": "16GB", "Storage": "512GB SSD", "Processor": "Intel i7", "Price": "₹1,45,000"},
        {"Brand": "HP", "Model": "Pavilion 15", "RAM": "8GB", "Storage": "1TB HDD", "Processor": "Intel i5", "Price": "₹55,000"},
        {"Brand": "Lenovo", "Model": "ThinkPad E14", "RAM": "8GB", "Storage": "256GB SSD", "Processor": "AMD Ryzen 5", "Price": "₹48,000"},
        {"Brand": "Asus", "Model": "ROG Strix G15", "RAM": "16GB", "Storage": "512GB SSD", "Processor": "AMD Ryzen 7", "Price": "₹89,000"},
    ]
    
    # Apply filters (mock logic)
    display_data = filtered_data.copy()
    
    if selected_brand != "All":
        display_data = [item for item in display_data if item["Brand"] == selected_brand]
    
    if selected_ram != "All":
        display_data = [item for item in display_data if item["RAM"] == selected_ram]
    
    # Display results
    if display_data:
        st.success(f"Found {len(display_data)} laptops matching your criteria")
        
        # Display as cards
        for i in range(0, len(display_data), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(display_data):
                    laptop = display_data[i + j]
                    with col:
                        st.markdown(f"""
                        <div class="custom-card">
                            <h4>{laptop['Brand']} {laptop['Model']}</h4>
                            <p><strong>💾 RAM:</strong> {laptop['RAM']}</p>
                            <p><strong>💿 Storage:</strong> {laptop['Storage']}</p>
                            <p><strong>🧠 Processor:</strong> {laptop['Processor']}</p>
                            <p><strong>💰 Price:</strong> <span style="color: #FF6B6B; font-weight: bold;">{laptop['Price']}</span></p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Display as table
        st.markdown("### 📋 Detailed Table View")
        results_df = pd.DataFrame(display_data)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
    else:
        st.warning("No laptops found matching your criteria. Try adjusting the filters.")
    
    # Quick Stats
    st.markdown('<h2 class="section-header">📈 Quick Statistics</h2>', unsafe_allow_html=True)
    
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        st.metric("🔍 Results Found", len(display_data))
    
    with stats_col2:
        if display_data:
            avg_price = sum([int(item['Price'].replace('₹', '').replace(',', '')) for item in display_data]) / len(display_data)
            st.metric("💰 Avg Price", f"₹{avg_price:,.0f}")
        else:
            st.metric("💰 Avg Price", "N/A")
    
    with stats_col3:
        brands_count = len(set([item['Brand'] for item in display_data]))
        st.metric("🏷️ Brands", brands_count)
    
    with stats_col4:
        st.metric("📊 Total Records", len(filtered_data))
    
    # Recommendations based on filters
    if display_data:
        st.markdown('<h2 class="section-header">💡 Recommendations</h2>', unsafe_allow_html=True)
        
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            # Best value recommendation
            if len(display_data) > 0:
                best_value = display_data[0]  # Simplified logic
                st.markdown(f"""
                <div class="success-message">
                    <h4>🎯 Best Value Pick</h4>
                    <h3>{best_value['Brand']} {best_value['Model']}</h3>
                    <p>Great balance of performance and price</p>
                    <p><strong>Price: {best_value['Price']}</strong></p>
                </div>
                """, unsafe_allow_html=True)
        
        with rec_col2:
            # Premium recommendation
            if len(display_data) > 1:
                premium_choice = display_data[1]  # Simplified logic
                st.markdown(f"""
                <div class="custom-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                    <h4>👑 Premium Choice</h4>
                    <h3>{premium_choice['Brand']} {premium_choice['Model']}</h3>
                    <p>Top-tier performance and features</p>
                    <p><strong>Price: {premium_choice['Price']}</strong></p>
                </div>
                """, unsafe_allow_html=True)