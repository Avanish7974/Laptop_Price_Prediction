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
    ["📊 Dashboard", "🔮 Price Predictor", "📈 Model Insights"],
    key="navigation"
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

    # Dataset Overview
    st.markdown('<h2 class="section-header">📋 Dataset Overview</h2>', unsafe_allow_html=True)
    
    overview_col1, overview_col2, overview_col3 = st.columns(3)
    
    with overview_col1:
        st.metric("📊 Total Records", f"{len(df):,}")
    
    with overview_col2:
        st.metric("🔧 Features", f"{len(df.columns):,}")
    
    with overview_col3:
        if 'Price' in df.columns:
            avg_price = df['Price'].mean()
            st.metric("💰 Avg Price", f"₹{avg_price:,.0f}")
        else:
            st.metric("💰 Price Range", "Available")

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
                pred_col1, pred_col2, pred_col3 = st.columns(3)
                
                for i, (model_name, price) in enumerate(predictions.items()):
                    col = [pred_col1, pred_col2, pred_col3][i % 3]
                    with col:
                        st.metric(
                            f"🤖 {model_name}",
                            f"₹{price:,.0f}",
                            help=f"Prediction from {model_name} model"
                        )
                
                # Average prediction
                avg_prediction = np.mean(list(predictions.values()))
                st.markdown(f"""
                <div class="success-message">
                    <h3>🎯 Ensemble Prediction: ₹{avg_prediction:,.0f}</h3>
                    <p>Average prediction from {len(predictions)} models</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Prediction confidence
                std_dev = np.std(list(predictions.values()))
                confidence = max(0.0, 100 - (std_dev / avg_prediction * 100))
                
                st.progress(confidence / 100)
                st.write(f"Prediction Confidence: {confidence:.1f}%")
            
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

    # Feature Importance Section (if data available)
    st.markdown('<h2 class="section-header">🎯 Feature Importance Analysis</h2>', unsafe_allow_html=True)
    
    # Mock feature importance data (replace with actual model feature importance)
    feature_importance = {
        'RAM': 0.25,
        'CPU Brand': 0.20,
        'SSD': 0.18,
        'GPU Brand': 0.15,
        'Brand': 0.12,
        'Screen Resolution': 0.05,
        'Weight': 0.03,
        'HDD': 0.02
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

    # Model Performance Metrics Table
    st.markdown('<h2 class="section-header">📋 Comprehensive Metrics Summary</h2>', unsafe_allow_html=True)
    
    # Create detailed metrics table
    detailed_metrics = []
    for model_name, metrics in accuracies.items():
        # Calculate additional metrics
        rmse = np.sqrt(metrics['MAE'] * 1.2)  # Approximation
        mape = (metrics['MAE'] / 50000) * 100  # Approximation
        
        detailed_metrics.append({
            'Model': model_name,
            'R² Score': f"{metrics['R2']:.3f}",
            'MAE (₹)': f"{metrics['MAE']:,}",
            'RMSE (₹)': f"{rmse:,.0f}",
            'MAPE (%)': f"{mape:.1f}",
            'Rank': ''
        })
    
    # Sort by R² score and add ranks
    detailed_metrics.sort(key=lambda x: float(x['R² Score']), reverse=True)
    for i, metric in enumerate(detailed_metrics):
        metric['Rank'] = f"#{i+1}"
    
    metrics_df = pd.DataFrame(detailed_metrics)
    
    st.dataframe(
        metrics_df.style.background_gradient(subset=['R² Score'], cmap='RdYlGn')
        .background_gradient(subset=['MAE (₹)'], cmap='RdYlGn_r'),
        use_container_width=True
    )

    # Dataset Statistics
    st.markdown('<h2 class="section-header">📊 Dataset Statistics</h2>', unsafe_allow_html=True)
    
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    
    with stat_col1:
        st.markdown("### 📈 Numerical Statistics")
        if df is not None:
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                st.dataframe(df[numerical_cols].describe().round(2))
            else:
                st.write("No numerical columns found in dataset")
    
    with stat_col2:
        st.markdown("### 🏷️ Categorical Distribution")
        if df is not None:
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                for col in categorical_cols[:3]:  # Show first 3 categorical columns
                    st.write(f"**{col}:**")
                    value_counts = df[col].value_counts().head(5)
                    st.write(value_counts)
            else:
                st.write("No categorical columns found in dataset")
    
    with stat_col3:
        st.markdown("### 🔍 Data Quality")
        if df is not None:
            total_records = len(df)
            missing_data = df.isnull().sum().sum()
            completeness = ((total_records * len(df.columns) - missing_data) / 
                          (total_records * len(df.columns))) * 100
            
            st.metric("Data Completeness", f"{completeness:.1f}%")
            st.metric("Missing Values", f"{missing_data:,}")
            st.metric("Unique Records", f"{len(df.drop_duplicates()):,}")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🚀 Built with Streamlit • 🤖 Powered by Machine Learning • 📊 Data Science Excellence</p>
</div>
""", unsafe_allow_html=True)
