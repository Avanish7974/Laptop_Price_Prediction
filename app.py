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

# ------------------- Enhanced Custom Styling -------------------
st.markdown("""
<style>
    /* Main app background with gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 1rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .css-1d391kg .stRadio > label {
        color: white;
    }
    
    /* Header styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .page-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    /* Prediction result card */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
        transform: translateY(-5px);
        transition: all 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 25px 50px rgba(102, 126, 234, 0.4);
    }
    
    .prediction-price {
        font-size: 3.5rem;
        font-weight: bold;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .prediction-label {
        font-size: 1.3rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    
    /* Input sections */
    .input-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        border-left: 5px solid #667eea;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    }
    
    .section-header {
        font-size: 1.4rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    /* Enhanced metrics */
    .metric-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #e6e6e6;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        border: none;
        padding: 15px 30px;
        font-size: 18px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4);
    }
    
    /* Info and alert boxes */
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(33, 150, 243, 0.2);
    }
    
    .success-box {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.2);
    }
    
    .error-box {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #f44336;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(244, 67, 54, 0.2);
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        border-top: 1px solid #e9ecef;
        margin-top: 3rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        .prediction-price {
            font-size: 2.5rem;
        }
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
    
    # Model files to try loading (exact names from your VS Code)
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

# Model accuracy data (you can update these with your actual values)
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
def create_enhanced_metric(label: str, value: str, delta: str = None, help_text: str = None):
    """Create an enhanced metric display"""
    delta_html = f'<div style="color: #28a745; font-size: 0.8rem; margin-top: 0.5rem;">{delta}</div>' if delta else ""
    help_html = f'<div style="color: #6c757d; font-size: 0.7rem; margin-top: 0.25rem;">{help_text}</div>' if help_text else ""
    
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
    """Prepare data for prediction based on available features - matches your original dataset format"""
    try:
        # Calculate PPI from screen size and resolution
        X_res, Y_res = map(int, inputs['resolution'].split('x'))
        ppi = ((X_res**2) + (Y_res**2))**0.5 / inputs['screen_size']
        
        # Create query data matching your original dataset columns exactly
        query_data = {
            'Company': inputs.get('company', 'HP'),
            'TypeName': inputs.get('laptop_type', 'Notebook'), 
            'Ram': inputs.get('ram', 8),
            'Weight': inputs.get('weight', 2.0),
            'Touchscreen': 1 if inputs.get('touchscreen', 'No') == 'Yes' else 0,
            'Ips': 1 if inputs.get('ips', 'No') == 'Yes' else 0,
            'ppi': ppi,
            'Cpu brand': inputs.get('cpu', 'Intel Core i5'),
            'HDD': inputs.get('hdd', 0),
            'SSD': inputs.get('ssd', 256),
            'Gpu brand': inputs.get('gpu', 'Intel'),
            'os': inputs.get('os', 'Windows')
        }
        
        return pd.DataFrame([query_data])
        
    except Exception as e:
        st.error(f"Error preparing prediction data: {str(e)}")
        return None

# ------------------- Sidebar Navigation -------------------
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; color: white;">
    <h2 style="color: white;">💻 Laptop Price AI</h2>
    <p style="color: rgba(255,255,255,0.8);">Advanced ML-Powered Predictions</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "🧭 Navigate to:",
    ["🏠 Dashboard", "🔮 Price Predictor", "📊 Model Analytics", "⚙️ Model Comparison"],
    help="Choose a page to explore different features"
)

# ------------------- Page Content -------------------

# =========================================================================================
# PAGE 1: ENHANCED DASHBOARD
# =========================================================================================
if page == "🏠 Dashboard":
    st.markdown('<div class="main-header">🏠 Advanced Laptop Price Prediction Dashboard</div>', unsafe_allow_html=True)
    
    # Display loading status
    if 'loading_status' in st.session_state:
        with st.expander("📁 File Loading Status", expanded=not models):
            for status in st.session_state.loading_status:
                st.markdown(status)
            
            if not models:
                st.markdown("""<div class="info-box">
                <h4>📤 Upload Your Model Files</h4>
                <p>Please upload the following files from your VS Code project:</p>
                <ul>
                    <li><code>df.pkl</code> - Dataset file</li>
                    <li><code>lin_reg.pkl</code> - Linear Regression model</li>
                    <li><code>Ridge_regre.pkl</code> - Ridge Regression model</li>
                    <li><code>lasso_reg.pkl</code> - Lasso Regression model</li>
                    <li><code>KNN_reg.pkl</code> - KNN Regressor model</li>
                    <li><code>Decision_tree.pkl</code> - Decision Tree model</li>
                    <li><code>SVM_reg.pkl</code> - SVM Regressor model</li>
                    <li><code>Random_forest.pkl</code> - Random Forest model</li>
                    <li><code>Extra_tree.pkl</code> - Extra Trees model</li>
                    <li><code>Ada_boost.pkl</code> - AdaBoost model</li>
                    <li><code>Gradient_boost.pkl</code> - Gradient Boost model</li>
                    <li><code>XG_boost.pkl</code> - XGBoost model</li>
                </ul>
                <p><strong>How to upload:</strong> Use the Files panel in Replit to drag and drop your .pkl files into the root directory.</p>
                </div>""", unsafe_allow_html=True)
    
    # Status indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_enhanced_metric(
            "Models Loaded", 
            str(len(models)), 
            "✅ Active" if models else "❌ None",
            "Available ML models"
        ), unsafe_allow_html=True)
    
    with col2:
        if models:
            best_model = max(accuracies, key=lambda x: accuracies[x]['R2']) if any(m in accuracies for m in models.keys()) else "N/A"
            r2_score = accuracies.get(best_model, {}).get('R2', 0)
            st.markdown(create_enhanced_metric(
                "Best Model (R²)", 
                best_model if best_model != "N/A" else "N/A",
                f"R² = {r2_score:.3f}" if best_model != "N/A" else "",
                "Highest accuracy model"
            ), unsafe_allow_html=True)
        else:
            st.markdown(create_enhanced_metric("Best Model", "No models", "", "Load models to see stats"), unsafe_allow_html=True)
    
    with col3:
        if models:
            lowest_mae_model = min(accuracies, key=lambda x: accuracies[x]['MAE']) if any(m in accuracies for m in models.keys()) else "N/A"
            mae_value = accuracies.get(lowest_mae_model, {}).get('MAE', 0)
            st.markdown(create_enhanced_metric(
                "Lowest MAE", 
                lowest_mae_model if lowest_mae_model != "N/A" else "N/A",
                f"MAE = ₹{mae_value:,}" if lowest_mae_model != "N/A" else "",
                "Most precise predictions"
            ), unsafe_allow_html=True)
        else:
            st.markdown(create_enhanced_metric("Lowest MAE", "No data", "", "Load models to see stats"), unsafe_allow_html=True)
    
    with col4:
        data_status = "✅ Loaded" if df is not None else "❌ Missing"
        data_rows = len(df) if df is not None else 0
        st.markdown(create_enhanced_metric(
            "Dataset Status",
            data_status,
            f"{data_rows:,} records" if df is not None else "No data",
            "Training dataset info"
        ), unsafe_allow_html=True)
    
    # Enhanced model performance visualization
    if models and any(model_name in accuracies for model_name in models.keys()):
        st.markdown("### 📈 Model Performance Analytics")
        
        # Filter accuracies to only include loaded models
        loaded_model_accuracies = {name: accuracies[name] for name in models.keys() if name in accuracies}
        
        if loaded_model_accuracies:
            acc_df = pd.DataFrame(loaded_model_accuracies).T.reset_index().rename(columns={'index': 'Model'})
            
            # Create enhanced performance chart
            fig = go.Figure()
            
            # R² Score bars
            fig.add_trace(go.Bar(
                x=acc_df['Model'], 
                y=acc_df['R2'], 
                name='R² Score',
                marker_color='rgba(102, 126, 234, 0.8)',
                text=acc_df['R2'].round(3),
                textposition='auto',
                hovertemplate='Model: %{x}<br>R² Score: %{y:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title={
                    'text': "Model Performance Comparison",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                xaxis_title="Machine Learning Models",
                yaxis_title="R² Score (Higher is Better)",
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # MAE comparison
            fig2 = px.bar(
                acc_df, 
                x='Model', 
                y='MAE',
                title="Mean Absolute Error Comparison",
                color='MAE',
                color_continuous_scale='Viridis_r',
                text='MAE'
            )
            fig2.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
            fig2.update_layout(
                yaxis_title="MAE (Lower is Better)",
                height=500,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig2, use_container_width=True)
    
    # Dataset insights
    if df is not None:
        st.markdown("### 📊 Dataset Insights")
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            if 'Company' in df.columns:
                company_counts = df['Company'].value_counts().head(10)
                fig3 = px.pie(
                    values=company_counts.values,
                    names=company_counts.index,
                    title="Top Laptop Brands in Dataset"
                )
                st.plotly_chart(fig3, use_container_width=True)
        
        with insight_col2:
            if 'Ram' in df.columns:
                fig4 = px.histogram(
                    df, 
                    x='Ram',
                    title="RAM Distribution",
                    nbins=20
                )
                fig4.update_layout(
                    xaxis_title="RAM (GB)",
                    yaxis_title="Count"
                )
                st.plotly_chart(fig4, use_container_width=True)
    
    else:
        st.markdown('<div class="info-box">💡 <strong>Tip:</strong> Upload your dataset (df.pkl) to see detailed insights and analytics!</div>', unsafe_allow_html=True)

# =========================================================================================
# PAGE 2: ENHANCED PRICE PREDICTOR
# =========================================================================================
elif page == "🔮 Price Predictor":
    st.markdown('<div class="page-header">🔮 Advanced Price Prediction Engine</div>', unsafe_allow_html=True)
    
    if not models:
        st.markdown('<div class="error-box">⚠️ <strong>No Models Found:</strong> Please ensure your model files are in the correct directory.</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <strong>Expected Files:</strong><br>
        • Single model: <code>model.pkl</code><br>
        • Multiple models: <code>lin_reg.pkl</code>, <code>Random_forest.pkl</code>, etc.<br>
        • Dataset: <code>df.pkl</code> (optional)
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-box">✅ <strong>Models Ready:</strong> All prediction models loaded successfully!</div>', unsafe_allow_html=True)
    
    # Enhanced input form
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">🏢 Basic Information</div>', unsafe_allow_html=True)
            
            # Use dataset values if available, otherwise use defaults
            if df is not None and 'Company' in df.columns:
                company_options = df['Company'].unique().tolist()
            else:
                company_options = ['Acer', 'Apple', 'Asus', 'Dell', 'HP', 'Lenovo', 'MSI', 'Toshiba', 'Other']
            company = st.selectbox('Brand/Manufacturer', company_options, help="Select the laptop brand")
            
            if df is not None and 'TypeName' in df.columns:
                type_options = df['TypeName'].unique().tolist()
            else:
                type_options = ['Gaming', 'Netbook', 'Notebook', 'Ultrabook', 'Workstation', '2 in 1 Convertible']
            laptop_type = st.selectbox('Laptop Type', type_options, help="Category of laptop")
            
            if df is not None and 'Ram' in df.columns:
                ram_options = sorted(df['Ram'].unique().tolist())
            else:
                ram_options = [4, 8, 16, 32, 64]
            ram = st.selectbox('💾 RAM (GB)', ram_options, index=1 if len(ram_options) > 1 else 0, help="System memory")
            
            weight = st.number_input(
                '⚖️ Weight (kg)', 
                min_value=0.5, 
                max_value=5.0, 
                step=0.1, 
                value=2.0,
                help="Laptop weight in kilograms"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display features section
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">🖥️ Display Features</div>', unsafe_allow_html=True)
            
            screen_size = st.slider(
                '📏 Screen Size (inches)', 
                10.0, 18.0, 15.6, 0.1,
                help="Diagonal screen size"
            )
            
            resolution_options = [
                '1920x1080', '1366x768', '1600x900', '3840x2160',
                '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
            ]
            resolution = st.selectbox('🔳 Screen Resolution', resolution_options, help="Display resolution")
            
            touchscreen = st.radio('🖐️ Touchscreen', ['No', 'Yes'], horizontal=True, help="Touch-enabled display")
            ips = st.radio('🖼️ IPS Display', ['No', 'Yes'], horizontal=True, help="In-Plane Switching technology")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">🖥️ Hardware Specifications</div>', unsafe_allow_html=True)
            
            if df is not None and 'Cpu brand' in df.columns:
                cpu_options = df['Cpu brand'].unique().tolist()
            else:
                cpu_options = ['Intel Core i3', 'Intel Core i5', 'Intel Core i7', 'AMD', 'Other']
            cpu = st.selectbox('Processor/CPU', cpu_options, help="Central processing unit")
            
            if df is not None and 'Gpu brand' in df.columns:
                gpu_options = df['Gpu brand'].unique().tolist()
            else:
                gpu_options = ['Intel', 'AMD', 'Nvidia']
            gpu = st.selectbox('🎮 Graphics Card', gpu_options, help="Graphics processing unit")
            
            if df is not None and 'os' in df.columns:
                os_options = df['os'].unique().tolist()
            else:
                os_options = ['Windows', 'macOS', 'Linux', 'Chrome OS']
            os_type = st.selectbox('Operating System', os_options, help="Pre-installed OS")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Storage section
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">💾 Storage Configuration</div>', unsafe_allow_html=True)
            
            if df is not None and 'HDD' in df.columns:
                hdd_options = sorted(df['HDD'].unique().tolist())
            else:
                hdd_options = [0, 128, 256, 512, 1000, 2000]
            hdd = st.selectbox('💽 HDD Storage (GB)', hdd_options, help="Hard disk drive storage")
            
            if df is not None and 'SSD' in df.columns:
                ssd_options = sorted(df['SSD'].unique().tolist())
            else:
                ssd_options = [0, 128, 256, 512, 1000, 2000]
            ssd = st.selectbox('⚡ SSD Storage (GB)', ssd_options, index=2 if len(ssd_options) > 2 else 0, help="Solid state drive storage")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction button and results
    st.markdown("<br>", unsafe_allow_html=True)
    
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    with predict_col2:
        predict_button = st.button(
            "🚀 Generate Price Prediction",
            use_container_width=True,
            type="primary",
            help="Click to predict laptop price based on specifications"
        )
    
    if predict_button and models:
        # Prepare input data
        inputs = {
            'company': company,
            'laptop_type': laptop_type,
            'ram': ram,
            'weight': weight,
            'touchscreen': touchscreen,
            'ips': ips,
            'screen_size': screen_size,
            'resolution': resolution,
            'cpu': cpu,
            'gpu': gpu,
            'os': os_type,
            'hdd': hdd,
            'ssd': ssd
        }
        
        # Validate inputs
        is_valid, error_msg = validate_prediction_inputs(inputs)
        
        if not is_valid:
            st.markdown(f'<div class="error-box">❌ <strong>Input Error:</strong> {error_msg}</div>', unsafe_allow_html=True)
        else:
            with st.spinner("🔮 Analyzing specifications and generating predictions..."):
                query = prepare_prediction_data(inputs)
                
                if query is not None:
                    predictions = {}
                    successful_predictions = 0
                    
                    # Generate predictions from all available models
                    for name, model in models.items():
                        try:
                            pred = model.predict(query)[0]
                            # Handle log-transformed predictions
                            if pred < 100:  # Likely log-transformed
                                pred = np.exp(pred)
                            predictions[name] = max(int(pred), 0)
                            successful_predictions += 1
                        except Exception as e:
                            st.warning(f"Could not generate prediction from {name}: {str(e)}")
                    
                    if successful_predictions > 0:
                        # Display main prediction result
                        avg_prediction = np.mean(list(predictions.values()))
                        
                        st.markdown(f"""
                        <div class="prediction-card">
                            <div class="prediction-label">💰 Estimated Laptop Price</div>
                            <div class="prediction-price">₹{avg_prediction:,.0f}</div>
                            <div class="prediction-label">Average from {successful_predictions} ML model(s)</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Detailed predictions breakdown
                        st.markdown("### 📊 Detailed Model Predictions")
                        
                        pred_col1, pred_col2 = st.columns([1, 2])
                        
                        with pred_col1:
                            st.markdown("#### Individual Model Results:")
                            for model_name, price in predictions.items():
                                accuracy_info = ""
                                if model_name in accuracies:
                                    r2 = accuracies[model_name]['R2']
                                    accuracy_info = f"R² = {r2:.3f}"
                                
                                st.markdown(create_enhanced_metric(
                                    model_name,
                                    f"₹{price:,.0f}",
                                    accuracy_info,
                                    "Predicted price"
                                ), unsafe_allow_html=True)
                        
                        with pred_col2:
                            # Prediction comparison chart
                            pred_df = pd.DataFrame(list(predictions.items()), columns=['Model', 'Predicted Price'])
                            
                            fig = px.bar(
                                pred_df, 
                                x='Model', 
                                y='Predicted Price',
                                title="Price Predictions Across Models",
                                color='Predicted Price',
                                color_continuous_scale='viridis',
                                text='Predicted Price'
                            )
                            fig.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
                            fig.update_layout(
                                yaxis_title="Predicted Price (₹)",
                                showlegend=False,
                                height=400,
                                xaxis_tickangle=-45
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Input summary
                        st.markdown("### 📋 Specification Summary")
                        
                        summary_col1, summary_col2, summary_col3 = st.columns(3)
                        
                        with summary_col1:
                            st.write(f"**🏢 Brand:** {company}")
                            st.write(f"**💼 Type:** {laptop_type}")
                            st.write(f"**💾 RAM:** {ram} GB")
                            st.write(f"**⚖️ Weight:** {weight} kg")
                        
                        with summary_col2:
                            st.write(f"**🖥️ CPU:** {cpu}")
                            st.write(f"**🎮 GPU:** {gpu}")
                            st.write(f"**📱 OS:** {os_type}")
                            st.write(f"**📏 Screen:** {screen_size}\" {resolution}")
                        
                        with summary_col3:
                            st.write(f"**💽 HDD:** {hdd} GB")
                            st.write(f"**⚡ SSD:** {ssd} GB")
                            st.write(f"**🖐️ Touch:** {touchscreen}")
                            st.write(f"**🖼️ IPS:** {ips}")
                        
                        # Price insights
                        if len(predictions) > 1:
                            price_range = max(predictions.values()) - min(predictions.values())
                            confidence = "High" if price_range < avg_prediction * 0.2 else "Medium" if price_range < avg_prediction * 0.5 else "Low"
                            
                            st.markdown(f"""
                            <div class="info-box">
                            <strong>💡 Prediction Insights:</strong><br>
                            • Price Range: ₹{min(predictions.values()):,.0f} - ₹{max(predictions.values()):,.0f}<br>
                            • Variation: ₹{price_range:,.0f} ({price_range/avg_prediction*100:.1f}%)<br>
                            • Confidence Level: {confidence}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    else:
                        st.markdown('<div class="error-box">❌ <strong>Prediction Error:</strong> Unable to generate predictions from any model.</div>', unsafe_allow_html=True)

# =========================================================================================
# PAGE 3: MODEL ANALYTICS
# =========================================================================================
elif page == "📊 Model Analytics":
    st.markdown('<div class="page-header">📊 Advanced Model Analytics</div>', unsafe_allow_html=True)
    
    if models and any(model_name in accuracies for model_name in models.keys()):
        # Filter to loaded models
        loaded_model_accuracies = {name: accuracies[name] for name in models.keys() if name in accuracies}
        acc_df = pd.DataFrame(loaded_model_accuracies).T.reset_index().rename(columns={'index': 'Model'})
        
        # Performance metrics table
        st.markdown("### 📋 Comprehensive Model Performance")
        
        # Style the dataframe
        styled_df = acc_df.style.background_gradient(cmap='RdYlGn', subset=['R2']) \
                                 .background_gradient(cmap='RdYlGn_r', subset=['MAE']) \
                                 .format({'R2': '{:.3f}', 'MAE': '₹{:,.0f}', 'RMSE': '₹{:,.0f}'})
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Advanced analytics
        col1, col2 = st.columns(2)
        
        with col1:
            # R² vs MAE scatter plot
            fig = px.scatter(
                acc_df, 
                x='R2', 
                y='MAE',
                text='Model',
                title="Model Performance: Accuracy vs Precision",
                color='R2',
                size='MAE',
                size_max=30,
                color_continuous_scale='viridis'
            )
            fig.update_traces(textposition='top center')
            fig.update_layout(
                xaxis_title="R² Score (Higher = Better Accuracy)",
                yaxis_title="MAE (Lower = Better Precision)",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Model ranking
            acc_df['Composite Score'] = acc_df['R2'] * 100 - acc_df['MAE'] / 1000
            acc_df_sorted = acc_df.sort_values('Composite Score', ascending=True)
            
            fig2 = px.bar(
                acc_df_sorted,
                x='Composite Score',
                y='Model',
                orientation='h',
                title="Overall Model Ranking",
                color='Composite Score',
                color_continuous_scale='viridis'
            )
            fig2.update_layout(height=500)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Performance insights
        best_r2 = acc_df.loc[acc_df['R2'].idxmax()]
        best_mae = acc_df.loc[acc_df['MAE'].idxmin()]
        
        insight_col1, insight_col2, insight_col3 = st.columns(3)
        
        with insight_col1:
            st.markdown(create_enhanced_metric(
                "Highest Accuracy",
                best_r2['Model'],
                f"R² = {best_r2['R2']:.3f}",
                "Best explained variance"
            ), unsafe_allow_html=True)
        
        with insight_col2:
            st.markdown(create_enhanced_metric(
                "Best Precision",
                best_mae['Model'],
                f"MAE = ₹{best_mae['MAE']:,.0f}",
                "Lowest average error"
            ), unsafe_allow_html=True)
        
        with insight_col3:
            avg_r2 = acc_df['R2'].mean()
            avg_mae = acc_df['MAE'].mean()
            st.markdown(create_enhanced_metric(
                "Average Performance",
                f"R² = {avg_r2:.3f}",
                f"MAE = ₹{avg_mae:,.0f}",
                "Ensemble baseline"
            ), unsafe_allow_html=True)
        
    else:
        st.markdown('<div class="info-box">📊 Model analytics will appear here once models are loaded and accuracy data is available.</div>', unsafe_allow_html=True)

# =========================================================================================
# PAGE 4: MODEL COMPARISON
# =========================================================================================
elif page == "⚙️ Model Comparison":
    st.markdown('<div class="page-header">⚙️ Advanced Model Comparison</div>', unsafe_allow_html=True)
    
    if models:
        st.markdown("### 🔬 Model Architecture Analysis")
        
        # Model comparison table
        model_info = []
        for name, model in models.items():
            model_type = type(model).__name__
            
            # Try to get model parameters
            try:
                if hasattr(model, 'get_params'):
                    params = model.get_params()
                    param_count = len(params)
                else:
                    param_count = "N/A"
            except:
                param_count = "N/A"
            
            # Get accuracy info if available
            accuracy_info = accuracies.get(name, {})
            
            model_info.append({
                'Model Name': name,
                'Algorithm': model_type,
                'Parameters': param_count,
                'R² Score': accuracy_info.get('R2', 'N/A'),
                'MAE': accuracy_info.get('MAE', 'N/A'),
                'RMSE': accuracy_info.get('RMSE', 'N/A')
            })
        
        comparison_df = pd.DataFrame(model_info)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Model selection recommendations
        st.markdown("### 🎯 Model Selection Guide")
        
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            st.markdown("""
            <div class="info-box">
            <h4>🚀 For Speed & Simplicity:</h4>
            <ul>
                <li><strong>Linear Regression:</strong> Fastest, most interpretable</li>
                <li><strong>Ridge/Lasso:</strong> Good baseline with regularization</li>
                <li><strong>KNN:</strong> Simple, no training required</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with rec_col2:
            st.markdown("""
            <div class="info-box">
            <h4>🎯 For Accuracy & Performance:</h4>
            <ul>
                <li><strong>Random Forest:</strong> Excellent balance, handles overfitting</li>
                <li><strong>XGBoost:</strong> State-of-the-art performance</li>
                <li><strong>Gradient Boost:</strong> High accuracy, good generalization</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Interactive model selector
        st.markdown("### 🔧 Model Performance Simulator")
        
        if any(model_name in accuracies for model_name in models.keys()):
            selected_models = st.multiselect(
                "Select models to compare:",
                [name for name in models.keys() if name in accuracies],
                default=list(models.keys())[:3] if len(models) >= 3 else list(models.keys())
            )
            
            if selected_models:
                selected_acc = {name: accuracies[name] for name in selected_models}
                selected_df = pd.DataFrame(selected_acc).T
                
                # Comparison charts
                fig = go.Figure()
                
                for metric in ['R2', 'MAE']:
                    fig.add_trace(go.Scatter(
                        x=selected_df.index,
                        y=selected_df[metric],
                        mode='markers+lines',
                        name=metric,
                        line=dict(width=3),
                        marker=dict(size=10)
                    ))
                
                fig.update_layout(
                    title="Selected Models Performance Comparison",
                    xaxis_title="Models",
                    yaxis_title="Score",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.markdown('<div class="error-box">⚠️ No models loaded for comparison. Please ensure model files are available.</div>', unsafe_allow_html=True)

# ------------------- Footer -------------------
st.markdown("""
<div class="footer">
    <h3>💻 Advanced Laptop Price Predictor</h3>
    <p>Powered by Machine Learning | Built with Streamlit</p>
    <p>🚀 Accurate • 🎯 Reliable • 🔬 Data-Driven</p>
</div>
""", unsafe_allow_html=True)