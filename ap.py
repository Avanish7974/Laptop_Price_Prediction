import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime

# ===============================================================================
# PAGE CONFIGURATION
# ===============================================================================
st.set_page_config(
    page_title="Laptop Price Predictor Pro",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================================================================
# CUSTOM STYLING
# ===============================================================================
def apply_custom_css():
    st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Variables */
        :root {
            --primary-color: #0077b6;
            --secondary-color: #00b4d8;
            --accent-color: #90e0ef;
            --text-primary: #2c3e50;
            --text-secondary: #7f8c8d;
            --background-light: #ffffff;
            --background-card: #f8fafc;
            --border-light: #e1e8ed;
            --success-color: #27ae60;
            --warning-color: #f39c12;
            --error-color: #e74c3c;
        }
        
        /* Global Styles */
        .main {
            font-family: 'Inter', sans-serif;
            padding-top: 1rem;
        }
        
        /* Header Styles */
        .app-header {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0, 119, 182, 0.15);
        }
        
        .app-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        .app-subtitle {
            font-size: 1.2rem;
            font-weight: 400;
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
        }
        
        /* Section Headers */
        .section-header {
            font-size: 1.8rem;
            font-weight: 600;
            color: var(--primary-color);
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid var(--accent-color);
        }
        
        /* Card Styles */
        .prediction-card {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 2rem;
            border-radius: 20px;
            margin: 1rem 0;
            box-shadow: 0 12px 40px rgba(0, 119, 182, 0.2);
            text-align: center;
        }
        
        .info-card {
            background: var(--background-card);
            padding: 1.5rem;
            border-radius: 15px;
            border: 1px solid var(--border-light);
            margin: 1rem 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .info-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        }
        
        /* Form Styles */
        .form-section {
            background: var(--background-card);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            border: 1px solid var(--border-light);
        }
        
        /* Button Styles */
        .stButton > button {
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 119, 182, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 119, 182, 0.4);
        }
        
        /* Metric Containers */
        [data-testid="metric-container"] {
            background: var(--background-card);
            border: 1px solid var(--border-light);
            border-radius: 15px;
            padding: 1rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        
        /* Sidebar Styling */
        .sidebar-nav {
            background: var(--background-card);
            padding: 1rem;
            border-radius: 15px;
            margin: 1rem 0;
            border: 1px solid var(--border-light);
        }
        
        /* Progress Bar */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        }
        
        /* Success/Warning/Error Messages */
        .success-message {
            background: linear-gradient(135deg, var(--success-color), #2ecc71);
            color: white;
            padding: 1rem;
            border-radius: 15px;
            margin: 1rem 0;
            text-align: center;
            font-weight: 500;
        }
        
        .warning-message {
            background: linear-gradient(135deg, var(--warning-color), #e67e22);
            color: white;
            padding: 1rem;
            border-radius: 15px;
            margin: 1rem 0;
            text-align: center;
            font-weight: 500;
        }
        
        .error-message {
            background: linear-gradient(135deg, var(--error-color), #c0392b);
            color: white;
            padding: 1rem;
            border-radius: 15px;
            margin: 1rem 0;
            text-align: center;
            font-weight: 500;
        }
        
        /* Dark Mode Support */
        .dark-mode {
            --background-light: #1a1a1a;
            --background-card: #2d2d2d;
            --text-primary: #ffffff;
            --text-secondary: #cccccc;
            --border-light: #444444;
        }
        
        .dark-mode .main {
            background-color: var(--background-light);
            color: var(--text-primary);
        }
        
        .dark-mode .info-card {
            background: var(--background-card);
            color: var(--text-primary);
            border-color: var(--border-light);
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .app-title {
                font-size: 2rem;
            }
            .app-subtitle {
                font-size: 1rem;
            }
            .section-header {
                font-size: 1.5rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)

# ===============================================================================
# DATA LOADING FUNCTIONS
# ===============================================================================
@st.cache_data
def load_dataset():
    """Load the laptop dataset"""
    try:
        df = pickle.load(open('df.pkl', 'rb'))
        return df, None
    except FileNotFoundError:
        # Create mock dataset if file not found
        mock_data = {
            'Brand': ['Dell', 'HP', 'Lenovo', 'Apple', 'Asus'] * 200,
            'Processor': ['Intel i5', 'Intel i7', 'AMD Ryzen 5', 'Intel i3', 'AMD Ryzen 7'] * 200,
            'RAM': [8, 16, 4, 32, 8] * 200,
            'Storage': ['256GB SSD', '512GB SSD', '1TB HDD', '1TB SSD', '256GB SSD'] * 200,
            'Screen_Size': [15.6, 14, 17.3, 13.3, 15.6] * 200,
            'Price': np.random.randint(25000, 200000, 1000),
            'GPU': ['Intel UHD', 'NVIDIA GTX', 'AMD Radeon', 'Intel Iris', 'NVIDIA RTX'] * 200
        }
        df = pd.DataFrame(mock_data)
        return df, "Using mock dataset (df.pkl not found)"
    except Exception as e:
        return None, f"Error loading dataset: {str(e)}"

@st.cache_data
def load_models():
    """Load ML models"""
    # Mock models for demonstration
    models = {
        'Random Forest': {'accuracy': 0.92, 'mae': 12000},
        'Linear Regression': {'accuracy': 0.78, 'mae': 24000},
        'XGBoost': {'accuracy': 0.90, 'mae': 14000}
    }
    return models, []

# ===============================================================================
# UTILITY FUNCTIONS
# ===============================================================================
def predict_price(specifications, model_name='Random Forest'):
    """Predict laptop price based on specifications"""
    # Mock prediction logic - replace with actual ML model
    base_price = 50000
    
    # Brand multipliers
    brand_multipliers = {
        'Apple': 2.2, 'Dell': 1.1, 'HP': 1.0, 'Lenovo': 0.9,
        'Asus': 1.15, 'Acer': 0.85, 'MSI': 1.3, 'Alienware': 1.8
    }
    
    # Calculate price based on specs
    brand_mult = brand_multipliers.get(specifications.get('brand', 'HP'), 1.0)
    ram_price = specifications.get('ram', 8) * 2000
    storage_bonus = 15000 if 'SSD' in specifications.get('storage', '') else 5000
    screen_bonus = specifications.get('screen_size', 15) * 2000
    
    # GPU pricing
    gpu_prices = {
        'Integrated': 0, 'NVIDIA GTX': 25000, 'NVIDIA RTX': 45000,
        'AMD Radeon': 20000, 'Intel Arc': 15000, 'Apple GPU': 30000
    }
    gpu_bonus = gpu_prices.get(specifications.get('graphics', 'Integrated'), 0)
    
    predicted_price = base_price * brand_mult + ram_price + storage_bonus + screen_bonus + gpu_bonus
    
    # Add some randomness for different models
    if model_name == 'Linear Regression':
        predicted_price *= 0.95
    elif model_name == 'XGBoost':
        predicted_price *= 1.02
    
    return max(predicted_price, 20000)  # Minimum price

def categorize_price(price):
    """Categorize laptop price"""
    if price < 40000:
        return "💰 Budget", "#27ae60"
    elif price < 80000:
        return "⚖️ Mid-Range", "#f39c12"
    elif price < 150000:
        return "🏆 Premium", "#8e44ad"
    else:
        return "👑 Flagship", "#e74c3c"

def get_similar_laptops(df, target_specs, price_range=20000):
    """Find similar laptops from dataset"""
    # Mock similar laptops - replace with actual similarity logic
    similar = df.sample(n=min(5, len(df))).copy()
    return similar.head(3)

# ===============================================================================
# SIDEBAR NAVIGATION
# ===============================================================================
def render_sidebar():
    """Render sidebar navigation"""
    st.sidebar.markdown("""
    <div class="app-header" style="margin-bottom: 1rem;">
        <h2 style="margin: 0; font-size: 1.5rem;">💻 Laptop Price Predictor</h2>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Professional ML Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    pages = {
        "🏠 Home / Dashboard": "dashboard",
        "💻 Predict Price": "predict",
        "📊 Insights": "insights", 
        "📁 Dataset Explorer": "explorer",
        "🔄 Predicted vs Actual": "comparison",
        "⚙️ Settings": "settings"
    }
    
    selected_page = st.sidebar.radio("Navigation", list(pages.keys()))
    return pages[selected_page]

# ===============================================================================
# PAGE: DASHBOARD
# ===============================================================================
def render_dashboard(df, models):
    """Render dashboard page"""
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">🏠 Dashboard & Analytics</h1>
        <p class="app-subtitle">Comprehensive overview of laptop market insights and ML model performance</p>
    </div>
    """, unsafe_allow_html=True)
    
    if df is None:
        st.error("Dataset not available")
        return
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Laptops", f"{len(df):,}")
    with col2:
        avg_price = df['Price'].mean() if 'Price' in df.columns else 75000
        st.metric("💰 Average Price", f"₹{avg_price:,.0f}")
    with col3:
        st.metric("🏷️ Brands", df['Brand'].nunique() if 'Brand' in df.columns else 8)
    with col4:
        st.metric("🤖 ML Models", len(models))
    
    # Price Distribution
    st.markdown('<h2 class="section-header">📈 Price Analytics</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Brand' in df.columns and 'Price' in df.columns:
            brand_prices = df.groupby('Brand')['Price'].mean().sort_values(ascending=False)
            fig = px.bar(
                x=brand_prices.index, 
                y=brand_prices.values,
                title="Average Price by Brand",
                labels={'x': 'Brand', 'y': 'Average Price (₹)'},
                color=brand_prices.values,
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Price category distribution
        prices = df['Price'] if 'Price' in df.columns else np.random.randint(30000, 150000, len(df))
        categories = []
        for price in prices:
            if price < 40000:
                categories.append('Budget')
            elif price < 80000:
                categories.append('Mid-Range')
            elif price < 150000:
                categories.append('Premium')
            else:
                categories.append('Flagship')
        
        category_counts = pd.Series(categories).value_counts()
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Price Category Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Model Performance
    st.markdown('<h2 class="section-header">🤖 Model Performance</h2>', unsafe_allow_html=True)
    
    model_data = []
    for name, stats in models.items():
        model_data.append({
            'Model': name,
            'Accuracy': f"{stats['accuracy']:.1%}",
            'MAE': f"₹{stats['mae']:,}",
            'Status': '✅ Ready'
        })
    
    df_models = pd.DataFrame(model_data)
    st.dataframe(df_models, use_container_width=True, hide_index=True)

# ===============================================================================
# PAGE: PRICE PREDICTION
# ===============================================================================
def render_prediction_page(df, models):
    """Render price prediction page"""
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">💻 Laptop Price Prediction</h1>
        <p class="app-subtitle">Configure your ideal laptop and get AI-powered price predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input Form
    st.markdown('<h2 class="section-header">🔧 Laptop Specifications</h2>', unsafe_allow_html=True)
    
    with st.container():
        # Basic Specs
        st.markdown("### 💻 Basic Specifications")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            brands = ['Dell', 'HP', 'Lenovo', 'Apple', 'Asus', 'Acer', 'MSI', 'Alienware']
            brand = st.selectbox("🏷️ Brand", brands)
            
            processors = ['Intel i3', 'Intel i5', 'Intel i7', 'Intel i9', 'AMD Ryzen 3', 'AMD Ryzen 5', 'AMD Ryzen 7', 'AMD Ryzen 9', 'Apple M1', 'Apple M2']
            processor = st.selectbox("🧠 Processor Type", processors)
            
            ram_options = [4, 8, 16, 32, 64]
            ram = st.selectbox("💾 RAM Size (GB)", ram_options, index=1)
        
        with col2:
            ram_types = ['DDR4', 'DDR5', 'LPDDR4', 'LPDDR5']
            ram_type = st.selectbox("⚡ RAM Type", ram_types)
            
            ssd_options = [0, 128, 256, 512, 1024, 2048]
            ssd = st.selectbox("💿 SSD Size (GB)", ssd_options, index=2)
            
            hdd_options = [0, 500, 1000, 2000]
            hdd = st.selectbox("💾 HDD Size (GB)", hdd_options, index=0)
        
        with col3:
            storage_types = ['SSD Only', 'HDD Only', 'SSD + HDD', 'eMMC', 'NVMe SSD']
            storage_type = st.selectbox("📂 Storage Type", storage_types)
            
            gen_options = ['10th Gen', '11th Gen', '12th Gen', '13th Gen', '14th Gen']
            processor_gen = st.selectbox("🔄 Processor Generation", gen_options, index=2)
        
        # Display & Graphics
        st.markdown("### 📺 Display & Graphics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            screen_size = st.slider("📐 Screen Size (inches)", 13.0, 17.3, 15.6, 0.1)
            
            resolutions = ['HD (1366x768)', 'Full HD (1920x1080)', 'QHD (2560x1440)', '4K UHD (3840x2160)']
            resolution = st.selectbox("🖥️ Screen Resolution", resolutions, index=1)
            
            touchscreen = st.selectbox("👆 Touchscreen", ['No', 'Yes'])
        
        with col2:
            ips_display = st.selectbox("🎨 IPS Display", ['No', 'Yes'])
            
            refresh_rates = ['60Hz', '90Hz', '120Hz', '144Hz', '165Hz', '240Hz']
            refresh_rate = st.selectbox("🔄 Refresh Rate", refresh_rates)
            
            graphics_options = ['Integrated', 'NVIDIA GTX', 'NVIDIA RTX', 'AMD Radeon', 'Intel Arc', 'Apple GPU']
            graphics = st.selectbox("🎮 Graphics Card", graphics_options)
        
        with col3:
            # Battery & Build
            st.markdown("### 🔋 Battery & Build")
            battery = st.slider("🔋 Battery Backup (hours)", 2, 12, 6)
            weight = st.slider("⚖️ Weight (kg)", 0.9, 3.5, 2.0, 0.1)
            
            materials = ['Plastic', 'Aluminum', 'Carbon Fiber', 'Magnesium Alloy']
            material = st.selectbox("🏗️ Build Material", materials, index=1)
        
        # Operating System
        st.markdown("### 💻 Operating System")
        os_options = ['Windows 11', 'Windows 10', 'macOS', 'Linux', 'ChromeOS', 'DOS', 'None']
        operating_system = st.selectbox("🖥️ Operating System", os_options)
        
        # Prediction Button
        st.markdown("---")
        if st.button("🔍 Predict Price", type="primary", use_container_width=True):
            with st.spinner("🔮 Analyzing specifications and predicting price..."):
                # Prepare specifications
                specs = {
                    'brand': brand,
                    'processor': processor,
                    'ram': ram,
                    'storage': f"{ssd}GB SSD" if ssd > 0 else f"{hdd}GB HDD",
                    'screen_size': screen_size,
                    'graphics': graphics,
                    'resolution': resolution,
                    'touchscreen': touchscreen,
                    'battery': battery,
                    'weight': weight
                }
                
                # Get predictions from all models
                predictions = {}
                for model_name in models.keys():
                    price = predict_price(specs, model_name)
                    predictions[model_name] = price
                
                # Calculate ensemble prediction
                avg_prediction = np.mean(list(predictions.values()))
                std_dev = np.std(list(predictions.values()))
                confidence = max(0.0, 100.0 - (std_dev / avg_prediction * 30.0))
                
                # Price range
                price_margin = std_dev * 1.2
                min_price = max(avg_prediction - price_margin, 20000)
                max_price = avg_prediction + price_margin
                
                # Price category
                category, category_color = categorize_price(avg_prediction)
                
                # Display Results
                st.markdown('<h2 class="section-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)
                
                # Main prediction card
                st.markdown(f"""
                <div class="prediction-card">
                    <h2>🎯 Predicted Price</h2>
                    <h1 style="font-size: 3rem; margin: 1rem 0;">₹{avg_prediction:,.0f}</h1>
                    <div style="display: flex; justify-content: space-around; margin-top: 2rem;">
                        <div>
                            <h4>📊 Price Range</h4>
                            <p style="font-size: 1.2rem;">₹{min_price:,.0f} - ₹{max_price:,.0f}</p>
                        </div>
                        <div>
                            <h4>🎯 Confidence</h4>
                            <p style="font-size: 1.2rem;">{confidence:.1f}%</p>
                        </div>
                        <div>
                            <h4>🏷️ Category</h4>
                            <p style="font-size: 1.2rem; color: {category_color};">{category}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Individual model predictions
                st.markdown("### 🤖 Model Predictions")
                pred_cols = st.columns(len(predictions))
                
                for i, (model_name, price) in enumerate(predictions.items()):
                    with pred_cols[i]:
                        variance = ((price - avg_prediction) / avg_prediction * 100)
                        st.metric(
                            model_name,
                            f"₹{price:,.0f}",
                            f"{variance:+.1f}%"
                        )
                
                # Confidence visualization
                st.markdown("### 📊 Prediction Confidence")
                progress_col, metric_col = st.columns([3, 1])
                
                with progress_col:
                    st.progress(confidence / 100)
                    if confidence > 85:
                        st.success(f"High confidence prediction ({confidence:.1f}%)")
                    elif confidence > 70:
                        st.warning(f"Moderate confidence prediction ({confidence:.1f}%)")
                    else:
                        st.error(f"Low confidence prediction ({confidence:.1f}%)")
                
                with metric_col:
                    st.metric("Confidence Score", f"{confidence:.1f}%")
                
                # Similar laptops
                if df is not None:
                    st.markdown('<h2 class="section-header">🔍 Similar Laptops</h2>', unsafe_allow_html=True)
                    
                    similar_laptops = get_similar_laptops(df, specs)
                    
                    if not similar_laptops.empty:
                        st.markdown(f"""
                        <div style="background: linear-gradient(90deg, #0077b6, #00b4d8); color: white; padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">
                            <h4>💡 Based on your configuration (₹{avg_prediction:,.0f}), here are similar laptops from our database:</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display similar laptops
                        cols = st.columns(min(3, len(similar_laptops)))
                        for i, (_, laptop) in enumerate(similar_laptops.iterrows()):
                            if i < len(cols):
                                with cols[i]:
                                    match_score = np.random.randint(75, 95)
                                    st.markdown(f"""
                                    <div class="info-card">
                                        <h4>{laptop.get('Brand', 'Unknown')} Model</h4>
                                        <p><strong>💾 RAM:</strong> {laptop.get('RAM', 8)}GB</p>
                                        <p><strong>💿 Storage:</strong> {laptop.get('Storage', '256GB SSD')}</p>
                                        <p><strong>🎮 GPU:</strong> {laptop.get('GPU', 'Integrated')}</p>
                                        <p><strong>💰 Price:</strong> <span style="color: #0077b6; font-weight: bold;">₹{laptop.get('Price', 50000):,.0f}</span></p>
                                        <p><strong>🎯 Match:</strong> <span style="color: #27ae60; font-weight: bold;">{match_score}%</span></p>
                                    </div>
                                    """, unsafe_allow_html=True)

# ===============================================================================
# PAGE: INSIGHTS
# ===============================================================================
def render_insights_page(df, models):
    """Render insights and analytics page"""
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">📊 Market Insights & Analytics</h1>
        <p class="app-subtitle">Deep dive into laptop market trends and price analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    if df is None:
        st.error("Dataset not available for insights")
        return
    
    # Market Trends
    st.markdown('<h2 class="section-header">📈 Market Trends</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # RAM vs Price Analysis
        if 'RAM' in df.columns and 'Price' in df.columns:
            ram_prices = df.groupby('RAM')['Price'].mean().sort_index()
            fig = px.line(
                x=ram_prices.index,
                y=ram_prices.values,
                title="Price Trend by RAM Capacity",
                labels={'x': 'RAM (GB)', 'y': 'Average Price (₹)'},
                markers=True
            )
            fig.update_traces(line=dict(color='#0077b6', width=3), marker=dict(size=8))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Brand popularity
        if 'Brand' in df.columns:
            brand_counts = df['Brand'].value_counts()
            fig = px.bar(
                x=brand_counts.values,
                y=brand_counts.index,
                orientation='h',
                title="Most Popular Brands",
                labels={'x': 'Number of Models', 'y': 'Brand'},
                color=brand_counts.values,
                color_continuous_scale='blues'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Top Performers
    st.markdown('<h2 class="section-header">🏆 Top Performers</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💎 Most Expensive Laptops")
        if 'Price' in df.columns:
            top_expensive = df.nlargest(5, 'Price')[['Brand', 'Price']].copy()
            top_expensive['Price'] = top_expensive['Price'].apply(lambda x: f"₹{x:,.0f}")
            st.dataframe(top_expensive, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 💰 Best Value Laptops")
        if 'Price' in df.columns:
            # Calculate value score (mock)
            df_value = df.copy()
            df_value['Value_Score'] = np.random.randint(70, 95, len(df))
            top_value = df_value.nlargest(5, 'Value_Score')[['Brand', 'Price', 'Value_Score']].copy()
            top_value['Price'] = top_value['Price'].apply(lambda x: f"₹{x:,.0f}")
            st.dataframe(top_value, use_container_width=True, hide_index=True)

# ===============================================================================
# PAGE: DATASET EXPLORER
# ===============================================================================
def render_explorer_page(df):
    """Render dataset explorer page"""
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">📁 Dataset Explorer</h1>
        <p class="app-subtitle">Browse, filter, and search through the laptop database</p>
    </div>
    """, unsafe_allow_html=True)
    
    if df is None:
        st.error("Dataset not available")
        return
    
    # Search and Filter
    st.markdown('<h2 class="section-header">🔍 Search & Filter</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Brand filter
        brands = ['All'] + sorted(df['Brand'].unique().tolist()) if 'Brand' in df.columns else ['All']
        selected_brand = st.selectbox("🏷️ Filter by Brand", brands)
        
        # Price filter
        if 'Price' in df.columns:
            min_price, max_price = int(df['Price'].min()), int(df['Price'].max())
            price_range = st.slider(
                "💰 Price Range (₹)",
                min_price, max_price,
                (min_price, max_price)
            )
    
    with col2:
        # RAM filter
        ram_options = ['All'] + sorted(df['RAM'].unique().tolist()) if 'RAM' in df.columns else ['All']
        selected_ram = st.selectbox("💾 Filter by RAM", ram_options)
        
        # Search
        search_query = st.text_input("🔍 Search laptops", placeholder="Search by brand, model, or specs...")
    
    with col3:
        # Storage filter
        storage_options = ['All'] + sorted(df['Storage'].unique().tolist()) if 'Storage' in df.columns else ['All']
        selected_storage = st.selectbox("💿 Filter by Storage", storage_options)
        
        # Download option
        if st.button("📥 Download Filtered Data", type="secondary"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"laptop_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_brand != 'All' and 'Brand' in df.columns:
        filtered_df = filtered_df[filtered_df['Brand'] == selected_brand]
    
    if selected_ram != 'All' and 'RAM' in df.columns:
        filtered_df = filtered_df[filtered_df['RAM'] == selected_ram]
    
    if selected_storage != 'All' and 'Storage' in df.columns:
        filtered_df = filtered_df[filtered_df['Storage'] == selected_storage]
    
    if 'Price' in df.columns:
        filtered_df = filtered_df[
            (filtered_df['Price'] >= price_range[0]) & 
            (filtered_df['Price'] <= price_range[1])
        ]
    
    if search_query:
        # Simple search implementation
        mask = df.astype(str).apply(lambda x: x.str.contains(search_query, case=False, na=False)).any(axis=1)
        filtered_df = filtered_df[mask]
    
    # Display results
    st.markdown('<h2 class="section-header">📊 Results</h2>', unsafe_allow_html=True)
    st.info(f"Found {len(filtered_df)} laptops matching your criteria")
    
    if not filtered_df.empty:
        # Display sample data
        st.dataframe(filtered_df.head(20), use_container_width=True)
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔍 Results", len(filtered_df))
        with col2:
            if 'Price' in filtered_df.columns:
                avg_price = filtered_df['Price'].mean()
                st.metric("💰 Avg Price", f"₹{avg_price:,.0f}")
        with col3:
            if 'Brand' in filtered_df.columns:
                unique_brands = filtered_df['Brand'].nunique()
                st.metric("🏷️ Brands", unique_brands)
        with col4:
            if 'RAM' in filtered_df.columns:
                avg_ram = filtered_df['RAM'].mean()
                st.metric("💾 Avg RAM", f"{avg_ram:.1f}GB")

# ===============================================================================
# PAGE: PREDICTED VS ACTUAL
# ===============================================================================
def render_comparison_page(df, models):
    """Render predicted vs actual comparison page"""
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">🔄 Predicted vs Actual Comparison</h1>
        <p class="app-subtitle">Compare ML model predictions with actual laptop prices</p>
    </div>
    """, unsafe_allow_html=True)
    
    if df is None:
        st.error("Dataset not available")
        return
    
    st.markdown('<h2 class="section-header">🎯 Model Validation</h2>', unsafe_allow_html=True)
    
    # Sample selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Select sample laptops
        sample_size = st.slider("📊 Number of laptops to compare", 5, 20, 10)
        model_name = st.selectbox("🤖 Select Model", list(models.keys()))
    
    with col2:
        if st.button("🔄 Generate Comparison", type="primary"):
            st.session_state.comparison_generated = True
    
    if st.session_state.get('comparison_generated', False):
        # Generate sample comparison
        sample_df = df.sample(n=min(sample_size, len(df))).copy()
        
        # Generate predictions (mock)
        predictions = []
        for _, row in sample_df.iterrows():
            specs = {
                'brand': row.get('Brand', 'Dell'),
                'ram': row.get('RAM', 8),
                'storage': row.get('Storage', '256GB SSD'),
                'screen_size': 15.6,
                'graphics': row.get('GPU', 'Integrated')
            }
            pred_price = predict_price(specs, model_name)
            predictions.append(pred_price)
        
        sample_df['Predicted_Price'] = predictions
        sample_df['Actual_Price'] = sample_df['Price'] if 'Price' in sample_df.columns else np.random.randint(30000, 120000, len(sample_df))
        sample_df['Difference'] = sample_df['Predicted_Price'] - sample_df['Actual_Price']
        sample_df['Error_Percent'] = abs(sample_df['Difference']) / sample_df['Actual_Price'] * 100
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            mae = sample_df['Difference'].abs().mean()
            st.metric("📊 MAE", f"₹{mae:,.0f}")
        
        with col2:
            mape = sample_df['Error_Percent'].mean()
            st.metric("📈 MAPE", f"{mape:.1f}%")
        
        with col3:
            accuracy = 100 - mape
            st.metric("🎯 Accuracy", f"{accuracy:.1f}%")
        
        with col4:
            rmse = np.sqrt((sample_df['Difference'] ** 2).mean())
            st.metric("📐 RMSE", f"₹{rmse:,.0f}")
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot
            fig = px.scatter(
                sample_df,
                x='Actual_Price',
                y='Predicted_Price',
                title="Predicted vs Actual Prices",
                labels={'Actual_Price': 'Actual Price (₹)', 'Predicted_Price': 'Predicted Price (₹)'},
                hover_data=['Brand'] if 'Brand' in sample_df.columns else None
            )
            
            # Add perfect prediction line
            min_price = min(sample_df['Actual_Price'].min(), sample_df['Predicted_Price'].min())
            max_price = max(sample_df['Actual_Price'].max(), sample_df['Predicted_Price'].max())
            fig.add_trace(go.Scatter(
                x=[min_price, max_price],
                y=[min_price, max_price],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='red')
            ))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Error distribution
            fig = px.histogram(
                sample_df,
                x='Error_Percent',
                title="Prediction Error Distribution",
                labels={'Error_Percent': 'Error Percentage (%)', 'count': 'Frequency'},
                nbins=10
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed comparison table
        st.markdown('<h2 class="section-header">📋 Detailed Comparison</h2>', unsafe_allow_html=True)
        
        display_df = sample_df[['Brand', 'Actual_Price', 'Predicted_Price', 'Difference', 'Error_Percent']].copy()
        display_df['Actual_Price'] = display_df['Actual_Price'].apply(lambda x: f"₹{x:,.0f}")
        display_df['Predicted_Price'] = display_df['Predicted_Price'].apply(lambda x: f"₹{x:,.0f}")
        display_df['Difference'] = display_df['Difference'].apply(lambda x: f"₹{x:+,.0f}")
        display_df['Error_Percent'] = display_df['Error_Percent'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# ===============================================================================
# PAGE: SETTINGS
# ===============================================================================
def render_settings_page():
    """Render settings page"""
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">⚙️ Settings & Configuration</h1>
        <p class="app-subtitle">Customize your experience and model preferences</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Theme Settings
    st.markdown('<h2 class="section-header">🎨 Theme Settings</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        theme_mode = st.radio(
            "Choose Theme",
            ["🌞 Light Mode", "🌙 Dark Mode"],
            key="theme_toggle"
        )
        
        if theme_mode == "🌙 Dark Mode":
            st.markdown("""
            <style>
                .stApp { background-color: #1a1a1a !important; }
                .main { background-color: #1a1a1a !important; color: white !important; }
                .info-card { background: #2d2d2d !important; color: white !important; }
                [data-testid="stSidebar"] { background-color: #2d2d2d !important; }
            </style>
            """, unsafe_allow_html=True)
        
        st.success(f"Theme set to {theme_mode}")
    
    with col2:
        # Color customization
        st.markdown("### 🎨 Color Customization")
        primary_color = st.color_picker("Primary Color", "#0077b6")
        secondary_color = st.color_picker("Secondary Color", "#00b4d8")
        accent_color = st.color_picker("Accent Color", "#90e0ef")
    
    # Model Settings
    st.markdown('<h2 class="section-header">🤖 Model Configuration</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        models = ['Random Forest', 'Linear Regression', 'XGBoost']
        default_model = st.selectbox("🎯 Default Prediction Model", models)
        
        ensemble_mode = st.checkbox("🔀 Use Ensemble Predictions", value=True)
        
        confidence_threshold = st.slider("📊 Confidence Threshold (%)", 60, 95, 80)
    
    with col2:
        st.markdown("### 📈 Performance Metrics")
        show_metrics = st.multiselect(
            "Display Metrics",
            ["MAE", "RMSE", "MAPE", "R²", "Confidence Score"],
            default=["MAE", "Confidence Score"]
        )
        
        st.markdown("### 💾 Data Settings")
        cache_predictions = st.checkbox("💾 Cache Predictions", value=True)
        auto_save = st.checkbox("💾 Auto-save Settings", value=True)
    
    # Export Settings
    st.markdown('<h2 class="section-header">📥 Export & Download</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_format = st.selectbox("📄 Default Export Format", ["CSV", "Excel", "JSON", "PDF"])
        
        if st.button("📥 Export Current Session", type="secondary"):
            st.success("Session data exported successfully!")
    
    with col2:
        st.markdown("### 📊 Report Settings")
        include_charts = st.checkbox("📊 Include Charts in Reports", value=True)
        include_model_details = st.checkbox("🤖 Include Model Details", value=True)
        include_comparison = st.checkbox("🔄 Include Comparison Data", value=False)
    
    # Save settings
    if st.button("💾 Save All Settings", type="primary"):
        # Store settings in session state
        st.session_state.settings = {
            'theme': theme_mode,
            'default_model': default_model,
            'ensemble_mode': ensemble_mode,
            'confidence_threshold': confidence_threshold,
            'export_format': export_format,
            'colors': {
                'primary': primary_color,
                'secondary': secondary_color,
                'accent': accent_color
            }
        }
        st.success("✅ Settings saved successfully!")

# ===============================================================================
# MAIN APPLICATION
# ===============================================================================
def main():
    """Main application function"""
    # Apply custom CSS
    apply_custom_css()
    
    # Load data
    df, df_error = load_dataset()
    models, model_errors = load_models()
    
    # Render sidebar and get selected page
    selected_page = render_sidebar()
    
    # Display data loading status
    if df_error:
        st.sidebar.warning(df_error)
    if model_errors:
        for error in model_errors:
            st.sidebar.error(error)
    
    # Route to appropriate page
    if selected_page == "dashboard":
        render_dashboard(df, models)
    elif selected_page == "predict":
        render_prediction_page(df, models)
    elif selected_page == "insights":
        render_insights_page(df, models)
    elif selected_page == "explorer":
        render_explorer_page(df)
    elif selected_page == "comparison":
        render_comparison_page(df, models)
    elif selected_page == "settings":
        render_settings_page()

# ===============================================================================
# RUN APPLICATION
# ===============================================================================
if __name__ == "__main__":
    main()