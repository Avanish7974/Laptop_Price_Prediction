import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="💻 Advanced Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
        /* Global Font & Body Styling */
        body {
            background-color: #f8f9fa;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #212529;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(135deg, #0077b6, #00b4d8);
            color: white;
        }

        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: white;
            text-align: center;
            font-weight: bold;
        }

        /* Header Title Styling */
        .main-title {
            text-align: center;
            font-size: 38px;
            font-weight: 800;
            color: #0077b6;
            margin-bottom: 5px;
        }

        /* Subtitle Styling */
        .sub-title {
            text-align: center;
            font-size: 18px;
            color: #495057;
            margin-bottom: 25px;
        }

        /* Prediction Result Card */
        .prediction-card {
            background: rgba(255, 255, 255, 0.85);
            padding: 25px;
            border-radius: 18px;
            box-shadow: 0 4px 15px rgba(0, 119, 182, 0.3);
            text-align: center;
            transition: transform 0.3s ease-in-out;
        }
        .prediction-card:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 20px rgba(0, 180, 216, 0.5);
        }

        /* Buttons */
        div.stButton > button {
            background-color: #0077b6;
            color: white;
            padding: 0.6em 1em;
            font-size: 1rem;
            border-radius: 10px;
            border: none;
            font-weight: bold;
            transition: all 0.3s ease-in-out;
        }
        div.stButton > button:hover {
            background-color: #00b4d8;
            color: white;
            transform: scale(1.05);
        }

        /* Chart Card Styling */
        .chart-card {
            background-color: white;
            padding: 20px;
            border-radius: 16px;
            box-shadow: 0px 4px 14px rgba(0,0,0,0.08);
        }

        /* Table Styling */
        .dataframe {
            border: 1px solid #dee2e6;
            border-radius: 8px;
            background-color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL ====================
model_path = Path("laptop_price_model.pkl")
df_path = Path("df.pkl")

if not model_path.exists() or not df_path.exists():
    st.error("❌ Required model or dataset file not found. Please check the files.")
    st.stop()

model = pickle.load(open(model_path, "rb"))
df = pickle.load(open(df_path, "rb"))

# ==================== HEADER ====================
st.markdown("<h1 class='main-title'>💻 Advanced Laptop Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Predict laptop prices with high accuracy using Machine Learning</p>", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
st.sidebar.title("⚙️ Input Specifications")
company = st.sidebar.selectbox("Select Company", df['Company'].unique())
type_name = st.sidebar.selectbox("Select Type", df['TypeName'].unique())
ram = st.sidebar.selectbox("RAM (GB)", sorted(df['Ram'].unique()))
weight = st.sidebar.slider("Weight (Kg)", 0.5, 5.0, 2.0)
touchscreen = st.sidebar.selectbox("Touchscreen", ["Yes", "No"])
ips = st.sidebar.selectbox("IPS Display", ["Yes", "No"])
ppi = st.sidebar.slider("PPI", 50, 400, 150)
cpu = st.sidebar.selectbox("CPU", df['Cpu brand'].unique())
hdd = st.sidebar.selectbox("HDD (GB)", sorted(df['Hdd'].unique()))
ssd = st.sidebar.selectbox("SSD (GB)", sorted(df['Ssd'].unique()))
gpu = st.sidebar.selectbox("GPU Brand", df['Gpu brand'].unique())
os = st.sidebar.selectbox("Operating System", df['os'].unique())

# ==================== PREDICTION ====================
if st.sidebar.button("💡 Predict Price"):
    # Convert categorical values
    touchscreen = 1 if touchscreen == "Yes" else 0
    ips = 1 if ips == "Yes" else 0

    query = np.array([[company, type_name, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]])
    query_df = pd.DataFrame(query, columns=["Company", "TypeName", "Ram", "Weight", "Touchscreen", "IPS", "PPI", "Cpu brand", "Hdd", "Ssd", "Gpu brand", "os"])

    # Predict price
    price = np.exp(model.predict(query_df)[0])

    st.markdown(f"""
        <div class="prediction-card">
            <h3>💰 Estimated Laptop Price:</h3>
            <h1 style="color:#0077b6;">₹ {int(price):,}</h1>
            <p style="color:#495057;">Based on your selected configuration</p>
        </div>
    """, unsafe_allow_html=True)

# ==================== DATA VISUALIZATION ====================
st.subheader("📊 Data Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    fig1 = px.histogram(df, x="Company", color="Company", title="Laptop Distribution by Company")
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    fig2 = px.pie(df, names="TypeName", title="Laptop Types Distribution")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
fig3 = px.box(df, x="Company", y="Price", color="Company", title="Price Range by Company")
st.plotly_chart(fig3, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)
