import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ===================== Page Configuration =====================
st.set_page_config(
    page_title="💻 Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

# ===================== Custom CSS =====================
st.markdown("""
    <style>
        /* Main App Background */
        .main {
            background-color: #f8fafc;
            padding: 0;
            margin: 0;
        }

        /* Title Styling */
        h1 {
            color: #0F4C81;
            font-weight: 700;
            font-size: 36px;
        }

        /* Subtitle Styling */
        h2, h3 {
            color: #1B1B1B;
            font-weight: 600;
        }

        /* Buttons */
        div.stButton > button {
            background-color: #0F4C81;
            color: white;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            border: none;
            font-size: 16px;
            font-weight: 600;
            transition: 0.3s ease-in-out;
        }
        div.stButton > button:hover {
            background-color: #0b3b66;
            transform: scale(1.03);
        }

        /* Cards / Containers */
        .stMarkdown, .stDataFrame, .stPlotlyChart {
            background-color: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }

        /* Input Fields */
        .stNumberInput, .stSelectbox {
            border-radius: 8px;
        }

        /* Footer Hide */
        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ===================== Load Dataset =====================
df = pickle.load(open('df.pkl', 'rb'))

# ===================== Sidebar =====================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2920/2920244.png", width=120)
st.sidebar.title("Laptop Price Predictor")
st.sidebar.markdown("### Configure your laptop specs")

company = st.sidebar.selectbox("Brand", df['Company'].unique())
laptop_type = st.sidebar.selectbox("Laptop Type", df['TypeName'].unique())
ram = st.sidebar.selectbox("RAM (GB)", sorted(df['Ram'].unique()))
weight = st.sidebar.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
touchscreen = st.sidebar.selectbox("Touchscreen", ['Yes', 'No'])
ips = st.sidebar.selectbox("IPS Display", ['Yes', 'No'])
screen_size = st.sidebar.number_input("Screen Size (inches)", min_value=10.0, max_value=18.0, value=15.6, step=0.1)
resolution = st.sidebar.selectbox("Screen Resolution", df['Resolution'].unique())
cpu = st.sidebar.selectbox("CPU", df['Cpu brand'].unique())
hdd = st.sidebar.selectbox("HDD (GB)", sorted(df['Hdd'].unique()))
ssd = st.sidebar.selectbox("SSD (GB)", sorted(df['Ssd'].unique()))
gpu = st.sidebar.selectbox("GPU Brand", df['Gpu brand'].unique())
os = st.sidebar.selectbox("Operating System", df['os'].unique())

# ===================== Prediction =====================
if st.sidebar.button("💡 Predict Price"):
    model = pickle.load(open('model.pkl', 'rb'))

    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    input_features = np.array([company, laptop_type, ram, weight, touchscreen,
                               ips, screen_size, resolution, cpu, hdd, ssd, gpu, os], dtype=object)
    
    encoded_df = pd.DataFrame([input_features], columns=['Company','TypeName','Ram','Weight',
                                                          'Touchscreen','Ips','ScreenSize',
                                                          'Resolution','Cpu brand','Hdd',
                                                          'Ssd','Gpu brand','os'])
    
    prediction = model.predict(encoded_df)
    st.markdown(f"""
        <div style="background-color:#0F4C81;padding:18px;border-radius:10px;text-align:center;">
            <h2 style="color:white;">💰 Predicted Price: ₹ {int(prediction[0]):,}</h2>
        </div>
    """, unsafe_allow_html=True)

# ===================== Dashboard Layout =====================
st.markdown("---")
st.subheader("📊 Laptop Dataset Insights")

# ---- Top Brands ----
col1, col2 = st.columns(2)
with col1:
    brand_counts = df['Company'].value_counts().reset_index()
    brand_counts.columns = ['Brand', 'Count']
    fig1 = px.bar(brand_counts, x="Brand", y="Count", title="Top Laptop Brands",
                  color="Brand", template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)

# ---- Laptop Type Pie Chart ----
with col2:
    type_counts = df['TypeName'].value_counts().reset_index()
    type_counts.columns = ['Type', 'Count']
    fig2 = px.pie(type_counts, values="Count", names="Type", title="Laptop Types Distribution",
                  hole=0.4, color_discrete_sequence=px.colors.sequential.Blues)
    st.plotly_chart(fig2, use_container_width=True)

# ---- RAM vs Price ----
st.markdown("### 💾 RAM vs Price")
fig3 = px.box(df, x="Ram", y="Price", color="Ram", template="plotly_white")
st.plotly_chart(fig3, use_container_width=True)
