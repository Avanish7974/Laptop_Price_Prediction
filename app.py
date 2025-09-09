import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit.components.v1 import html as st_html

# ===================== Page Configuration =====================
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== Particle Background (tsParticles) =====================
# This component injects a full-screen particle canvas behind the app content.
PARTICLE_HTML = """
<div id="tsparticles" style="position:fixed; top:0; left:0; width:100%; height:100vh; z-index:0; pointer-events:none;"></div>
<script src="https://cdn.jsdelivr.net/npm/tsparticles@2.10.1/tsparticles.bundle.min.js"></script>
<script>
  (async () => {
    await tsParticles.load("tsparticles", {
      fullScreen: { enable: false },
      fpsLimit: 60,
      detectRetina: true,
      background: { color: "transparent" },
      particles: {
        number: { value: 90, density: { enable: true, area: 1200 } },
        color: { value: ["#00f5ff", "#7c4dff", "#00e676", "#ff6b6b"] },
        shape: { type: "circle" },
        opacity: {
          value: 0.6,
          random: { enable: true, minimumValue: 0.2 },
          anim: { enable: true, speed: 0.8, opacity_min: 0.2, sync: false }
        },
        size: {
          value: { min: 1, max: 6 },
          random: true,
          anim: { enable: true, speed: 3, size_min: 0.5, sync: false }
        },
        links: {
          enable: true,
          distance: 160,
          color: "#00f5ff",
          opacity: 0.12,
          width: 1
        },
        move: {
          enable: true,
          speed: 0.8,
          direction: "none",
          random: true,
          straight: false,
          out_mode: "out",
          attract: { enable: false, rotateX: 600, rotateY: 1200 }
        }
      },
      interactivity: {
        detectsOn: "canvas",
        events: {
          onHover: { enable: true, mode: "repulse" },
          onClick: { enable: true, mode: "push" },
          resize: true
        },
        modes: {
          grab: { distance: 400, links: { opacity: 0.5 } },
          bubble: { distance: 400, size: 8, duration: 2, opacity: 0.8 },
          repulse: { distance: 120, duration: 0.6 },
          push: { quantity: 4 },
          remove: { quantity: 2 }
        }
      },
      retina_detect: true
    });
  })();
</script>
"""

# Render the particle HTML (full width). Use height small so component doesn't change layout.
st_html(PARTICLE_HTML, height=1)

# ===================== Custom Styling (app content sits above particles) =====================
st.markdown(
    """
    <style>
    /* Ensure app content is above the particle canvas */
    .stApp {
        background: linear-gradient(135deg, rgba(15,32,39,0.85), rgba(32,58,67,0.85));
        color: #ffffff;
        position: relative;
        z-index: 1;
        font-family: 'Segoe UI', Roboto, sans-serif;
    }

    /* Card / glass effect for content */
    .glass-card {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 14px;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 8px 30px rgba(2,6,23,0.6);
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #00d4ff, #0066ff) !important;
        color: white !important;
        border-radius: 10px;
        padding: 10px 18px;
        font-weight: 600;
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.12);
    }
    .stButton>button:hover {
        transform: scale(1.03);
    }

    /* Metric style */
    [data-testid="stMetricValue"] {
        color: #00f5ff !important;
        font-weight: 700;
    }

    /* DataFrame card */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Sidebar tweaks */
    [data-testid="stSidebar"] {
        background: rgba(4,8,15,0.9);
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* Reduce top padding so header feels closer */
    header ~ div { padding-top: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===================== Load Data & Models =====================
@st.cache_data
def load_data():
    df = pickle.load(open('df.pkl', 'rb'))
    models = {
        "Linear Regression": pickle.load(open("lin_reg.pkl", "rb")),
        "Ridge Regression": pickle.load(open("Ridge_regre.pkl", "rb")),
        "Lasso Regression": pickle.load(open("lasso_reg.pkl", "rb")),
        "KNN Regressor": pickle.load(open("KNN_reg.pkl", "rb")),
        "Decision Tree": pickle.load(open("Decision_tree.pkl", "rb")),
        "SVM Regressor": pickle.load(open("SVM_reg.pkl", "rb")),
        "Random Forest": pickle.load(open("Random_forest.pkl", "rb")),
        "Extra Trees": pickle.load(open("Extra_tree.pkl", "rb")),
        "AdaBoost": pickle.load(open("Ada_boost.pkl", "rb")),
        "Gradient Boost": pickle.load(open("Gradient_boost.pkl", "rb")),
        "XGBoost": pickle.load(open("XG_boost.pkl", "rb"))
    }
    return df, models

df, models = load_data()

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

# ===================== Sidebar Navigation =====================
st.sidebar.title("🚀 Navigation")
page = st.sidebar.radio("Choose Page", ["📊 Dashboard", "🔮 Price Predictor", "📈 Model Insights"])

# =========================================================================================
# PAGE 1: DASHBOARD
# =========================================================================================
if page == "📊 Dashboard":
    st.title("📊 Laptop Price Prediction Dashboard")
    st.markdown("### Futuristic overview of model performance")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    acc_df = pd.DataFrame(accuracies).T.reset_index().rename(columns={'index': 'Model'})

    col1, col2, col3 = st.columns(3)
    best_r2_model = max(accuracies, key=lambda x: accuracies[x]['R2'])
    lowest_mae_model = min(accuracies, key=lambda x: accuracies[x]['MAE'])

    with col1:
        st.metric("🏆 Best Model", best_r2_model, f"{accuracies[best_r2_model]['R2']:.2f}")
    with col2:
        st.metric("📉 Lowest MAE", lowest_mae_model, f"₹{accuracies[lowest_mae_model]['MAE']:,}")
    with col3:
        st.metric("🤖 Total Models", len(models))

    st.markdown("</div>", unsafe_allow_html=True)

    # Radar Chart (Model R² Comparison)
    st.subheader("🌐 Model Accuracy Radar")
    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=acc_df['R2'],
        theta=acc_df['Model'],
        fill='toself',
        name='R² Scores',
        line=dict(color='#00f5ff')
    ))
    radar_fig.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=True, range=[0,1])))
    st.plotly_chart(radar_fig, use_container_width=True)

    # 3D Bubble Chart (R² vs MAE vs Model)
    st.subheader("🔮 Model Performance 3D Visualization")
    bubble_fig = px.scatter_3d(
        acc_df,
        x='R2', y='MAE', z='Model',
        color='R2',
        size='MAE',
        symbol='Model',
        color_continuous_scale='Viridis',
    )
    bubble_fig.update_traces(marker=dict(opacity=0.85))
    bubble_fig.update_layout(template="plotly_dark", height=520)
    st.plotly_chart(bubble_fig, use_container_width=True)

    # Gauge Chart for Best Model R²
    st.subheader("⚡ Best Model Performance Indicator")
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=accuracies[best_r2_model]['R2'] * 100,
        title={'text': f"{best_r2_model} R² (%)"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#00f5ff"}}
    ))
    gauge_fig.update_layout(template="plotly_dark", height=320)
    st.plotly_chart(gauge_fig, use_container_width=True)

# =========================================================================================
# PAGE 2: PRICE PREDICTOR
# =========================================================================================
elif page == "🔮 Price Predictor":
    st.title("🔮 Laptop Price Prediction")
    st.markdown("### Enter laptop specifications below to predict price.")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            company = st.selectbox('🏢 Brand', df['Company'].unique())
            laptop_type = st.selectbox('💼 Type', df['TypeName'].unique())
            ram = st.selectbox('💾 RAM (GB)', sorted(df['Ram'].unique()))
            weight = st.number_input('⚖️ Weight (kg)', min_value=0.5, max_value=5.0, step=0.1, value=1.5)
            touchscreen = st.radio('🖐️ Touchscreen', ['No','Yes'], horizontal=True)
            ips = st.radio('🖼️ IPS Display', ['No','Yes'], horizontal=True)

        with col2:
            screen_size = st.slider('📏 Screen Size (inches)', 10.0, 18.0, 15.6)
            resolution = st.selectbox('🔳 Resolution', [
                '1920x1080','1366x768','1600x900','3840x2160',
                '3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'
            ])
            cpu = st.selectbox('🖥️ CPU', df['Cpu brand'].unique())
            hdd = st.selectbox('💽 HDD (GB)', sorted(df['HDD'].unique()))
            ssd = st.selectbox('⚡ SSD (GB)', sorted(df['SSD'].unique()))
            gpu = st.selectbox('🎮 GPU', df['Gpu brand'].unique())
            os = st.selectbox('🖥️ Operating System', df['os'].unique())

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔍 Predict Price"):
        touchscreen_val = 1 if touchscreen == 'Yes' else 0
        ips_val = 1 if ips == 'Yes' else 0
        X_res, Y_res = map(int, resolution.split('x'))
        ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

        query = pd.DataFrame([[company, laptop_type, ram, weight, touchscreen_val, ips_val,
                               ppi, cpu, hdd, ssd, gpu, os]],
                             columns=['Company','TypeName','Ram','Weight','Touchscreen','Ips',
                                      'ppi','Cpu brand','HDD','SSD','Gpu brand','os'])

        predictions = {}
        for name, model in models.items():
            # make sure model.predict returns log(price) as before
            predictions[name] = int(np.exp(model.predict(query)[0]))

        pred_df = pd.DataFrame(predictions.items(), columns=['Model', 'Predicted Price'])

        # Treemap for Predictions
        st.subheader("🌳 Predicted Prices Treemap")
        tree_fig = px.treemap(pred_df, path=['Model'], values='Predicted Price',
                              color='Predicted Price', color_continuous_scale='Viridis')
        tree_fig.update_layout(template="plotly_dark", height=480)
        st.plotly_chart(tree_fig, use_container_width=True)

        # Animated-like line chart (with markers)
        st.subheader("📈 Price Prediction Comparison")
        line_fig = px.line(pred_df.sort_values('Predicted Price'), x='Model', y='Predicted Price',
                           markers=True)
        line_fig.update_traces(marker=dict(size=10), line=dict(width=3))
        line_fig.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(line_fig, use_container_width=True)

# =========================================================================================
# PAGE 3: MODEL INSIGHTS
# =========================================================================================
elif page == "📈 Model Insights":
    st.title("📈 Model Insights & Analysis")
    acc_df = pd.DataFrame(accuracies).T.reset_index().rename(columns={'index': 'Model'})

    # Heatmap for R² vs MAE
    st.subheader("🔥 Model Performance Heatmap")
    # Create matrix-like DataFrame for heatmap (R2 as columns, MAE as values) - approximate view
    heat_df = acc_df.copy()
    heat_df['R2_scaled'] = (heat_df['R2'] - heat_df['R2'].min()) / (heat_df['R2'].max() - heat_df['R2'].min())
    heatmap_fig = px.imshow([heat_df['R2_scaled']], x=heat_df['Model'], y=["R² scaled"],
                             color_continuous_scale='Viridis', aspect="auto")
    heatmap_fig.update_layout(template="plotly_dark", height=220)
    st.plotly_chart(heatmap_fig, use_container_width=True)

    # Boxplot for Model Variability (though limited single-value per model, show as stylized)
    st.subheader("📦 R² Score Distribution (stylized)")
    box_fig = px.box(acc_df, y='R2', x='Model', color='Model')
    box_fig.update_layout(template="plotly_dark", height=520, showlegend=False)
    st.plotly_chart(box_fig, use_container_width=True)

    # Donut Chart for Top 5 Models Contribution
    st.subheader("🍩 Top 5 Models by R² Contribution")
    top5 = acc_df.sort_values(by="R2", ascending=False).head(5)
    donut_fig = px.pie(top5, values='R2', names='Model', hole=0.5,
                       color_discrete_sequence=px.colors.sequential.Viridis)
    donut_fig.update_layout(template="plotly_dark", height=420)
    st.plotly_chart(donut_fig, use_container_width=True)
