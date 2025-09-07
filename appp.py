
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ---------------------------
# Helpers
# ---------------------------
def load_pickle_safe(path):
    p = Path(path)
    if not p.exists():
        st.error(f"Required file not found: `{path}` — make sure it's in the app folder.")
        st.stop()
    return pickle.load(open(p, "rb"))

def calc_ppi(resolution, screen_size):
    X_res, Y_res = map(int, resolution.split('x'))
    return ((X_res**2 + Y_res**2)**0.5) / screen_size

def predict_all(models, query_df):
    preds = {}
    for name, model in models.items():
        try:
            val = model.predict(query_df)[0]
        except Exception as e:
            st.warning(f"Prediction failed for {name}: {str(e)}")
            val = np.nan
        if not pd.isna(val):
            try:
                price = int(np.exp(val))
            except Exception:
                price = int(val)
        else:
            price = None
        preds[name] = price
    return preds

def mk_metric_card(title, value, subtitle=""):
    st.markdown(
        f"""
        <div class="card-metric">
            <div class="card-metric-title">{title}</div>
            <div class="card-metric-value">{value}</div>
            <div class="card-metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------
# Load data & models
# ---------------------------
st.set_page_config(page_title="Laptop Price AI", page_icon="💻", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    :root{
      --card-bg:#ffffff;
      --muted:#6b7280;
      --primary:#0055CC;
      --accent:#003366;
      --card-radius:14px;
      --shadow: 0 6px 24px rgba(18, 38, 63, 0.08);
    }
    body { background: #F5F6F9; }
    .topbar { position:sticky; top:0; z-index:100; background:#fff; padding:18px 24px; display:flex; align-items:center; justify-content:space-between; box-shadow:0 2px 10px rgba(0,0,0,0.05); }
    .brand { font-weight:700; color:var(--primary); font-size:20px; display:flex; gap:10px; align-items:center; }
    .nav-pill { background:#fff; border-radius:12px; padding:8px 16px; margin-right:8px; border:1px solid rgba(31,59,87,0.1); cursor:pointer; font-weight:500; color:var(--primary); transition:all 0.2s; }
    .nav-pill.active { background:var(--primary); color:white; }
    .main-card { background:var(--card-bg); border-radius:var(--card-radius); padding:20px; box-shadow:var(--shadow); }
    .card-metric { padding:16px; border-radius:12px; background:var(--card-bg); box-shadow:var(--shadow); }
    .card-metric-title { color:var(--muted); font-size:13px; }
    .card-metric-value { font-size:22px; font-weight:700; margin-top:6px; color:var(--primary); }
    .card-metric-sub { color:var(--muted); font-size:12px; margin-top:4px; }
    .model-card { padding:16px; border-radius:12px; background:var(--card-bg); box-shadow:var(--shadow); margin-bottom:16px; }
    .model-title { font-weight:700; color:var(--primary); }
    .badge { display:inline-block; font-size:11px; padding:6px 10px; border-radius:999px; background:var(--primary); color:white; }
    .right-panel { background:transparent; }
    .small-muted { color:var(--muted); font-size:13px; }
    .model-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }
    @media (max-width: 1100px) {
      .model-grid { grid-template-columns:repeat(1,1fr); }
    }
    </style>
    """, unsafe_allow_html=True
)

# Load data
df = load_pickle_safe("df.pkl")

models = {
    "Linear Regression": load_pickle_safe("lin_reg.pkl"),
    "Ridge Regression": load_pickle_safe("Ridge_regre.pkl"),
    "Lasso Regression": load_pickle_safe("lasso_reg.pkl"),
    "KNN Regressor": load_pickle_safe("KNN_reg.pkl"),
    "Decision Tree": load_pickle_safe("Decision_tree.pkl"),
    "SVM Regressor": load_pickle_safe("SVM_reg.pkl"),
    "Random Forest": load_pickle_safe("Random_forest.pkl"),
    "Extra Trees": load_pickle_safe("Extra_tree.pkl"),
    "AdaBoost": load_pickle_safe("Ada_boost.pkl"),
    "Gradient Boost": load_pickle_safe("Gradient_boost.pkl"),
    "XGBoost": load_pickle_safe("XG_boost.pkl")
}

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

model_categories = {
    "Ensemble Methods": ["Random Forest", "Extra Trees", "XGBoost", "Gradient Boost", "AdaBoost"], 
    "Linear Methods": ["Linear Regression", "Ridge Regression", "Lasso Regression"],
    "Other Methods": ["KNN Regressor", "SVM Regressor"]
}

# ---------------------------
# Navigation
# ---------------------------
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# Top navigation bar
st.markdown('<div class="topbar">', unsafe_allow_html=True)
st.markdown('<div class="brand">🧠 Laptop Price AI</div>', unsafe_allow_html=True)

nav_html = '<div style="display:flex; gap:10px;">'
for tab in ["Dashboard", "Price Predictor", "Model Insights"]:
    active = "active" if st.session_state.page == tab else ""
    nav_html += f'<div class="nav-pill {active}" onclick="window.location.href='/?page={tab}'">{tab}</div>'
nav_html += '</div>'
st.markdown(nav_html, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Sidebar as alternative
page = st.sidebar.radio("Navigation", ["Dashboard", "Price Predictor", "Model Insights"])

# Update session state when sidebar changes
st.session_state.page = page

# ---------------------------
# Dashboard Page
# ---------------------------
if st.session_state.page == "Dashboard":
    st.header("📊 Model Performance Overview")
    best_model = max(accuracies, key=lambda k: accuracies[k]["R2"])
    best_r2 = accuracies[best_model]["R2"]
    best_mae_model = min(accuracies, key=lambda k: accuracies[k]["MAE"])
    avg_r2 = np.mean([v["R2"] for v in accuracies.values()])

    k1, k2, k3, k4 = st.columns([1,1,1,1])
    with k1:
        mk_metric_card("Best Model", best_model, f"R² Score: {best_r2:.2f}")
    with k2:
        mk_metric_card("Lowest MAE", best_mae_model, f"MAE: ₹{accuracies[best_mae_model]['MAE']:,}")
    with k3:
        mk_metric_card("Total Models", f"{len(models)}", "ML algorithms available")
    with k4:
        mk_metric_card("Avg R²", f"{avg_r2:.2f}", "Mean R² Score")

    r2_df = pd.DataFrame({
        "Model": list(accuracies.keys()),
        "R2": [v["R2"] for v in accuracies.values()],
        "MAE": [v["MAE"] for v in accuracies.values()]
    }).sort_values("R2", ascending=False)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**R² Score Ranking**")
        fig_r2 = px.bar(r2_df, x="R2", y="Model", orientation='h', text="R2",
                        width=700, height=520, color="R2", color_continuous_scale="Blues")
        st.plotly_chart(fig_r2, use_container_width=True)
    with col_b:
        st.markdown("**Mean Absolute Error**")
        fig_mae = px.bar(r2_df, x="MAE", y="Model", orientation='h', text="MAE",
                         width=700, height=520, color="MAE", color_continuous_scale="reds")
        st.plotly_chart(fig_mae, use_container_width=True)

# ---------------------------
# Price Predictor Page
# ---------------------------
elif st.session_state.page == "Price Predictor":
    st.header("🔮 Price Predictor")
    st.markdown("Configure laptop specs and predict price using multiple ML models.")

    left_col, right_col = st.columns([1, 1.1])
    with left_col:
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.subheader("Laptop Specifications")
        company = st.selectbox('Brand', df['Company'].unique())
        laptop_type = st.selectbox('Type', df['TypeName'].unique())
        ram = st.selectbox('RAM (GB)', sorted(df['Ram'].unique()))
        weight = st.slider('Weight (kg)', min_value=0.5, max_value=5.0, value=1.5, step=0.1)
        touchscreen = st.radio('Touchscreen', ['No', 'Yes'], horizontal=True)
        ips = st.radio('IPS display', ['No', 'Yes'], horizontal=True)
        screen_size = st.slider('Screen Size (inches)', 10.0, 18.0, 13.3, 0.1)
        resolution = st.selectbox('Resolution', ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
        cpu = st.selectbox('CPU', df['Cpu brand'].unique())
        gpu = st.selectbox('GPU', df['Gpu brand'].unique())
        hdd = st.selectbox('HDD (GB)', [0,128,256,512,1024,2048])
        ssd = st.selectbox('SSD (GB)', [0,8,128,256,512,1024])
        os = st.selectbox('Operating System', df['os'].unique())
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("<div class='main-card right-panel'>", unsafe_allow_html=True)
        st.subheader("Detailed ML Predictions")
        st.markdown("<div style='color:var(--muted)'>Average predicted price and model-by-model breakdown</div>")

        if st.button("Predict Price"):
            ts = 1 if touchscreen == 'Yes' else 0
            ips_v = 1 if ips == 'Yes' else 0
            ppi = calc_ppi(resolution, screen_size)

            query = pd.DataFrame([[company, laptop_type, ram, weight, ts, ips_v, ppi, cpu, hdd, ssd, gpu, os]],
                                 columns=['Company','TypeName','Ram','Weight','Touchscreen','Ips','ppi','Cpu brand','HDD','SSD','Gpu brand','os'])

            with st.spinner("Running predictions across models..."):
                predictions = predict_all(models, query)

            preds_df = pd.DataFrame(list(predictions.items()), columns=['Model','Price']).dropna().sort_values('Price', ascending=False).reset_index(drop=True)
            avg_price = int(preds_df['Price'].mean())
            st.markdown(f"**Average:** ₹{avg_price:,}")
            highest = preds_df.iloc[0]
            lowest = preds_df.iloc[-1]
            st.markdown(f"**Highest prediction:** {highest['Model']} → ₹{highest['Price']:,}")
            st.markdown(f"**Lowest prediction:** {lowest['Model']} → ₹{lowest['Price']:,}")
            st.markdown("---")

            fig_price = px.bar(preds_df, x='Model', y='Price', text='Price', title="Model-wise Predicted Price")
            fig_price.update_traces(texttemplate='₹%{y:,}', textposition='outside')
            st.plotly_chart(fig_price, use_container_width=True)
        else:
            st.info("Fill the laptop specs on the left and click Predict Price to see model comparisons.")
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Model Insights Page
# ---------------------------
elif st.session_state.page == "Model Insights":
    st.header("📈 Model Insights")
    r2_df = pd.DataFrame({
        "Model": list(accuracies.keys()),
        "R2": [v["R2"] for v in accuracies.values()],
        "MAE": [v["MAE"] for v in accuracies.values()]
    }).sort_values("R2", ascending=False)

    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("R² Score Ranking")
        fig = go.Figure()
        fig.add_trace(go.Bar(y=r2_df["Model"], x=r2_df["R2"], orientation='h', marker_color='#0055CC'))
        fig.update_layout(height=560, margin=dict(l=0,r=20,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Mean Absolute Error")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(y=r2_df["Model"], x=r2_df["MAE"], orientation='h', marker_color='#003366'))
        fig2.update_layout(height=560, margin=dict(l=0,r=20,t=20,b=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Full Accuracy Table")
    st.dataframe(r2_df.set_index("Model"))
