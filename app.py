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
# Page Config
# ---------------------------
st.set_page_config(page_title="Laptop Price AI", page_icon="💻", layout="wide")

# ---------------------------
# Custom CSS Styling
# ---------------------------
st.markdown(
    """
    <style>
    :root {
      --bg-color: #F5F6F9;
      --card-bg: #FFFFFF;
      --primary: #0055CC;
      --accent: #003366;
      --text-color: #1C1C1C;
      --muted: #6B7280;
      --card-radius: 14px;
      --shadow: 0 6px 24px rgba(0, 0, 0, 0.08);
    }
    body { background: var(--bg-color); }
    .topbar {
        position: sticky;
        top: 0;
        z-index: 100;
        background: var(--card-bg);
        padding: 14px 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: var(--shadow);
        border-radius: 0 0 12px 12px;
    }
    .brand {
        font-weight: 700;
        color: var(--accent);
        font-size: 22px;
        display: flex;
        gap: 10px;
        align-items: center;
    }
    .nav-container {
        display: flex;
        gap: 12px;
    }
    .nav-btn {
        padding: 8px 18px;
        border-radius: 8px;
        background: transparent;
        border: 1px solid transparent;
        font-weight: 600;
        color: var(--muted);
        cursor: pointer;
        transition: all 0.2s ease-in-out;
    }
    .nav-btn:hover {
        background: #eaf1ff;
        color: var(--primary);
    }
    .nav-btn.active {
        background: var(--primary);
        color: #fff;
        border: 1px solid var(--primary);
        box-shadow: 0 3px 10px rgba(0,85,204,0.2);
    }
    .main-card {
        background: var(--card-bg);
        border-radius: var(--card-radius);
        padding: 20px;
        box-shadow: var(--shadow);
        margin-bottom: 16px;
    }
    .card-metric {
        padding: 16px;
        border-radius: 12px;
        background: var(--card-bg);
        box-shadow: var(--shadow);
    }
    .card-metric-title {
        color: var(--muted);
        font-size: 13px;
    }
    .card-metric-value {
        font-size: 22px;
        font-weight: 700;
        margin-top: 6px;
        color: var(--primary);
    }
    .card-metric-sub {
        color: var(--muted);
        font-size: 12px;
        margin-top: 4px;
    }
    .model-card {
        padding: 16px;
        border-radius: 12px;
        background: var(--card-bg);
        box-shadow: var(--shadow);
        margin-bottom: 16px;
    }
    .model-title {
        font-weight: 700;
        color: var(--accent);
    }
    .badge {
        display: inline-block;
        font-size: 11px;
        padding: 6px 10px;
        border-radius: 999px;
        background: var(--primary);
        color: #fff;
        font-weight: 600;
    }
    .model-grid {
        display: grid;
        grid-template-columns: repeat(3,1fr);
        gap: 16px;
    }
    @media (max-width: 1100px) {
      .model-grid { grid-template-columns: repeat(1,1fr); }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Load data & models
# ---------------------------
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

# accuracies (static for now)
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
# Top Navigation
# ---------------------------
st.markdown('<div class="topbar">', unsafe_allow_html=True)
st.markdown('<div class="brand">💻 Laptop Price AI</div>', unsafe_allow_html=True)

# Navigation buttons
nav_buttons_html = '<div class="nav-container">'
pages = ["Dashboard", "Price Predictor", "Model Insights"]
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
for p in pages:
    active = "active" if st.session_state.page == p else ""
    nav_buttons_html += f'<button class="nav-btn {active}" onclick="window.location.search=\'?page={p}\'">{p}</button>'
nav_buttons_html += "</div>"
st.markdown(nav_buttons_html, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Page selection
query_params = st.experimental_get_query_params()
if "page" in query_params:
    st.session_state.page = query_params["page"][0]
page = st.session_state.page

# ---------------------------
# Page Routing
# ---------------------------
if page == "Dashboard":
    st.header("📊 Model Performance Overview")
    st.write("Comprehensive summary of model accuracy and errors.")
    st.markdown("---")
    # KPIs
    best_model = max(accuracies, key=lambda k: accuracies[k]["R2"])
    best_r2 = accuracies[best_model]["R2"]
    best_mae_model = min(accuracies, key=lambda k: accuracies[k]["MAE"])
    avg_r2 = np.mean([v["R2"] for v in accuracies.values()])

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        mk_metric_card("Best Model", best_model, f"R² Score: {best_r2:.2f}")
    with k2:
        mk_metric_card("Lowest MAE", best_mae_model, f"MAE: ₹{accuracies[best_mae_model]['MAE']:,}")
    with k3:
        mk_metric_card("Total Models", f"{len(models)}", "ML algorithms available")
    with k4:
        mk_metric_card("Avg R²", f"{avg_r2:.2f}", "Mean R² Score")
    st.markdown("---")

    # R² and MAE charts
    r2_df = pd.DataFrame({
        "Model": list(accuracies.keys()),
        "R2": [v["R2"] for v in accuracies.values()],
        "MAE": [v["MAE"] for v in accuracies.values()]
    }).sort_values("R2", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**R² Score Ranking**")
        fig_r2 = px.bar(r2_df, x="R2", y="Model", orientation="h", text="R2", color_discrete_sequence=["#0055CC"])
        fig_r2.update_traces(texttemplate='%{x:.2f}', textposition='outside')
        st.plotly_chart(fig_r2, use_container_width=True)
    with col2:
        st.markdown("**Mean Absolute Error**")
        fig_mae = px.bar(r2_df, x="MAE", y="Model", orientation="h", text="MAE", color_discrete_sequence=["#003366"])
        fig_mae.update_traces(texttemplate='₹%{x:,}', textposition='outside')
        st.plotly_chart(fig_mae, use_container_width=True)

    # Detailed Model Cards
    st.subheader("Detailed Model Analysis")
    st.markdown("<div class='model-grid'>", unsafe_allow_html=True)
    for m in r2_df["Model"]:
        r2 = accuracies[m]["R2"]
        mae = accuracies[m]["MAE"]
        badge = "Best" if r2 >= 0.9 else "Good" if r2 >= 0.85 else "Fair"
        card_html = f"""
            <div class="model-card">
                <div style='display:flex;justify-content:space-between;align-items:center'>
                    <div class='model-title'>{m}</div>
                    <div class='badge'>{badge}</div>
                </div>
                <div style='margin-top:12px;color:var(--muted)'>R² Score: <b>{r2:.2f}</b></div>
                <div style='color:var(--muted); margin-top:6px'>MAE: <b>₹{mae:,}</b></div>
                <div style='margin-top:10px'>
                    <div style='background:#f1f5f9; height:10px; border-radius:8px;'>
                        <div style='width:{int(r2*100)}%; background:linear-gradient(90deg,#0055CC,#003366); height:10px; border-radius:8px;'></div>
                    </div>
                </div>
            </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Price Predictor":
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
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
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
            st.markdown("")
            highest = preds_df.iloc[0]
            lowest = preds_df.iloc[-1]
            st.markdown(f"**Highest prediction:** {highest['Model']} → ₹{highest['Price']:,}")
            st.markdown(f"**Lowest prediction:** {lowest['Model']} → ₹{lowest['Price']:,}")
            st.markdown("---")
            for idx, row in preds_df.iterrows():
                m = row['Model']; price = row['Price']; r2 = accuracies[m]['R2']; mae = accuracies[m]['MAE']
                percent_from_avg = ((price - avg_price) / avg_price) * 100
                percent_label = f"{percent_from_avg:+.1f}% vs avg"
                card_html = f"""
                    <div class='model-card'>
                      <div style='display:flex;justify-content:space-between;align-items:center'>
                        <div style='font-weight:700'>{idx+1}. {m}</div>
                        <div style='text-align:right'>
                          <div style='font-size:18px; font-weight:700'>₹{price:,}</div>
                          <div style='color:var(--muted); font-size:12px'>{percent_label}</div>
                        </div>
                      </div>
                      <div style='margin-top:8px;'>
                        <span class='small-muted'>R²: <b>{r2:.2f}</b> &nbsp;&nbsp; MAE: <b>₹{mae:,}</b></span>
                        <div style='height:8px; background:#f1f5f9; border-radius:8px; margin-top:8px'>
                          <div style='width:{int(r2*100)}%; background:linear-gradient(90deg,#0055CC,#003366); height:8px;border-radius:8px;'></div>
                        </div>
                      </div>
                    </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
            fig_price = px.bar(preds_df, x='Model', y='Price', text='Price', title="Model-wise Predicted Price", color_discrete_sequence=["#0055CC"])
            fig_price.update_traces(texttemplate='₹%{y:,}', textposition='outside')
            fig_price.update_layout(margin=dict(t = 30, b = 0, l = 0, r = 0))
            st.plotly_chart(fig_price, use_container_width=True)
        else:
            st.info("Configure laptop specs and click 'Predict Price' to see results.")
        st.markdown("</div>", unsafe_allow_html=True)
