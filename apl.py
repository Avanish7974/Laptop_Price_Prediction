import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------- Page Config -------------------
st.set_page_config(
    page_title="Laptop Price Predictor Dashboard",
    page_icon="💻",
    layout="wide"
)

# ------------------- Load Dataset -------------------
df = pickle.load(open('df.pkl', 'rb'))

# ------------------- Load Models -------------------
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

# ------------------- Accuracy Scores -------------------
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
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["📊 Dashboard", "🔮 Price Predictor", "📈 Model Insights"])

# =========================================================================================
# PAGE 1: DASHBOARD
# =========================================================================================
if page == "📊 Dashboard":
    st.title("📊 Laptop Price Prediction Dashboard")
    st.markdown("### Overview of Model Predictions and Performance Metrics")

    # Display key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Model (R²)",
                  max(accuracies, key=lambda x: accuracies[x]['R2']),
                  f"{max([v['R2'] for v in accuracies.values()]):.2f}")
    with col2:
        st.metric("Lowest MAE Model",
                  min(accuracies, key=lambda x: accuracies[x]['MAE']),
                  f"{min([v['MAE'] for v in accuracies.values()]):,.0f}")
    with col3:
        st.metric("Total Models", len(models))

    # Accuracy bar chart
    st.subheader("📈 R² Score Comparison")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(accuracies.keys(), [v['R2'] for v in accuracies.values()], color="skyblue")
    ax.set_ylabel("R² Score")
    ax.set_title("Model Accuracy")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # MAE chart
    st.subheader("📉 Mean Absolute Error Comparison")
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.bar(accuracies.keys(), [v['MAE'] for v in accuracies.values()], color="orange")
    ax2.set_ylabel("MAE")
    ax2.set_title("Model Error Comparison")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

# =========================================================================================
# PAGE 2: PRICE PREDICTOR
# =========================================================================================
elif page == "🔮 Price Predictor":
    st.title("🔮 Laptop Price Prediction Tool")
    st.markdown("### Enter laptop details to predict price using multiple models")

    col1, col2 = st.columns(2)
    with col1:
        company = st.selectbox('🏢 Brand', df['Company'].unique())
        laptop_type = st.selectbox('💼 Type', df['TypeName'].unique())
        ram = st.selectbox('💾 RAM (GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
        weight = st.number_input('⚖️ Weight (kg)', min_value=0.5, max_value=5.0, step=0.1)
        touchscreen = st.radio('🖐️ Touchscreen', ['No', 'Yes'], horizontal=True)
        ips = st.radio('🖼️ IPS Display', ['No', 'Yes'], horizontal=True)

    with col2:
        screen_size = st.slider('📏 Screen Size (inches)', 10.0, 18.0, 13.0)
        resolution = st.selectbox('🔳 Resolution', [
            '1920x1080', '1366x768', '1600x900', '3840x2160',
            '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
        ])
        cpu = st.selectbox('🖥️ CPU', df['Cpu brand'].unique())
        hdd = st.selectbox('💽 HDD (GB)', [0, 128, 256, 512, 1024, 2048])
        ssd = st.selectbox('⚡ SSD (GB)', [0, 8, 128, 256, 512, 1024])
        gpu = st.selectbox('🎮 GPU', df['Gpu brand'].unique())
        os = st.selectbox('🖥️ Operating System', df['os'].unique())

    # Predict button
    if st.button("🔍 Predict Price"):
        touchscreen = 1 if touchscreen == 'Yes' else 0
        ips = 1 if ips == 'Yes' else 0
        X_res, Y_res = map(int, resolution.split('x'))
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

        # Prepare query for prediction
        query = pd.DataFrame([[company, laptop_type, ram, weight, touchscreen, ips,
                               ppi, cpu, hdd, ssd, gpu, os]],
                             columns=['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'Ips',
                                      'ppi', 'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os'])

        predictions = {}
        for name, model in models.items():
            predictions[name] = int(np.exp(model.predict(query)[0]))

        # Display predictions
        st.subheader("📊 Predictions by Model")
        cols = st.columns(3)
        for i, (name, price) in enumerate(predictions.items()):
            with cols[i % 3]:
                st.metric(label=name, value=f"₹ {price:,}", delta=f"R²: {accuracies[name]['R2']:.2f}")

        # Chart for prediction comparison
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        ax3.bar(predictions.keys(), predictions.values(), color="green")
        ax3.set_ylabel("Predicted Price (₹)")
        ax3.set_title("Model-wise Price Predictions")
        plt.xticks(rotation=45)
        st.pyplot(fig3)

# =========================================================================================
# PAGE 3: MODEL INSIGHTS
# =========================================================================================
elif page == "📈 Model Insights":
    st.title("📈 Model Insights & Comparison")
    st.markdown("### Explore performance, accuracy & prediction difference")

    acc_df = pd.DataFrame(accuracies).T
    st.dataframe(acc_df, use_container_width=True)

    # R² vs MAE scatter plot
    st.subheader("📊 R² vs MAE Scatter Plot")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.scatter([v['R2'] for v in accuracies.values()],
                [v['MAE'] for v in accuracies.values()],
                color="red", s=80)
    for i, name in enumerate(accuracies.keys()):
        ax4.text([v['R2'] for v in accuracies.values()][i],
                 [v['MAE'] for v in accuracies.values()][i],
                 name, fontsize=8)
    ax4.set_xlabel("R² Score")
    ax4.set_ylabel("MAE")
    ax4.set_title("Model Accuracy vs Error")
    st.pyplot(fig4)
