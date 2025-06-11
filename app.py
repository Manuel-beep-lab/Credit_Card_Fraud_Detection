import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load('xgb_best_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Credit Card Fraud Detector", layout="centered")
st.title("🔍 Credit Card Fraud Detection")
st.markdown("This app uses a trained machine learning model to detect fraudulent credit card transactions.")

# Sample inputs
sample_legit = [-1.359807, -0.072781, 2.536346, 1.378155, -0.338321,
                0.462388, 0.239599, 0.098698, 0.363787, 0.090794,
                -0.551600, -0.617801, -0.991390, -0.311169, 1.468177,
                -0.470401, 0.207971, 0.025791, 0.403993, 0.251412,
                -0.018307, 0.277838, -0.110474, 0.066928, 0.128539,
                -0.189115, 0.133558, -0.021053]

sample_fraud = [1.191857, 0.266151, 0.166480, 0.448154, 0.060018,
                -0.082361, -0.078803, 0.085102, -0.255425, -0.166974,
                1.612726, 1.065235, 0.489095, -0.143772, 0.635558,
                0.463917, -0.114805, -0.183361, -0.145783, -0.069083,
                -0.225775, -0.638672, 0.101288, -0.339846, 0.167170,
                0.125895, -0.008983, 0.014724]

# Input mode
st.markdown("### 🔘 Enter Transaction Details or Use a Sample:")
input_mode = st.radio("Input mode", ["Manual Input", "Use Sample Legit", "Use Sample Fraud"])

if input_mode == "Use Sample Legit":
    V_features = sample_legit
    amount = 2.69
elif input_mode == "Use Sample Fraud":
    V_features = sample_fraud
    amount = 149.62
else:
    V_features = []
    for i in range(1, 29):
        V_features.append(st.number_input(f"V{i}", step=0.01, key=f"v{i}"))
    amount = st.number_input("Transaction Amount", step=0.01, key="amount")

# Prediction
if st.button("Predict"):
    input_data = np.array(V_features + [amount]).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("## 🔎 Prediction Result")
    st.metric("Fraud Probability", f"{probability*100:.2f} %")

    # Gauge (simulated)
    st.markdown("### Risk Level:")
    if probability >= 0.75:
        st.progress(100, text="🔴 Very High Risk")
    elif probability >= 0.5:
        st.progress(75, text="🟠 Moderate Risk")
    elif probability >= 0.25:
        st.progress(50, text="🟡 Slight Risk")
    else:
        st.progress(25, text="🟢 Low Risk")

    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")

# Divider
st.markdown("---")
st.subheader("📁 Upload CSV for Bulk Fraud Detection")
csv_file = st.file_uploader("Upload a CSV file with 29 columns (V1–V28 + Amount):", type=["csv"])

if csv_file is not None:
    try:
        df = pd.read_csv(csv_file)
        if df.shape[1] != 29:
            st.warning("❌ CSV must have 29 columns: V1 to V28 and Amount.")
        else:
            df_scaled = scaler.transform(df.values)
            preds = model.predict(df_scaled)
            probs = model.predict_proba(df_scaled)[:, 1]

            df["Fraud_Prediction"] = preds
            df["Fraud_Probability"] = probs

            st.success("✅ Predictions completed.")
            st.dataframe(df.head())

            # Download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results CSV", data=csv, file_name="fraud_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Error: {e}")
