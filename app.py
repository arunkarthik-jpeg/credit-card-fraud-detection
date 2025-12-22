import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection App")

st.write("""
Upload a CSV file with transaction data  
(Columns: Time, V1–V28, Amount)
""")

MODEL_PATH = "fraud_model_rf.pkl"

# Check model file
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found: fraud_model_rf.pkl")
    st.stop()

# Load model
try:
    model = joblib.load(MODEL_PATH)
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error("❌ Error loading model")
    st.exception(e)
    st.stop()

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Preview of uploaded data")
    st.dataframe(data.head())

    # Drop Class column if present
    if "Class" in data.columns:
        X = data.drop("Class", axis=1)
    else:
        X = data

    try:
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]

        result = data.copy()
        result["Fraud_Prediction"] = preds
        result["Fraud_Probability"] = probs

        st.subheader("Prediction Results")
        st.dataframe(result.head(50))

        fraud_count = (preds == 1).sum()
        normal_count = (preds == 0).sum()

        st.write(f"🚨 Fraudulent transactions: {fraud_count}")
        st.write(f"✅ Normal transactions: {normal_count}")

        csv = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Results",
            csv,
            "fraud_predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error("❌ Prediction failed")
        st.exception(e)

