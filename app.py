import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)

st.title("📉 Customer Churn Prediction")
st.write("Predict churn risk and view churn probability.")

# -----------------------------
# Load model & encoder
# -----------------------------
model = joblib.load("models/churn_model.pkl")
encoders = joblib.load("models/encoders.pkl")

# Load data for dropdown values
df = pd.read_csv("data/clean_churn.csv")

st.subheader("🧾 Customer Information")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (Months)", min_value=0)
    monthly_charge = st.number_input("Monthly Charge", min_value=0.0)
    total_charges = st.number_input("Total Charges", min_value=0.0)
    contract = st.selectbox("Contract", df["Contract"].unique())
    payment = st.selectbox("Payment Method", df["Payment Method"].unique())

with col2:
    internet = st.selectbox("Internet Service", df["Internet Service"].unique())
    gender = st.selectbox("Gender", df["Gender"].unique())
    married = st.selectbox("Married", df["Married"].unique())
    dependents = st.number_input("Number of Dependents", min_value=0)

# Create input dataframe
input_df = pd.DataFrame({
    "Tenure in Months": [tenure],
    "Monthly Charge": [monthly_charge],
    "Total Charges": [total_charges],
    "Contract": [contract],
    "Payment Method": [payment],
    "Internet Service": [internet],
    "Gender": [gender],
    "Married": [married],
    "Number of Dependents": [dependents]
})

# Encode categorical inputs
for col in input_df.select_dtypes(include="object"):
    input_df[col] = encoders[col].transform(input_df[col])

st.markdown("---")

if st.button("🔍 Predict Churn"):
    prob = model.predict_proba(input_df)[0][1] * 100

    st.metric("Churn Probability", f"{prob:.2f}%")

    if prob >= 70:
        st.error("⚠️ High Risk: Likely to churn")
    elif prob >= 40:
        st.warning("⚠️ Medium Risk")
    else:
        st.success("✅ Low Risk: Likely to stay")
