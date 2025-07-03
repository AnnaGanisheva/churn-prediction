import streamlit as st
from src.inference.predict import predict

st.title("Churn Prediction")

st.markdown("Enter client data:")

# Input parameters
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has Partner?", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", value=70.0)

# Creare DataFrame
input_dict = {
    "customerID": "0000-BGTRT",
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": "No",
    "tenure": tenure,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": monthly_charges,
    "TotalCharges": tenure * monthly_charges  # simple estimation
}


# Precidiction
if st.button("Predict"):
    prediction, probability = predict(input_dict)
    st.markdown(f"### Churn: {'Yes' if prediction else 'No'}")
    st.markdown(f"### Probability: {probability:.2f}")