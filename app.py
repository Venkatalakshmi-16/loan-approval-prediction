import streamlit as st
import pickle
import numpy as np

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Loan Approval Prediction System")

st.write("Enter details below (Loan Amount is in thousands)")


# Input Fields with Restrictions


income = st.number_input(
    "Applicant Income",
    min_value=150,
    max_value=81000,
    step=100
)

co_income = st.number_input(
    "Coapplicant Income",
    min_value=0,
    max_value=40000,
    step=100
)

loan_amount = st.number_input(
    "Loan Amount (in thousands)",
    min_value=9,
    max_value=700,
    step=1
)

credit_history = st.selectbox(
    "Credit History",
    [0, 1]
)

# Prediction

if st.button("Predict"):

    total_income = income + co_income

    # Feature order must match training
    input_data = np.array([[
        income,
        co_income,
        loan_amount,
        credit_history,
        total_income
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    if prediction[0] == 1:
        st.success(f"Loan Approved ✅ (Probability: {probability[0][1]*100:.2f}%)")
    else:
        st.error(f"Loan Rejected ❌ (Probability: {probability[0][0]*100:.2f}%)")