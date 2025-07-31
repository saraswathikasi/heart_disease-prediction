import streamlit as st
import numpy as np
import joblib

# Load the saved scaler and model
scaler = joblib.load("scaler.pkl")
model = joblib.load("stack_model.pkl")

st.title("💓 Heart Disease Prediction App")

st.write("Please input the following details:")

# Example: Accept 13 inputs as per Heart Disease dataset (customize names as needed)
age = st.number_input("Age", 18, 100, 45)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", 80, 200, 120)
chol = st.number_input("Cholesterol (chol)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Resting ECG results (restecg)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", 60, 250, 150)
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of peak exercise ST segment", [0, 1, 2])
ca = st.selectbox("Number of major vessels (ca)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

if st.button("Predict"):
    # Prepare input
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])

    # Scale it
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    # Output
    if prediction[0] == 1:
        st.error("❌ The model predicts: Heart Disease")
    else:
        st.success("✅ The model predicts: No Heart Disease")

