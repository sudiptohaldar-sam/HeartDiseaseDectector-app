import streamlit as st
import pandas as pd
import joblib

model = joblib.load("SVM_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("Heart Disease Prediction")
st.write("Provide the following details:")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "TA", "NAP", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-induced Angina", ["Y", "N"])
old_peak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):

    # Label Encoding
    sex = 1 if sex == "M" else 0
    exercise_angina = 1 if exercise_angina == "Y" else 0

    # Create DataFrame
    df = pd.DataFrame({
        "Age":[age],
        "Sex":[sex],
        "ChestPainType":[chest_pain],
        "RestingBP":[resting_bp],
        "Cholesterol":[cholesterol],
        "FastingBS":[fasting_bs],
        "RestingECG":[resting_ecg],
        "MaxHR":[max_hr],
        "ExerciseAngina":[exercise_angina],
        "Oldpeak":[old_peak],
        "ST_Slope":[st_slope]
    })

    # One-Hot Encoding
    df = pd.get_dummies(df)

    # Match training columns
    df = df.reindex(columns=expected_columns, fill_value=0)

    # Scale input
    df_scaled = scaler.transform(df)

    # Prediction
    prediction = model.predict(df_scaled)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")