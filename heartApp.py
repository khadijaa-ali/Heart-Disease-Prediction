import streamlit as st
import pandas as pd
import joblib

model = joblib.load("heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
columns = joblib.load("heart_columns.pkl")

# streamlit
st.title("Heart Disease Prediction")
st.write("Enter your health details to check the risk of Heart Attack.")

# form
age = st.number_input("Age", min_value=1, max_value=100, placeholder="Enter your age")
sex_input = st.radio("Select your gender", ["Male", "Female"])

sex = pd.Series([sex_input]).map({"Male": 1, "Female": 0})[0]
cp_input = st.selectbox("Chest Pain Type",
                  ["Typical Angina - chest hurts with activity",
                   "Atypical Angina - chest hurts sometimes",
                   "Non-anginal - little or no chest pain",
                   "Asymptomatic - no chest pain"])
cp_map = {
    "Typical Angina - chest hurts with activity": 0,
    "Atypical Angina - chest hurts sometimes": 1,
    "Non-anginal - little or no chest pain": 2,
    "Asymptomatic - no chest pain": 3
}
cp = cp_map[cp_input]

bp = st.slider("Blood Pressure", 30, 260, 120)
chol = st.slider("Cholesterol", 100, 300, 200)
s_input = st.radio("Sugar", ["Low", "High"])
s = 1 if s_input == "High" else 0
ecg_input = st.selectbox("ECG Result", 
                       ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
ecg_map = {"Normal": 0, "ST-T Abnormality": 1, "Left Ventricular Hypertrophy": 2}
ecg = ecg_map[ecg_input]

rate = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)

input_data = pd.DataFrame([[age, sex, cp, bp, chol, s, ecg, rate]], columns=columns)


input_scaled = scaler.transform(input_data)


# prediction
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("High risk of heart Attack")
    else:
        st.success("Low risk of heart Attack")
