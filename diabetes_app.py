
# diabetes_app.py
import streamlit as st
import numpy as np
import joblib

# Custom CSS styling
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            padding: 10px 24px;
        }
        .stTextInput, .stNumberInput {
            border-radius: 10px;
        }
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #2E86C1;
        }
        .subtitle {
            font-size: 20px;
            color: #2874A6;
        }
    </style>
""", unsafe_allow_html=True)

# Load the model
model = joblib.load("diabetes_model.pkl")

st.markdown('<div class="title">🩺 Diabetes Mellitus Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">By DSA 2025</div>', unsafe_allow_html=True)
st.markdown("---")

st.write("Enter your medical information below to predict whether you're likely to have diabetes.")

# Input fields in columns for better layout
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0.0, step=1.0)
    glucose = st.number_input("Glucose Level", min_value=0.0, step=1.0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, step=1.0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0.0, step=1.0)

with col2:
    insulin = st.number_input("Insulin", min_value=0.0, step=1.0)
    bmi = st.number_input("BMI", min_value=0.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01)
    age = st.number_input("Age", min_value=1.0, step=1.0)

st.markdown("---")

# Predict button
if st.button("Predict Diabetes Status"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]
    result = "🟢 You are likely Non-Diabetic." if prediction == 0 else "🔴 You are likely Diabetic."
    
    st.success(result if prediction == 0 else result)
