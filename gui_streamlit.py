import streamlit as st
import pandas as pd
import numpy as np
import pickle  

try:
    try:
        import joblib
        model = joblib.load("diabetes_model.pkl")
        scaler = joblib.load("scaler.pkl")
    except ImportError:
        with open("diabetes_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
            
    feature_names = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]

    st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
    st.title("Prediksi Diabetes dengan Machine Learning")
    st.write("Masukkan data pasien untuk memprediksi kemungkinan diabetes")

    st.subheader("Data Pasien")

    inputs = {}

    col1, col2 = st.columns(2)

    with col1:
        inputs["Pregnancies"] = st.number_input("Pregnancies", min_value=0, step=1)
        inputs["Glucose"] = st.number_input("Glucose", min_value=0, step=1)
        inputs["BloodPressure"] = st.number_input("BloodPressure", min_value=0, step=1)
        inputs["SkinThickness"] = st.number_input("SkinThickness", min_value=0, step=1)

    with col2:
        inputs["Insulin"] = st.number_input("Insulin", min_value=0, step=1)
        inputs["BMI"] = st.number_input("BMI", min_value=0.0, step=0.1, format="%.1f")
        inputs["DiabetesPedigreeFunction"] = st.number_input(
            "DiabetesPedigreeFunction", min_value=0.0, step=0.01, format="%.2f"
        )
        inputs["Age"] = st.number_input("Age", min_value=0, step=1)

    if st.button("Prediksi", type="primary"):
        input_df = pd.DataFrame([inputs])
        
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        
        result = "POSITIF Diabetes" if prediction == 1 else "NEGATIF Diabetes"
        
        if prediction == 1:
            st.error(f"### Hasil: {result}")
        else:
            st.success(f"### Hasil: {result}")
        
        prediction_proba = model.predict_proba(input_scaled)[0]
        st.write(f"Probabilitas diabetes: {prediction_proba[1]:.2%}")
        
        st.subheader("Data yang Dimasukkan:")
        st.dataframe(input_df)
        
except Exception as e:
    st.error(f"An error occurred: {str(e)}")            
    st.error("Please make sure you have the model and scaler files in the correct location.")