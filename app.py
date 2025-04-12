import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine, text
import uuid


DATABASE_URL = "postgresql://diabetes_o37h_user:zfbumBme0orBezURrNuEXvBABG1NxNtz@dpg-cvsku2re5dus7397ebgg-a.oregon-postgres.render.com/diabetes_o37h"


engine = create_engine(DATABASE_URL)


with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS user_predictions (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(100),
            name VARCHAR(100),
            phone VARCHAR(20),
            glucose INTEGER,
            bmi FLOAT,
            blood_pressure INTEGER,
            insulin INTEGER,
            dpf_score FLOAT,
            age INTEGER,
            result VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))

# Load ML model, scaler, and feature order
model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
FEATURE_ORDER = pickle.load(open("feature_order.pkl", "rb"))

# Streamlit UI
st.title("🩺 Diabetes Prediction App")

st.subheader("👤 User Information")
user_id = str(uuid.uuid4())
name = st.text_input("Name")
phone = st.text_input("Phone Number")

st.subheader("🧪 Medical Inputs")
glucose = st.number_input("Glucose Level", 0, 200)
bmi = st.number_input("BMI", 0.0, 50.0)
blood_pressure = st.number_input("Blood Pressure", 0, 150)
insulin = st.number_input("Insulin Level", 0, 300)
age = st.number_input("Age", 1, 100)

st.subheader("👪 Family History")
parents = st.radio("Parents have diabetes?", ["No", "Yes"])
siblings = st.radio("Siblings have diabetes?", ["No", "Yes"])
grandparents = st.radio("Grandparents have diabetes?", ["No", "Yes"])

# DPF score calculation
dpf_score = 0.0
if parents == "Yes": dpf_score += 0.3
if siblings == "Yes": dpf_score += 0.2
if grandparents == "Yes": dpf_score += 0.2

# Predict button
if st.button("🔍 Predict"):
    input_df = pd.DataFrame([[glucose, bmi, blood_pressure, insulin, dpf_score, age]],
                            columns=["Glucose", "BMI", "BloodPressure", "Insulin", "DiabetesPedigreeFunction", "Age"])
    input_df = input_df[FEATURE_ORDER]
    input_scaled = scaler.transform(input_df)
    result = model.predict(input_scaled)[0]
    result_label = "Positive" if result == 1 else "Negative"
    st.success(f"Prediction: **{result_label}**")

    # Insert into DB
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO user_predictions (
                user_id, name, phone, glucose, bmi, blood_pressure, insulin, dpf_score, age, result
            ) VALUES (
                :user_id, :name, :phone, :glucose, :bmi, :blood_pressure, :insulin, :dpf_score, :age, :result
            )
        """), {
            "user_id": user_id,
            "name": name,
            "phone": phone,
            "glucose": glucose,
            "bmi": bmi,
            "blood_pressure": blood_pressure,
            "insulin": insulin,
            "dpf_score": dpf_score,
            "age": age,
            "result": result_label
        })

    st.info("✅ Prediction saved to the database.")


