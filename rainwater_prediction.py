# -*- coding: utf-8 -*-
"""Streamlit Rainfall Prediction App"""

import streamlit as st
import pandas as pd
import pickle

# =========================================================
# Load trained model
# =========================================================
@st.cache_resource
def load_model():
    """
    Load the saved Random Forest model and feature names.
    """
    with open("rainfall_prediction_model.pkl", "rb") as file:
        model_data = pickle.load(file)
    return model_data["model"], model_data["features_names"]

model, feature_names = load_model()

# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="🌧️ Rainfall Prediction App", layout="centered")
st.title("🌦️ Rainfall Prediction System")
st.write("Predict whether it will **rain or not** based on the weather parameters below.")

# Input fields
st.subheader("Enter Weather Parameters")
col1, col2 = st.columns(2)

with col1:
    pressure = st.number_input("Pressure (hPa)", min_value=900.0, max_value=1100.0, value=1015.9)
    dewpoint = st.number_input("Dew Point (°C)", min_value=0.0, max_value=30.0, value=19.9)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=95.0)
    cloud = st.number_input("Cloud (%)", min_value=0.0, max_value=100.0, value=81.0)

with col2:
    sunshine = st.number_input("Sunshine (hours)", min_value=0.0, max_value=15.0, value=0.0)
    winddirection = st.number_input("Wind Direction (°)", min_value=0.0, max_value=360.0, value=40.0)
    windspeed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=13.7)

# =========================================================
# Prediction
# =========================================================
if st.button("Predict Rainfall ☔"):
    # Prepare input dataframe
    input_df = pd.DataFrame([[pressure, dewpoint, humidity, cloud, sunshine, winddirection, windspeed]],
                            columns=feature_names)
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    result = "🌧️ Rainfall Expected" if prediction == 1 else "☀️ No Rainfall"
    
    # Display result
    st.success(f"Prediction Result: {result}")
    
    # Optional: show input data
    with st.expander("See Input Data"):
        st.dataframe(input_df)

# =========================================================
# Footer
# =========================================================
st.markdown("""
---
Made with ❤️ using **Streamlit** and **Random Forest Classifier**
""")
