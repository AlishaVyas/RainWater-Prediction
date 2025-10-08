# -*- coding: utf-8 -*-
"""RainWater_prediction.py"""

# =============================
# Importing the dependencies
# =============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# =============================
# Data Collection and Processing
# =============================
data = pd.read_csv("/content/Rainfall.csv")

# Clean columns
data.columns = data.columns.str.strip()
data = data.drop(columns=["day"])

# Handle missing values
data["winddirection"] = data["winddirection"].fillna(data["winddirection"].mode()[0])
data["windspeed"] = data["windspeed"].fillna(data["windspeed"].median())

# Convert categorical rainfall column
data["rainfall"] = data["rainfall"].map({"yes": 1, "no": 0})

# Drop highly correlated columns
data = data.drop(columns=['maxtemp', 'temparature', 'mintemp'])

# Balance the dataset
df_majority = data[data["rainfall"] == 1]
df_minority = data[data["rainfall"] == 0]
df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)
df_downsampled = pd.concat([df_majority_downsampled, df_minority]).sample(frac=1, random_state=42).reset_index(drop=True)

# Split features and target
x = df_downsampled.drop(columns=["rainfall"])
y = df_downsampled["rainfall"]

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

# =============================
# Model Training with GridSearchCV
# =============================
rf_model = RandomForestClassifier(random_state=42)
param_grid_rf = {
    "n_estimators": [50, 100, 200],
    "max_features": ["sqrt", "log2"],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train, Y_train)

best_rf_model = grid_search_rf.best_estimator_
print("Best parameters for Random Forest:", grid_search_rf.best_params_)

# =============================
# Model Evaluation
# =============================
cv_scores = cross_val_score(best_rf_model, X_train, Y_train, cv=5)
print("Mean CV Score:", np.mean(cv_scores))

y_pred = best_rf_model.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print("Test Accuracy:", accuracy)
print("Classification Report:\n", classification_report(Y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred))

# =============================
# Save Model
# =============================
model_data = {"model": best_rf_model, "features_names": x.columns.tolist()}
with open("rainfall_prediction_model.pkl", "wb") as file:
    pickle.dump(model_data, file)

print("✅ Model saved as rainfall_prediction_model.pkl")

# =========================================================
# STREAMLIT FRONTEND SECTION (for deployment)
# =========================================================

import streamlit as st

# ------------------------------
# Load the trained model
# ------------------------------
with open("rainfall_prediction_model.pkl", "rb") as file:
    model_data = pickle.load(file)

model = model_data["model"]
feature_names = model_data["features_names"]

# ------------------------------
# Streamlit Page Configuration
# ------------------------------
st.set_page_config(page_title="🌧️ Rainfall Prediction App", layout="centered")

st.title("🌦️ Rainfall Prediction System")
st.write("""
Predict whether it will **rain or not** based on the weather conditions.
""")

# ------------------------------
# Input Fields for Features
# ------------------------------
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

# ------------------------------
# Make Prediction
# ------------------------------
if st.button("Predict Rainfall ☔"):
    input_data = pd.DataFrame([[pressure, dewpoint, humidity, cloud, sunshine, winddirection, windspeed]],
                              columns=feature_names)

    prediction = model.predict(input_data)[0]
    result = "🌧️ Rainfall Expected" if prediction == 1 else "☀️ No Rainfall"

    # Display result
    st.success(f"**Prediction Result:** {result}")

    # Optional: Show the input data
    with st.expander("See Input Data"):
        st.dataframe(input_data)

# ------------------------------
# Footer
# ------------------------------
st.markdown("""
---
Made with ❤️ using **Streamlit** and **Random Forest Classifier**
""")
