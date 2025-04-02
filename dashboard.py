# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the machine learning model pipeline (including preprocessing steps)
model = joblib.load("lightgbm_model.pkl")

# Get the exact column names the model expects
expected_features = model.named_steps["preprocessor"].transformers_[0][2]

# Define extended ranges for input sliders
feature_ranges = {
    "injection_efficiency": (30, 180),
    "APVs - Specific injection pressure peak value": (500, 1400),
    "Mold temperature": (40, 120),
    "Mm - Torque mean value current cycle": (5, 35),
    "Melt temperature": (160, 350),
    "SKx - Closing force": (25, 75),
    "SKs - Clamping force peak value": (30, 110),
    "ZUx - Cycle time": (40, 110)
}

# Define what each quality label means
quality_labels = {
    1: "Waste: Product fails to meet basic standards and must be scrapped.",
    2: "Acceptable: Product meets minimum quality standards but is not ideal.",
    3: "Target: Product meets the desired quality specifications.",
    4: "Inefficient: Product is above acceptable but falls short of target quality due to process inefficiencies."
}

# Set up page configuration
st.set_page_config(page_title="Quality Prediction Dashboard", layout="centered")
st.title("📊 Quality Prediction Dashboard")
st.markdown("Use this tool to predict the **quality class** of a manufactured product based on input parameters.")

# Sidebar for input sliders
st.sidebar.header("🔧 Input Parameters")
input_data = {}

# Input sliders for each feature in expected order
for feature in expected_features:
    min_val, max_val = feature_ranges[feature]
    default_val = (min_val + max_val) / 2
    user_val = st.sidebar.number_input(
        label=feature,
        value=float(default_val),
        min_value=float(min_val),
        max_value=float(max_val),
        step=1.0,
        format="%.2f"
    )
    input_data[feature] = user_val

# Predict button
if st.sidebar.button("🚀 Predict"):
    # Ensure correct DataFrame structure and order
    input_df = pd.DataFrame([input_data])[expected_features]

    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    # Display predicted class
    st.subheader("🎯 Predicted Quality Class")
    st.markdown(f"### `Class {prediction}` – {quality_labels[prediction]}")

    # Show prediction probabilities
    st.subheader("📊 Prediction Confidence")
    categories = ["Waste", "Acceptable", "Target", "Inefficient"]
    colors = ["red", "orange", "green", "blue"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(categories, prediction_proba, color=colors)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.2f}", ha="center", va="bottom")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Confidence")
    st.pyplot(fig)

    # Show feature importance
    st.subheader("🔍 Feature Importance")
    importances = model.named_steps["classifier"].feature_importances_
    norm_importance = importances / np.sum(importances)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sorted_idx = np.argsort(norm_importance)
    ax2.barh(np.array(expected_features)[sorted_idx], norm_importance[sorted_idx], color="steelblue")
    ax2.set_xlabel("Normalized Importance")
    ax2.set_title("Feature Importance")
    st.pyplot(fig2)
