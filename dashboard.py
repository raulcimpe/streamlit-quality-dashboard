# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the machine learning model pipeline (including preprocessing steps)
model = joblib.load("lightgbm_model.pkl")

# Define acceptable ranges for each input feature based on training data
feature_ranges = {
    "injection efficiency": (30, 180),
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

# Set up the page settings and title for the app
st.set_page_config(page_title="Quality Prediction Dashboard", layout="centered")
st.title("📊 Quality Prediction Dashboard")
st.markdown("Use this tool to predict the **quality class** of a manufactured product based on input parameters.")

# Input section
st.sidebar.header("🔧 Input Parameters")

input_data = {}
out_of_range_flags = []

# Create number inputs for each feature
for feature, (min_val, max_val) in feature_ranges.items():
    default_val = (min_val + max_val) / 2
    user_val = st.sidebar.number_input(label=feature, value=float(default_val), min_value=float(min_val), max_value=float(max_val))
    input_data[feature] = user_val

# Prediction button
predict_button = st.sidebar.button("🚀 Predict")

if predict_button:
    input_df = pd.DataFrame([input_data])

    # Warn if any input is out of expected range (unlikely with number_input limits, but safe to include)
    for feature, val in input_data.items():
        min_val, max_val = feature_ranges[feature]
        if val < min_val or val > max_val:
            out_of_range_flags.append((feature, val, min_val, max_val))

    if out_of_range_flags:
        st.warning("⚠️ Some input values are outside the expected range:")
        for feature, val, min_v, max_v in out_of_range_flags:
            st.markdown(f"- **{feature}**: {val} (Expected range: {min_v} - {max_v})")

    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    # Display prediction result
    st.subheader("🎯 Predicted Quality Class")
    st.markdown(f"#### `{prediction}` - {quality_labels[prediction]}")

    # Display prediction confidence
    st.subheader("📊 Prediction Confidence")
    categories = ["Waste", "Acceptable", "Target", "Inefficient"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(categories, prediction_proba, color=["red", "orange", "green", "blue"])
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Confidence")
    st.pyplot(fig)

    # Display feature importance
    st.subheader("🔍 Feature Importance")
    feature_importance = model.named_steps["classifier"].feature_importances_
    normalized_importance = feature_importance / np.sum(feature_importance)
    feature_names = list(input_df.columns)
    fig, ax = plt.subplots(figsize=(7, 5))
    sorted_idx = np.argsort(normalized_importance)
    ax.barh(np.array(feature_names)[sorted_idx], normalized_importance[sorted_idx], color="steelblue")
    ax.set_xlabel("Normalized Importance")
    ax.set_title("Normalized Feature Importance in Prediction")
    st.pyplot(fig)
