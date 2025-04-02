# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the machine learning model pipeline (including preprocessing steps)
model = joblib.load("lightgbm_model.pkl")

# Updated wider feature ranges
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

# Page setup
st.set_page_config(page_title="Quality Prediction Dashboard", layout="centered")
st.title("📊 Quality Prediction Dashboard")
st.markdown("Use this tool to predict the **quality class** of a manufactured product based on input parameters.")

# Input section
st.sidebar.header("🔧 Input Parameters")
input_data = {}
out_of_range_flags = []

# Feature sliders
for feature, (min_val, max_val) in feature_ranges.items():
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
# Predict button
if st.sidebar.button("🚀 Predict"):
    input_df = pd.DataFrame([input_data])

    # Force correct column order and naming
    expected_features = [
        "injection_efficiency",
        "APVs - Specific injection pressure peak value",
        "Mold temperature",
        "Mm - Torque mean value current cycle",
        "Melt temperature",
        "SKx - Closing force",
        "SKs - Clamping force peak value",
        "ZUx - Cycle time"
    ]
    input_df = input_df[expected_features]

    # Check for out-of-range values (optional)
    for feature, val in input_data.items():
        min_val, max_val = feature_ranges[feature]
        if val < min_val or val > max_val:
            out_of_range_flags.append((feature, val, min_val, max_val))

    if out_of_range_flags:
        st.warning("⚠️ Some input values are outside the expected range:")
        for feature, val, min_v, max_v in out_of_range_flags:
            st.markdown(f"- **{feature}**: {val} (Expected range: {min_v} - {max_v})")

    # Prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    # Display prediction result
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
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Confidence")
    st.pyplot(fig)

    # Feature importance
    st.subheader("🔍 Feature Importance")
    classifier = model.named_steps["classifier"]
    feature_importance = classifier.feature_importances_
    normalized_importance = feature_importance / np.sum(feature_importance)
    feature_names = list(input_df.columns)

    sorted_idx = np.argsort(normalized_importance)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.array(feature_names)[sorted_idx], normalized_importance[sorted_idx], color="steelblue")
    ax.set_xlabel("Normalized Importance")
    ax.set_title("Normalized Feature Importance in Prediction")
    st.pyplot(fig)
