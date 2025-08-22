# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# Load the machine learning model pipeline (including preprocessing steps)
model = joblib.load("lightgbm_model.pkl")

# Define acceptable ranges for each input feature based on training data
feature_ranges = {
    "injection efficiency": (75, 155),  # Global range: 77.5 – 152.6
    "APVs - Specific injection pressure peak value": (780, 940),  # Global range: 780.5 – 937.7
    "Mold temperature": (78, 83),  # Global range: 78.4 – 82.1
    "Mm - Torque mean value current cycle": (76, 115),  # Global range: 76.5 – 114.9
    "Melt temperature": (80, 155),  # Global range: 80 – 154.9
    "SKx - Closing force": (878, 933),  # Global range: 878 – 932.8
    "SKs - Clamping force peak value": (894, 947),  # Global range: 894.8 – 946.5
    "ZUx - Cycle time": (74.7, 75.8)  # Global range: 74.7 – 75.79
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



# Example input values for each class in preferred order
example_data = {
    "Class": ["Waste", "Acceptable", "Target", "Inefficient"],
    "injection efficiency": [104.7, 131.0, 141.1, 81.7],
    "APVs - Specific injection pressure peak value": [918.2, 912.0, 883.3, 894.8],
    "Mold temperature": [81.2, 81.2, 81.1, 81.9],
    "Mm - Torque mean value current cycle": [104.7, 104.7, 105.3, 105.3],
    "Melt temperature": [106.0, 106.2, 106.0, 108.8],
    "SKx - Closing force": [899, 912, 894, 902],
    "SKs - Clamping force peak value": [915, 928, 915, 918],
    "ZUx - Cycle time": [74.8, 74.8, 75.7, 75.6]
}

# Create the DataFrame
example_df = pd.DataFrame(example_data)

# Set the 'Class' column as the index
example_df.set_index("Class", inplace=True)

# Display the table before the prediction button
st.markdown("### 🧪 Example Input Values for Each Class")
st.dataframe(example_df.style.format("{:.2f}").set_properties(**{'text-align': 'center'}), use_container_width=True)


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
    # Display feature importance (non-normalized)
    st.subheader("🔍 Feature Importance")
    feature_importance = model.named_steps["classifier"].feature_importances_
    feature_names = list(input_df.columns)

    # Create DataFrame and sort by importance
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importance
    }).sort_values(by="Importance", ascending=False)

    # Plot the importance
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
    ax.set_title("Feature Importance - LightGBM")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    st.pyplot(fig)
