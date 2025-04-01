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
    "injection efficiency": (50, 150),
    "APVs - Specific injection pressure peak value": (600, 1200),
    "Mold temperature": (60, 90),
    "Mm - Torque mean value current cycle": (10, 25),
    "Melt temperature": (200, 300),
    "SKx - Closing force": (40, 60),
    "SKs - Clamping force peak value": (50, 90),
    "ZUx - Cycle time": (60, 90)
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

# Create a section in the sidebar for entering input values
st.sidebar.header("🔧 Input Parameters")

# Dictionary to hold user input values
input_data = {}
# List to keep track of any inputs outside the expected range
out_of_range_flags = []

# Create a number input for each feature
for feature, (min_val, max_val) in feature_ranges.items():
    default_val = (min_val + max_val) / 2  # Use the midpoint as the default
    val = st.sidebar.number_input(f"{feature}", value=float(default_val))
    input_data[feature] = val

    # Check if the entered value is out of the expected range
    if val < min_val or val > max_val:
        out_of_range_flags.append((feature, val, min_val, max_val))

# Create a button to run the prediction
if st.sidebar.button("🚀 Predict"):
    # Convert the input data into a DataFrame format
    input_df = pd.DataFrame([input_data])

    # Ensure the column order matches the model’s expected input
    expected_features = model.named_steps["preprocessor"].transformers_[0][2]
    input_df = input_df[expected_features]

    # If any inputs are out of range, display a warning
    if out_of_range_flags:
        st.warning("⚠️ Some input values are outside the expected range:")
        for feature, val, min_v, max_v in out_of_range_flags:
            st.markdown(f"- **{feature}**: {val} (Expected range: {min_v} - {max_v})")

    # Run the prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    # ----------------------------
    # Display the prediction probabilities as a bar chart
    # ----------------------------
    st.subheader("📊 Prediction Confidence")

    # Define shorter labels for display
    categories = ["Waste", "Acceptable", "Target", "Inefficient"]
    proba_mapping = dict(zip(categories, prediction_proba))

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(categories, prediction_proba, color=['red', 'orange', 'green', 'blue'])

    # Show the probability value above each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.2f}", 
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Confidence")
    st.pyplot(fig)

    # ----------------------------
    # Show feature importance
    # ----------------------------
    st.subheader("🔍 Feature Importance")
    
    # Get feature importance values from the model
    feature_importance = model.named_steps['classifier'].feature_importances_
    
    # Normalize them to show as proportions
    normalized_importance = feature_importance / np.sum(feature_importance)
    feature_names = list(input_df.columns)

    # Plot feature importance as a horizontal bar chart
    fig, ax = plt.subplots(figsize=(7, 5))
    sorted_idx = np.argsort(normalized_importance)
    ax.barh(np.array(feature_names)[sorted_idx], normalized_importance[sorted_idx], color="steelblue")
    ax.set_xlabel("Normalized Importance")
    ax.set_title("Normalized Feature Importance in Prediction")
    st.pyplot(fig)
