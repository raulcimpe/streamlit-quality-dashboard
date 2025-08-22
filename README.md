# Streamlit Quality Dashboard (LightGBM)

Interactive dashboard to predict **manufacturing product quality** from process variables and explain the main drivers.  
Built with **Python, scikit-learn, LightGBM, and Streamlit**.

---

## Screenshot

![Dashboard Screenshot](images/Quality-Prediction-Dashboard---Streamlit-04-02-2025_12_18_PM.png)

---

## Features
- Classifies products into **Waste, Acceptable, Target, Inefficient**
- Uses a **LightGBM pipeline** with preprocessing and feature selection
- Interactive **Streamlit interface** for predictions
- Shows **prediction probabilities** and **feature importance**
- Clean project structure for reproducibility

---

## Project Structure
```
app/                # Streamlit app (dashboard.py)
notebooks/          # Jupyter notebooks (EDA, model training)
data/               # Sample dataset (small example only)
images/             # Screenshots for the README
src/                # Helper functions (inference, utils)
lightgbm_model.pkl  # Pretrained LightGBM model
requirements.txt    # Python dependencies
README.md           # Project documentation
LICENSE             # MIT License
```

---

## Quickstart

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it
# Windows:
.venv/Scripts/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app/dashboard.py
```

---

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
