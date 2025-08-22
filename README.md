# Streamlit Quality Dashboard (LightGBM)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open in Streamlit](https://img.shields.io/badge/Live-Dashboard-brightgreen)](https://app-quality-dashboard-xzktf8sfvx8s7utc6cid2f.streamlit.app)

Interactive dashboard to **predict manufacturing product quality** from process variables and to explain the main drivers.  
Built with **Python, scikit-learn, LightGBM, and Streamlit**.

---

## 📖 Demo
👉 [Open Dashboard](https://app-quality-dashboard-xzktf8sfvx8s7utc6cid2f.streamlit.app)

---

## 🖼️ Screenshot
![Dashboard Screenshot](images/dashboard.png)

---

## 🚀 Features
- Classifies products into **Waste, Acceptable, Target, Inefficient**  
- Uses a **LightGBM pipeline** with preprocessing and feature selection  
- Interactive **Streamlit interface** for real-time predictions  
- Displays **prediction probabilities** and **feature importance**  
- Clean, modular project structure for reproducibility  

---


## 📂 Project Structure
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



## ⚙️ Quickstart
```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it
# On Windows
.venv\Scripts\activate
# On Linux / Mac
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app/dashboard.py


---

## 📜 License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
