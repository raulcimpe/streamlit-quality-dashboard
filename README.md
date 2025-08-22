# Streamlit Quality Dashboard (LightGBM)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open in Streamlit](https://img.shields.io/badge/Live-Dashboard-brightgreen)](https://app-quality-dashboard-xzktf8sfvx8s7utc6cid2f.streamlit.app)

This project was developed as part of a data science portfolio to demonstrate how **machine learning models can be integrated into interactive web applications**.  
The goal is to predict the **manufacturing quality class of products** based on process parameters, and to provide **interpretable insights** through a simple dashboard.  

By combining **LightGBM** for classification with **Streamlit** for deployment, the project illustrates how raw industrial data can be transformed into a tool that supports decision-making for production efficiency and quality control.

---

## Demo
Open the app: https://app-quality-dashboard-xzktf8sfvx8s7utc6cid2f.streamlit.app

---

## Screenshot
![Dashboard Screenshot](images/screenshot.png)

---

## Features
- Classifies products into **Waste, Acceptable, Target, Inefficient**  
- Uses a **LightGBM pipeline** with preprocessing and feature selection  
- Interactive **Streamlit interface** for real-time predictions  
- Displays **prediction probabilities** and **feature importance**  
- Clean, modular project structure for reproducibility  

---

## Project Structure
```
streamlit-quality-dashboard/
├── app/                # Streamlit app (dashboard.py)
├── notebooks/          # Jupyter notebooks (EDA, model training)
├── data/               # Sample dataset (small example only)
├── images/             # Screenshots for the README
├── src/                # Helper functions (inference, utils)
├── lightgbm_model.pkl  # Pretrained LightGBM model
├── requirements.txt    # Python dependencies
├── LICENSE             # MIT License
└── README.md           # Project documentation
```

---

## Quickstart

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
```

The app will launch locally at: http://localhost:8501

---

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
