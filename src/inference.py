import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "lightgbm_model.pkl"

def load_model():
    return joblib.load(MODEL_PATH)

def predict(df: pd.DataFrame):
    model = load_model()
    proba = model.predict_proba(df)
    preds = model.predict(df)
    return preds, proba
