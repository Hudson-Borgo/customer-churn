from pathlib import Path

import joblib
import pandas as pd

from src.utils.config import load_config


def load_model():
    config = load_config()
    model_path = Path(config["artifacts"]["model_path"])

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    return joblib.load(model_path)


def predict_churn(input_data: dict) -> dict:
    model = load_model()

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "prediction": int(prediction),
        "churn_probability": float(probability),
    }