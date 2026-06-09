from pathlib import Path

import joblib
import pandas as pd

from src.monitoring.logger import log_prediction
from src.utils.config import load_config


def load_model():
    """Load the trained model from the specified path."""
    config = load_config()
    model_path = Path(config["artifacts"]["model_path"])

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    return joblib.load(model_path)


def predict_churn(input_data: dict) -> dict:
    """Predict customer churn based on input data."""
    model = load_model()

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]


    #Log the prediction result
    result = {
    "prediction": int(prediction),
    "churn_probability": float(probability),
    }

    log_prediction(
    input_data=input_data,
    prediction=result["prediction"],
    probability=result["churn_probability"],
    )

    log_prediction(input_data, prediction, probability)

    return {
        "prediction": int(prediction),
        "churn_probability": float(probability),
    }
