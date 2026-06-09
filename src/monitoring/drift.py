from pathlib import Path

import pandas as pd

TRAIN_PATH = Path(
    "data/gold/training_dataset.parquet"
)

PREDICTION_LOG_PATH = Path(
    "logs/predictions.jsonl"
)

def load_training_data() -> pd.DataFrame:
    return pd.read_parquet(TRAIN_PATH)

def load_prediction_logs() -> pd.DataFrame:

    return pd.read_json(
        PREDICTION_LOG_PATH,
        lines=True,
    )

def detect_drift():

    training_df = load_training_data()

    prediction_logs = load_prediction_logs()

    production_df = pd.json_normalize(
        prediction_logs["input"]
    )

    numerical_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    for feature in numerical_features:

        training_mean = (
            training_df[feature]
            .mean()
        )

        production_mean = (
            production_df[feature]
            .mean()
        )

        difference = abs(
            production_mean
            - training_mean
        )

        print(f"\nFeature: {feature}")

        print(
            f"Training Mean: "
            f"{training_mean:.2f}"
        )

        print(
            f"Production Mean: "
            f"{production_mean:.2f}"
        )

        print(
            f"Difference: "
            f"{difference:.2f}"
        )

        if difference > 20:
            print(
                "WARNING: Possible drift detected."
            )

if __name__ == "__main__":
    detect_drift()
