import json
from datetime import datetime
from pathlib import Path

LOG_PATH = Path("logs/predictions.jsonl")


def log_prediction(
    input_data: dict,
    prediction: int,
    probability: float,
) -> None:
    """Log the prediction result to a JSONL file."""

    LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": input_data,
        "prediction": prediction,
        "probability": probability,
    }

    with open(
        LOG_PATH,
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            json.dumps(log_entry) + "\n"
        )
