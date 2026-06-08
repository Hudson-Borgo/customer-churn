from pathlib import Path

import pandas as pd

from src.utils.config import load_config


def run() -> None:
    config = load_config()

    source_path = Path(config["paths"]["bronze_data"])
    target_path = Path(config["paths"]["silver_data"])

    if not source_path.exists():
        raise FileNotFoundError(f"Input file not found: {source_path}")

    df = pd.read_csv(source_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(target_path, index=False)

    print(f"Ingestion completed. " f"Rows: {len(df)} | " f"Output: {target_path}")


if __name__ == "__main__":
    run()
