import pandas as pd

from pathlib import Path

from src.utils.config import load_config


def drop_identifier_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Drop identifier columns from the DataFrame."""
    return df.drop(
        columns=["customerID"]
    )


def create_target(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create the target variable for churn prediction."""
    df = df.copy()

    df["target"] = (
        df["Churn"]
        .map(
            {
                "No": 0,
                "Yes": 1,
            }
        )
    )

    df = df.drop(
        columns=["Churn"]
    )

    return df

def run() -> None:
    """Run the feature engineering process."""
    config = load_config()

    source_path = Path(
        config["paths"]["validated_data"]
    )

    target_path = Path(
        config["paths"]["gold_data"]
    )

    if not source_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {source_path}"
        )

    df = pd.read_parquet(
        source_path
    )

    df = drop_identifier_columns(df)

    df = create_target(df)

    target_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        target_path,
        index=False,
    )

    print(
        f"Feature engineering completed. "
        f"Rows: {len(df)} | "
        f"Columns: {len(df.columns)} | "
        f"Output: {target_path}"
    )
    
if __name__ == "__main__":
    run()