from pathlib import Path

import pandas as pd

from src.utils.config import load_config

REQUIRED_COLUMNS = [
    "customerID",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def validate_empty_dataset(df: pd.DataFrame) -> None:
    """Check if the dataset is empty."""
    if df.empty:
        raise ValueError("Dataset is empty.")


def validate_required_columns(df: pd.DataFrame) -> None:
    """Check if all required columns are present in the dataset."""
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def validate_duplicate_customers(df: pd.DataFrame) -> None:
    """Check if there are any duplicate customer IDs."""
    duplicate_count = df["customerID"].duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(f"Found {duplicate_count} duplicated customer IDs.")


def validate_total_charges(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Check if TotalCharges can be converted to numeric and handle missing values,
    following the business logic."""

    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    valid_missing_mask = (df["tenure"] == 0) & (df["TotalCharges"].isna())

    df.loc[valid_missing_mask, "TotalCharges"] = (
        0  # Set TotalCharges to 0 for customers with tenure 0 and missing TotalCharges,
    )
    # because its a new client who has not been charged yet.

    remaining_nulls = df["TotalCharges"].isna().sum()

    if remaining_nulls > 0:
        raise ValueError(f"Found {remaining_nulls} invalid TotalCharges values.")

    print(f"Replaced {valid_missing_mask.sum()} " "missing TotalCharges values.")
    return df


def validate_churn_values(df: pd.DataFrame) -> None:
    """Check if Churn column contains only valid values."""
    valid_values = {"Yes", "No"}

    invalid_values = set(df["Churn"].unique()) - valid_values

    if invalid_values:
        raise ValueError(f"Invalid churn values: {invalid_values}")


def run() -> None:
    """Run the data validation process."""
    config = load_config()

    source_path = Path(config["paths"]["silver_data"])

    target_path = Path(config["paths"]["validated_data"])

    if not source_path.exists():
        raise FileNotFoundError(f"Input file not found: {source_path}")

    df = pd.read_parquet(source_path)

    validate_empty_dataset(df)

    validate_required_columns(df)

    validate_duplicate_customers(df)

    df = validate_total_charges(df)

    validate_churn_values(df)

    target_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        target_path,
        index=False,
    )

    print(f"Validation completed. " f"Rows: {len(df)} | " f"Output: {target_path}")


if __name__ == "__main__":
    run()
