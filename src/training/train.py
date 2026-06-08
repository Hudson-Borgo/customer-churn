from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.config import load_config


def load_dataset(config: dict) -> pd.DataFrame:
    """Load the training dataset."""
    dataset_path = Path(config["paths"]["gold_data"])

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    return pd.read_parquet(dataset_path)


def split_data(df: pd.DataFrame, random_state: int):
    """Split the dataset into train, validation, and test sets."""
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=random_state,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=random_state,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """Build preprocessing and model pipeline."""
    categorical_features = X_train.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numerical_features = X_train.select_dtypes(
        exclude=["object", "string"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "numerical",
                StandardScaler(),
                numerical_features,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )


def train_model(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series):
    """Train the model."""
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Evaluate the model and return accuracy."""
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions))

    return accuracy


def save_model(model: Pipeline, model_path: Path) -> None:
    """Save the trained model locally."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def run() -> None:
    """Run the training pipeline."""
    config = load_config()

    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run():
        df = load_dataset(config)

        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        ) = split_data(df, config["training"]["random_state"])

        pipeline = build_pipeline(X_train)
        model = train_model(pipeline, X_train, y_train)

        test_accuracy = evaluate_model(model, X_test, y_test)

        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("random_state", config["training"]["random_state"])
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("validation_rows", len(X_val))
        mlflow.log_param("test_rows", len(X_test))

        mlflow.log_metric("test_accuracy", test_accuracy)

        model_path = Path(config["artifacts"]["model_path"])
        save_model(model, model_path)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )


if __name__ == "__main__":
    run()
