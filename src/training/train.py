from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.utils.config import load_config

def load_dataset(config: dict) -> pd.DataFrame:
    dataset_path = Path(config["paths"]["gold_data"])

    return pd.read_parquet(dataset_path)

def split_data(
    df: pd.DataFrame,
    random_state: int,
):
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

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )
    
    
def build_pipeline(X_train):
    categorical_features = X_train.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_features = X_train.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features,
            ),
            (
                "numerical",
                "passthrough",
                numerical_features,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000
                ),
            ),
        ]
    )

    return pipeline

def train_model(
    pipeline,
    X_train,
    y_train,
):
    pipeline.fit(
        X_train,
        y_train,
    )

    return pipeline

def evaluate_model(
    model,
    X_test,
    y_test,
):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print(
        f"Test Accuracy: {accuracy:.4f}"
    )

    print(
        classification_report(
            y_test,
            predictions,
        )
    )

def save_model(
    model,
    model_path: Path,
):
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        model_path,
    )

def run():

    config = load_config()

    df = load_dataset(config)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(
        df,
        config["training"]["random_state"],
    )

    pipeline = build_pipeline(
        X_train
    )

    model = train_model(
        pipeline,
        X_train,
        y_train,
    )

    evaluate_model(
        model,
        X_test,
        y_test,
    )

    save_model(
        model,
        Path(
            config["artifacts"]["model_path"]
        ),
    )

if __name__ == "__main__":
    run()