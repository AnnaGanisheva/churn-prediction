import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.pipelines.data_preprocessing import build_preprocessing_pipeline
from src.utils.common import read_yaml
from src.utils.logger import logger


def train():
    logger.info("Starting training pipeline...")

    # Load environment variables
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("test-churn")

    # Load config
    config = read_yaml(Path("src/config/config.yaml"))
    train_path = Path(config.data_paths.train_data)

    # Load data
    df = pd.read_csv(train_path)
    logger.info("Loaded training data: %s", df.shape)

    # Split X/y
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(df, target_column="Churn")

    # Build full model pipeline
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=1000))]
    )

    # Fit
    pipeline.fit(X, y)
    logger.info("Training complete.")

    # Make predictions
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    # Calculate metrics
    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_proba)

    # MLflow logging
    with mlflow.start_run():
        mlflow.log_params(pipeline.named_steps["model"].get_params())
        mlflow.log_metrics(
            {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
            }
        )
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        logger.info("Model logged to MLflow.")


if __name__ == "__main__":
    train()
