import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.common import read_yaml
from src.utils.logger import logger


def evaluate_model(config_path: Path):
    '''Evaluate the model using the test dataset.'''
    logger.info("Starting model evaluation")
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("model-evaluation")

    config = read_yaml(config_path)
    test_path = Path(config.data_paths.test_data)

    df = pd.read_csv(test_path)
    logger.info("Loaded test set with shape %s", df.shape)

    target_column = "Churn"
    X_test = df.drop(columns=[target_column])
    y_test = df[target_column]

    # Load model from MLflow
    model_uri = f"models:/{config.model_registry.name}/{config.model_registry.stage}"
    logger.info("Trying to load model from MLflow: %s", model_uri)
    model = mlflow.sklearn.load_model(model_uri)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    logger.info("Test Accuracy: %.4f, F1: %.4f, ROC AUC: %.4f", acc, f1, roc_auc)

    with mlflow.start_run():
        mlflow.log_metrics(
            {
                "test_accuracy": acc,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1": f1,
                "test_roc_auc": roc_auc,
            }
        )


if __name__ == "__main__":
    evaluate_model(Path("src/config/config.yaml"))
