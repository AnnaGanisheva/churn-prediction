import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from src.utils.common import read_yaml
from src.utils.logger import logger


def load_model():
    '''Load the model from the MLflow model registry.'''
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    config = read_yaml(Path("src/config/config.yaml"))
    client = MlflowClient()
    latest_version = client.get_latest_versions(config.model_registry.name)[0].version
    model_uri = f"models:/{config.model_registry.name}/{latest_version}"
    logger.info("Trying to load model from MLflow: %s", model_uri)
    model = mlflow.sklearn.load_model(model_uri)
    return model


def predict(input_data: dict):
    '''Predict churn based on input data.'''
    model = load_model()
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return int(prediction), float(round(probability, 4))
