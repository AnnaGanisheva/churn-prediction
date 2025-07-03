from pathlib import Path

import mlflow
import pandas as pd

from src.utils.common import read_yaml


def load_model():
    '''Load the model from the MLflow model registry.'''
    config = read_yaml(Path("src/config/config.yaml"))
    model_uri = f"models:/{config.model_registry.name}/{config.model_registry.stage}"
    model = mlflow.sklearn.load_model(model_uri)
    return model


def predict(input_data: dict):
    '''Predict churn based on input data.'''
    model = load_model()
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return int(prediction), float(round(probability, 4))
