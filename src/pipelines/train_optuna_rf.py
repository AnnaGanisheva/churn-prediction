import os
from pathlib import Path
from functools import partial

import pandas as pd
import optuna

from dotenv import load_dotenv
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import mlflow
import mlflow.sklearn

from src.utils.common import read_yaml
from src.utils.logger import logger
from src.pipelines.data_preprocessing import build_preprocessing_pipeline


def objective(trial, X_train, y_train, X_val, y_val, train_df, target_column):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 4, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "random_state": 42,
    }

    preprocessor = build_preprocessing_pipeline(train_df, target_column)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(class_weight="balanced", **params))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)
    f1 = f1_score(y_val, preds)

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("f1", f1)

    return f1


def train_rf_optuna(config_path: Path):
    logger.info("Starting RandomForest Optuna Tuning")
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("optuna-randomforest")

    config = read_yaml(config_path)
    train_df = pd.read_csv(config.data_paths.train_data)
    val_df = pd.read_csv(config.data_paths.val_data)

    target_column = "Churn"
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    X_val = val_df.drop(columns=[target_column])
    y_val = val_df[target_column]

    objective_func = partial(
        objective,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        train_df=train_df,
        target_column=target_column
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective_func, n_trials=30)
    logger.info(f"Best params: {study.best_params}")

    # Final model training
    best_params = study.best_params
    best_params["random_state"] = 42

    final_preprocessor = build_preprocessing_pipeline(train_df, target_column)
    final_pipeline = Pipeline([
        ("preprocessor", final_preprocessor),
        ("model", RandomForestClassifier(class_weight="balanced", **best_params))
    ])

    final_pipeline.fit(X_train, y_train)
    val_preds = final_pipeline.predict(X_val)

    acc = accuracy_score(y_val, val_preds)
    precision = precision_score(y_val, val_preds)
    recall = recall_score(y_val, val_preds)
    f1 = f1_score(y_val, val_preds)
    roc_auc = roc_auc_score(y_val, val_preds)

    logger.info(f"Final metrics on validation set: F1={f1:.4f}, ROC AUC={roc_auc:.4f}")

    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc
        })
        mlflow.sklearn.log_model(final_pipeline, artifact_path="model", registered_model_name="rf-optuna-model")


if __name__ == "__main__":
    train_rf_optuna(Path("src/config/config.yaml"))
