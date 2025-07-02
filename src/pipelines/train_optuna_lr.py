import os
import pandas as pd
import optuna
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from pathlib import Path
from functools import partial

import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

from src.utils.common import read_yaml
from src.utils.logger import logger
from src.pipelines.data_preprocessing import build_preprocessing_pipeline


def objective(trial, X_train, y_train, X_val, y_val, df, target_column):
    params = {
        "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 1000),
        "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
    }

    preprocessor = build_preprocessing_pipeline(df, target_column)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(class_weight="balanced", **params))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)

    f1 = f1_score(y_val, preds)
    acc = accuracy_score(y_val, preds)
    precision = precision_score(y_val, preds)
    recall = recall_score(y_val, preds)
    roc_auc = roc_auc_score(y_val, preds)

    logger.info(f"Trial {trial.number}: F1={f1:.4f}, Accuracy={acc:.4f}")

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc
        })

    return f1


def train_final_model(best_params, train_df, val_df, target_column, config):
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_val = val_df.drop(columns=[target_column])
    y_val = val_df[target_column]

    preprocessor = build_preprocessing_pipeline(train_df, target_column)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(class_weight="balanced", **best_params))
    ])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_val)
    acc = accuracy_score(y_val, preds)
    precision = precision_score(y_val, preds)
    recall = recall_score(y_val, preds)
    f1 = f1_score(y_val, preds)
    roc_auc = roc_auc_score(y_val, preds)

    logger.info(f"Final metrics — F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc
        })

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name="logreg-optuna-model"
        )
    mlflow.end_run()


def train_optuna_logreg(config_path: Path):
    logger.info("Start Optuna tuning for Logistic Regression")
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("optuna-logreg")

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
        df=train_df,
        target_column=target_column
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective_func, n_trials=30)
    logger.info(f"Best params: {study.best_params}")

    train_final_model(study.best_params, train_df, val_df, target_column, config)


if __name__ == "__main__":
    train_optuna_logreg(Path("src/config/config.yaml"))
