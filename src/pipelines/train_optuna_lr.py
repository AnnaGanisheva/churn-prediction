import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from src.pipelines.data_preprocessing import build_preprocessing_pipeline
from src.utils.common import train_model
from src.utils.logger import logger


def objective(trial, X_train, y_train, X_val, y_val, train_df, target_column):
    params = {
        "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 1000),
        "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
    }

    preprocessor = build_preprocessing_pipeline(train_df, target_column)
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(class_weight="balanced", **params)),
        ]
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)

    f1 = f1_score(y_val, preds)
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("f1", f1)

    return f1


if __name__ == "__main__":
    logger.info("Train logistig regression optuna")
    train_model("optuna-logreg", objective, "logistig regression")
