import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from src.pipelines.data_preprocessing import build_preprocessing_pipeline
from src.utils.common import train_model
from src.utils.logger import logger


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
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(class_weight="balanced", **params)),
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
    logger.info("Train Random forest optuna")
    train_model("optuna-randomforest", objective, "randomforest")
