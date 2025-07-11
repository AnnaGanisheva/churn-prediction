from unittest.mock import MagicMock, patch

import pandas as pd
from sklearn.impute import SimpleImputer

from src.pipelines.train_optuna_rf import objective


@patch("src.pipelines.train_optuna_rf.mlflow")
@patch("src.pipelines.train_optuna_rf.build_preprocessing_pipeline")
def test_objective_randomforest(mock_build_pipeline, mock_mlflow):
    trial = MagicMock()
    trial.suggest_int.side_effect = [
        100,
        10,
        3,
        2,
    ]  # n_estimators, max_depth, min_split, min_leaf
    trial.suggest_categorical.return_value = "sqrt"

    X_train = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    y_train = pd.Series([0, 1])
    X_val = pd.DataFrame({"a": [1.5, 2.5], "b": [3.5, 4.5]})
    y_val = pd.Series([0, 1])
    train_df = X_train.copy()
    train_df["Churn"] = y_train

    mock_build_pipeline.return_value = SimpleImputer()

    mock_run = MagicMock()
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

    f1 = objective(
        trial, X_train, y_train, X_val, y_val, train_df, target_column="Churn"
    )

    assert isinstance(f1, float)
    assert 0.0 <= f1 <= 1.0
    mock_mlflow.log_params.assert_called_once()
    mock_mlflow.log_metric.assert_called_once_with("f1", f1)
