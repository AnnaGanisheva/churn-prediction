from unittest.mock import MagicMock, patch

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.pipelines.train_model import train


@patch("src.pipelines.train_model.build_preprocessing_pipeline")
@patch("src.pipelines.train_model.mlflow")
@patch("src.pipelines.train_model.read_yaml")
def test_train_model(mock_read_yaml, mock_mlflow, mock_build_preprocessor, tmp_path):
    train_file = tmp_path / "train.csv"
    mock_read_yaml.return_value = MagicMock(
        data_paths=MagicMock(train_data=str(train_file))
    )

    df = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4],
            "feature2": [5, 6, 7, 8],
            "Churn": [0, 1, 0, 1],
        }
    )
    df.to_csv(train_file, index=False)

    mock_build_preprocessor.return_value = StandardScaler()

    mock_run = MagicMock()
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

    train()

    mock_mlflow.log_metrics.assert_called_once()
    mock_mlflow.log_params.assert_called_once()
    mock_mlflow.sklearn.log_model.assert_called_once()
