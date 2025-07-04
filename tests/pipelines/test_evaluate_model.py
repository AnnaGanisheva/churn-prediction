from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.pipelines.evaluate_model import evaluate_model


@patch("src.pipelines.evaluate_model.mlflow")
@patch("src.pipelines.evaluate_model.read_yaml")
def test_evaluate_model(mock_read_yaml, mock_mlflow, tmp_path):
    mock_read_yaml.return_value = MagicMock(
        data_paths=MagicMock(test_data=str(tmp_path / "test.csv")),
        model_registry=MagicMock(name="test_model", stage="Staging"),
    )

    df = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4],
            "feature2": [10, 20, 30, 40],
            "Churn": [0, 1, 0, 1],
        }
    )
    df.to_csv(tmp_path / "test.csv", index=False)

    mock_model = MagicMock()
    mock_model.predict.return_value = [0, 1, 0, 1]
    mock_mlflow.sklearn.load_model.return_value = mock_model

    mock_run = MagicMock()
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

    evaluate_model(Path("fake_config_path.yaml"))

    mock_model.predict.assert_called_once()
    mock_mlflow.log_metrics.assert_called_once()
